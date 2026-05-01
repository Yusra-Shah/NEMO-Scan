"""
PneumoScan - Inference Engine
Location: core/inference/engine.py

Central engine that:
  1. Loads all 7 trained model weight files
  2. Preprocesses uploaded X-ray images
  3. Runs MobileNetV3 immediately for instant result
  4. Runs remaining 5 voting models sequentially (thread-safe)
  5. Runs AttentionCNN separately for Grad-CAM heatmap
  6. Returns complete diagnosis dictionary to the GUI

Output dictionary structure (GUI reads this exactly):
{
    "prediction":    "PNEUMONIA" | "NORMAL",
    "confidence":    float (0.0 - 1.0),
    "severity":      "None" | "Mild" | "Moderate" | "Severe",
    "subtype":       "Bacterial" | "Viral" | None,
    "model_votes":   { model_name: float, ... },
    "ensemble_prob": float,
    "heatmap_path":  str | None
}
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import numpy as np
import cv2
import os
import time
import threading
from pathlib import Path
from PIL import Image
from torchvision import transforms
from typing import Dict, Optional, Callable

from core.inference.attention_arch import AttentionCNN


# ---- Model registry ----
MODEL_REGISTRY = {
    'mobilenetv3': {
        'timm_name':   'mobilenetv3_large_100',
        'weight_file': 'mobilenetv3_best.pth',
        'weight':      0.10,
        'img_size':    224,
    },
    'resnet50': {
        'timm_name':   'resnet50',
        'weight_file': 'resnet50_best.pth',
        'weight':      0.18,
        'img_size':    224,
    },
    'densenet121': {
        'timm_name':   'densenet121',
        'weight_file': 'densenet121_best.pth',
        'weight':      0.25,
        'img_size':    224,
    },
    'efficientnet_b4': {
        'timm_name':   'efficientnet_b4',
        'weight_file': 'efficientnet_b4_best.pth',
        'weight':      0.20,
        'img_size':    380,
    },
    'vit_b16': {
        'timm_name':   'vit_base_patch16_224',
        'weight_file': 'vit_b16_best.pth',
        'weight':      0.17,
        'img_size':    224,
    },
    'inception_v3': {
        'timm_name':   'inception_v3',
        'weight_file': 'inception_v3_best.pth',
        'weight':      0.10,
        'img_size':    299,
    },
}

ATTENTION_CNN_FILE = 'attention_cnn_best.pth'

SEVERITY_THRESHOLDS = [
    (0.75, 'Severe'),
    (0.55, 'Moderate'),
    (0.30, 'Mild'),
    (0.00, 'None'),
]

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


def get_severity(pneumonia_prob: float) -> str:
    for threshold, label in SEVERITY_THRESHOLDS:
        if pneumonia_prob >= threshold:
            return label
    return 'None'


def get_subtype(pneumonia_prob: float, model_votes: Dict) -> Optional[str]:
    if pneumonia_prob < 0.50:
        return None
    pneumo_votes = [v for v in model_votes.values() if v > 0.50]
    avg = sum(pneumo_votes) / len(pneumo_votes) if pneumo_votes else 0.0
    return 'Bacterial' if avg > 0.80 else 'Viral'


def preprocess_image(image_path: str, img_size: int = 224) -> torch.Tensor:
    """
    Loads and preprocesses an image for inference.
    Returns tensor of shape (1, 3, img_size, img_size).
    """
    img = Image.open(image_path).convert('RGB')
    tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return tf(img).unsqueeze(0)


def load_timm_model(timm_name: str, weight_path: str, device: torch.device) -> nn.Module:
    """
    Loads a timm model with trained weights from checkpoint.
    Checkpoint format: {'model_name': ..., 'state_dict': ..., ...}
    """
    checkpoint = torch.load(weight_path, map_location=device)
    state_dict = checkpoint['state_dict']

    model = timm.create_model(timm_name, pretrained=False, num_classes=2)
    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)
    return model


def load_attention_cnn(weight_path: str, device: torch.device) -> AttentionCNN:
    """Loads the custom AttentionCNN from checkpoint."""
    checkpoint = torch.load(weight_path, map_location=device)
    state_dict = checkpoint['state_dict']

    model = AttentionCNN(num_classes=2)
    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)
    return model


class InferenceEngine:
    """
    Central inference engine for PneumoScan.

    All model calls are sequential and protected by a lock.
    This guarantees correct BatchNorm behavior and deterministic results.

    Usage:
        engine = InferenceEngine()
        engine.load_models('weights/lung')                      # float32
        engine.load_models('weights/lung/float16', 'float16')  # float16 (GPU)

        quick  = engine.predict_quick('path/to/xray.jpg')
        result = engine.predict_full('path/to/xray.jpg', 'outputs/heatmaps')
    """

    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.voting_models: Dict[str, nn.Module] = {}
        self.attention_cnn: Optional[AttentionCNN] = None
        self.models_loaded = False
        self.precision = 'float32'

        # Single lock serializes ALL model forward passes.
        # This is the fix: no two forward passes can overlap.
        self._inference_lock = threading.Lock()

        print(f'InferenceEngine initialized on {self.device}')

    def load_models(self, weights_dir: str, precision: str = 'float32') -> Dict[str, bool]:
        """
        Loads all model weights from the weights directory.

        Args:
            weights_dir: Path to directory containing .pth files.
            precision:   'float32' (default, desktop-safe) or 'float16'
                         (halves VRAM/RAM; voting models run in fp16 while
                         AttentionCNN stays fp32 for Grad-CAM gradient stability).

        Returns dict of {model_name: loaded_successfully}.
        """
        self.precision = precision
        weights_dir = Path(weights_dir)
        status = {}

        print(f'Loading model weights (precision={precision})...')

        for name, cfg in MODEL_REGISTRY.items():
            weight_path = weights_dir / cfg['weight_file']
            if weight_path.exists():
                try:
                    model = load_timm_model(
                        cfg['timm_name'],
                        str(weight_path),
                        self.device,
                    )
                    if precision == 'float16':
                        model.half()
                    self.voting_models[name] = model
                    status[name] = True
                    print(f'  Loaded {name}')
                except Exception as e:
                    status[name] = False
                    print(f'  Failed {name}: {e}')
            else:
                status[name] = False
                print(f'  Missing {name}: {weight_path}')

        attn_path = weights_dir / ATTENTION_CNN_FILE
        if attn_path.exists():
            try:
                # AttentionCNN always stays float32: Grad-CAM gradients can
                # underflow to zero in float16, producing blank heatmaps.
                self.attention_cnn = load_attention_cnn(str(attn_path), self.device)
                status['attention_cnn'] = True
                print(f'  Loaded attention_cnn (float32 for Grad-CAM)')
            except Exception as e:
                status['attention_cnn'] = False
                print(f'  Failed attention_cnn: {e}')
        else:
            status['attention_cnn'] = False
            print(f'  Missing attention_cnn')

        loaded_count = sum(status.values())
        print(f'Models loaded: {loaded_count}/{len(status)}')
        self.models_loaded = loaded_count > 0
        return status

    def _run_single_model(self, name: str, image_path: str) -> float:
        """
        Runs a single voting model and returns P(Pneumonia).

        CRITICAL: model.eval() is called explicitly every time before
        the forward pass. torch.no_grad() prevents any gradient state
        from accumulating. The inference lock prevents thread overlap.
        """
        img_size = MODEL_REGISTRY[name]['img_size']
        tensor = preprocess_image(image_path, img_size).to(self.device)
        if self.precision == 'float16':
            tensor = tensor.half()
        model = self.voting_models[name]

        with self._inference_lock:
            model.eval()                      # force eval mode every time
            with torch.no_grad():
                logits = model(tensor)
                probs  = F.softmax(logits.float(), dim=1)  # float32 for softmax stability
                return probs[0, 1].item()     # P(Pneumonia)

    def predict_quick(self, image_path: str) -> Dict:
        """
        Runs MobileNetV3 only and returns an instant result.
        Called by the GUI immediately after image upload.
        Full ensemble result follows in a background thread.
        """
        if 'mobilenetv3' not in self.voting_models:
            return {'error': 'MobileNetV3 not loaded'}

        prob = self._run_single_model('mobilenetv3', image_path)
        prediction = 'PNEUMONIA' if prob >= 0.50 else 'NORMAL'

        return {
            'prediction':  prediction,
            'confidence':  round(prob if prediction == 'PNEUMONIA' else 1.0 - prob, 4),
            'model':       'mobilenetv3',
            'probability': round(prob, 4),
        }

    def predict_full(
        self,
        image_path: str,
        heatmaps_dir: Optional[str] = None,
        progress_callback: Optional[Callable[[str, float], None]] = None,
    ) -> Dict:
        """
        Runs all voting models sequentially and generates Grad-CAM heatmap.
        Returns the complete diagnosis dictionary.

        Models run one at a time. This is intentional: on CPU, sequential
        execution is actually faster than threaded because there is no GIL
        contention, no BatchNorm corruption, and no shared state race conditions.

        Args:
            image_path:        Path to the uploaded X-ray image
            heatmaps_dir:      Directory to save Grad-CAM overlay image
            progress_callback: Optional callback(model_name, probability)
                               called as each model completes
        """
        if not self.voting_models:
            return {'error': 'No models loaded'}

        model_votes = {}

        # Sequential execution: one model at a time, deterministic results
        for name in self.voting_models:
            try:
                prob = self._run_single_model(name, image_path)
                model_votes[name] = prob
                if progress_callback:
                    progress_callback(name, prob)
            except Exception as e:
                print(f'Model {name} failed during inference: {e}')
                model_votes[name] = 0.5  # neutral fallback

        # Weighted ensemble
        total_weight = sum(
            MODEL_REGISTRY[n]['weight']
            for n in model_votes
            if n in MODEL_REGISTRY
        )
        ensemble_prob = sum(
            model_votes[n] * MODEL_REGISTRY[n]['weight']
            for n in model_votes
            if n in MODEL_REGISTRY
        )
        if total_weight > 0:
            ensemble_prob /= total_weight

        prediction = 'PNEUMONIA' if ensemble_prob >= 0.50 else 'NORMAL'
        confidence = ensemble_prob if prediction == 'PNEUMONIA' else 1.0 - ensemble_prob
        severity   = get_severity(ensemble_prob)
        subtype    = get_subtype(ensemble_prob, model_votes)

        heatmap_path = None
        if self.attention_cnn is not None and heatmaps_dir is not None:
            try:
                heatmap_path = self._generate_heatmap(image_path, heatmaps_dir)
            except Exception as e:
                print(f'Heatmap generation failed: {e}')

        return {
            'prediction':    prediction,
            'confidence':    round(confidence, 4),
            'severity':      severity,
            'subtype':       subtype,
            'model_votes':   {k: round(v, 4) for k, v in model_votes.items()},
            'ensemble_prob': round(ensemble_prob, 4),
            'heatmap_path':  heatmap_path,
        }

    def _generate_heatmap(self, image_path: str, heatmaps_dir: str) -> str:
        """
        Generates a Grad-CAM heatmap using AttentionCNN.
        Runs under the inference lock to prevent state corruption
        from overlapping with any voting model calls.
        """
        os.makedirs(heatmaps_dir, exist_ok=True)

        with self._inference_lock:
            model = self.attention_cnn
            model.eval()

            # Fresh tensor with grad enabled for backprop
            tensor = preprocess_image(image_path, 224).to(self.device)
            tensor.requires_grad_(True)

            target_layer = model.get_gradcam_target_layer()

            activations = []
            gradients   = []

            def forward_hook(module, input, output):
                activations.append(output.detach())

            def backward_hook(module, grad_input, grad_output):
                gradients.append(grad_output[0].detach())

            fh = target_layer.register_forward_hook(forward_hook)
            bh = target_layer.register_backward_hook(backward_hook)

            output = model(tensor)
            model.zero_grad()
            output[0, 1].backward()  # class 1 = Pneumonia

            fh.remove()
            bh.remove()

        # Compute CAM outside the lock (pure numpy, no model state)
        acts  = activations[0].squeeze(0)
        grads = gradients[0].squeeze(0)

        weights = grads.mean(dim=(1, 2))
        cam     = (weights[:, None, None] * acts).sum(0)
        cam     = F.relu(cam)

        cam = cam.cpu().numpy()
        if cam.max() > 0:
            cam = cam / cam.max()

        original = cv2.imread(image_path)
        if original is None:
            pil_img = Image.open(image_path).convert('RGB')
            original = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

        h, w = original.shape[:2]
        cam_resized     = cv2.resize(cam, (w, h))
        cam_uint8       = (cam_resized * 255).astype(np.uint8)
        heatmap_colored = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)
        overlay         = cv2.addWeighted(original, 0.55, heatmap_colored, 0.45, 0)

        filename  = f'heatmap_{int(time.time())}.jpg'
        save_path = os.path.join(heatmaps_dir, filename)
        cv2.imwrite(save_path, overlay)

        return save_path