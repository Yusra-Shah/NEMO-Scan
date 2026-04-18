import torch

checkpoint = torch.load(
    r'D:\PneumoScan\Pneumo\weights\lung\densenet121_best.pth',
    map_location='cpu'
)
print('Model name:     ', checkpoint['model_name'])
print('Val accuracy:   ', checkpoint['val_accuracy'])
print('Test accuracy:  ', checkpoint['test_accuracy'])
print('Class mapping:  ', checkpoint['class_to_idx'])
print('State dict keys (first 3):', list(checkpoint['state_dict'].keys())[:3])
print('File is valid.')