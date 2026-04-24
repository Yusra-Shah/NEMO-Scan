FROM python:3.12-slim

RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements_web.txt .
RUN pip install --no-cache-dir -r requirements_web.txt

COPY . .

RUN mkdir -p outputs/heatmaps outputs/reports static

EXPOSE 7860

CMD ["uvicorn", "web_app:app", "--host", "0.0.0.0", "--port", "7860"]
