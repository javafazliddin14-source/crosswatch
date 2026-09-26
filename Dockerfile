# Evaluation image:  docker build -t crosswatch .  &&  docker run --gpus all -v /data/test:/data/test crosswatch
# Website + demo:    docker run --gpus all -p 8000:8000 crosswatch python -m website.server
FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime
WORKDIR /app
ENV PYTHONUNBUFFERED=1 YOLO_OFFLINE=1
COPY requirements.txt website/requirements.txt ./reqs/
RUN pip install --no-cache-dir -r reqs/requirements.txt fastapi==0.141.1 uvicorn==0.53.0 python-multipart==0.0.32
COPY . .
RUN sh weights/download.sh
CMD ["python", "run_submission.py", "--videos", "/data/test", "--out", "/data/predictions.json", "--team", "crosswatch"]
