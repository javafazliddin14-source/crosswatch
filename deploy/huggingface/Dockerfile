# Website + live demo on a Hugging Face Space (Docker SDK, free CPU tier).
# Built by deploy/huggingface/publish.py, which copies the repo next to this file.
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user PATH=/home/user/.local/bin:$PATH PYTHONUNBUFFERED=1 \
    YOLO_OFFLINE=1 YOLO_CONFIG_DIR=/tmp/Ultralytics PORT=7860
WORKDIR /home/user/app

# CPU-only torch keeps the image small (the demo runs on CPU)
RUN pip install --no-cache-dir --user torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cpu
COPY --chown=user requirements.txt website/requirements.txt ./reqs/
RUN pip install --no-cache-dir --user -r reqs/requirements.txt fastapi==0.141.1 uvicorn==0.53.0 python-multipart==0.0.32

COPY --chown=user . .
EXPOSE 7860
CMD ["python", "-m", "website.server"]
