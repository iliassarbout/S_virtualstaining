FROM python:3.8-slim

RUN mkdir -p /app
WORKDIR /app

RUN apt-get update 

RUN apt-get update && apt-get install -y \
    python3-openslide \
    libjpeg-dev \
    zlib1g-dev \
    libvips \
    libvips-tools \
    gcc \
    ffmpeg \
    libsm6 \
    libxext6
    
COPY . /app

RUN curl -s https://packagecloud.io/install/repositories/cytomine-uliege/Cytomine-python-client/script.python.sh | bash
RUN pip install --upgrade pip setuptools wheel
RUN pip install --upgrade pip
RUN pip install torch==1.13.1+cpu torchvision==0.14.1+cpu -f https://download.pytorch.org/whl/torch_stable.html
RUN pip install -r requirements.txt


ENTRYPOINT ["python","/app/app.py"]
