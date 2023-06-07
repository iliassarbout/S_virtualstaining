FROM python:3.7-slim

RUN mkdir -p /app
WORKDIR /app

RUN apt-get update 

RUN apt-get install -y \
    python3-openslide \
    libjpeg-dev \
    zlib1g-dev \
    libvips \
    libvips-tools \
    gcc
    
RUN curl -s https://packagecloud.io/install/repositories/cytomine-uliege/Cytomine-python-client/script.python.sh | bash
    
COPY . /app

RUN pip install -r requirements.txt


ENTRYPOINT []
