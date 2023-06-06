FROM python:3.7-slim


RUN apt-get update && apt-get install -y \
    python3-openslide \
    libjpeg-dev \
    zlib1g-dev \
    libvips \
    libvips-tools \
    gcc
    
RUN curl -s https://packagecloud.io/install/repositories/cytomine-uliege/Cytomine-python-client/script.python.sh | bash

RUN pip install -r requirements.txt


ENTRYPOINT ["python", "app.py"]
