FROM python:3.7-slim

RUN mkdir -p /app
WORKDIR /app

RUN apt-get update && apt-get install -y \
    python3-openslide \
    libjpeg-dev \
    zlib1g-dev \
    libvips \
    libvips-tools \
    gcc
    
RUN curl -s https://packagecloud.io/install/repositories/cytomine-uliege/Cytomine-python-client/script.python.sh | bash
    
COPY . /app

RUN pip install -r requirements.txt

RUN python app.py --cytomine_host https://cytomine-staging.icm-institute.org --cytomine_public_key 41f0fdd9-5ce2-4e33-ae62-06b22395318c --cytomine_private_key bc21ca11-1e11-4547-95fa-f0f480e1cf25 --cytomine_id_project 137597 --cytomine_id_software 577492 --image_id 142507

ENTRYPOINT ["python", "app.py", \
            "--cytomine_host", "https://cytomine-staging.icm-institute.org", \
            "--cytomine_public_key", "41f0fdd9-5ce2-4e33-ae62-06b22395318c", \
            "--cytomine_private_key", "bc21ca11-1e11-4547-95fa-f0f480e1cf25", \
            "--cytomine_id_project", "137597", \
            "--cytomine_id_software", "577492", \
            "--image_id", "142507"]
