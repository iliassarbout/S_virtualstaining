from cytomine import Cytomine
from cytomine.models import StorageCollection, Project
import os


def upload_img(host,public_key,private_key,upload_host,id_project,filename):

    with Cytomine(host=host, public_key=public_key, private_key=private_key) as cytomine:
    
            # Check that the file exists on your file system
            if not os.path.exists(filename):
                raise ValueError("The file you want to upload does not exist")
    
            # Check that the given project exists
            if id_project:
                project = Project().fetch(id_project)
                if not project:
                    raise ValueError("Project not found")
    
            # To upload the image, we need to know the ID of your Cytomine storage.
            storages = StorageCollection().fetch()
            my_storage = next(filter(lambda storage: storage.user == cytomine.current_user.id, storages))
            if not my_storage:
                raise ValueError("Storage not found")
    
            uploaded_file = cytomine.upload_image(upload_host=upload_host,
                                                  filename=filename,
                                                  id_storage=my_storage.id,
                                                  id_project=id_project)
    
            print(uploaded_file)