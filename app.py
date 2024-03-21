from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from cytomine import CytomineJob
from cytomine.models import ImageInstance

from extraction_utils import extraction
from cytomine import Cytomine

from cytomine.models import StorageCollection, Project, UploadedFile, ImageInstanceCollection, Job

import os
import sys



import logging
import sys
from argparse import ArgumentParser

import os

from cytomine import Cytomine
from cytomine.models import StorageCollection, Project



def upload(filename,host,public_key,private_key,upload_host,id_project):
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
                try: #to modif, when launching from cytomine current_user does not own storage ??? (stopiteration)
                    my_storage = next(filter(lambda storage: storage.user == cytomine.current_user.id, storages))
                    storage_id = my_storage.id
                    #if not my_storage:
                        #raise ValueError("Storage not found")
                except:
                    storage_id=80

                try:
                    uploaded_file = cytomine.upload_image(upload_host=upload_host,
                                                              filename=filename,
                                                              id_storage=storage_id,
                                                              id_project=id_project)
                    print(uploaded_file)
                except:
                    pass
                


def main(argv):
    print(argv)
    
    print(os.getcwd())
    #os.chdir('/app')
    #print(os.getcwd())
    #print('ok')
    upload_host = 'http://localhost-upload'
    with CytomineJob.from_cli(argv) as cj:
                cj.job.update(status=Job.RUNNING, progress=0, statusComment="Image loading...")
                host = cj._parameters.cytomine_host
                public_key = cj._parameters.cytomine_public_key
                private_key = cj._parameters.cytomine_private_key
                image_id = cj._parameters.image_id
                project_id = cj.parameters.cytomine_id_project


                image = ImageInstance().fetch(image_id)
                image_name = image.originalFilename.split('.')[0] #image needs format NAME.extension

                list_of_current_images = [i.instanceFilename for i in ImageInstanceCollection().fetch_with_filter("project", project_id)._data]
                if image_name+'_WSI_fake_0.tif' in list_of_current_images:#to modify, need to check every stain
                    cj.job.update(status=Job.RUNNING, progress=100, statusComment="Image stains were already computed, check images within the project !")
                    return()
                image_dl = image.download(dest_pattern=os.path.join(os.getcwd(),'IMAGE.ndpi'),override=False)
                if not image_dl:
                    print('Error fetching the image. Did you provide a valide image ID ? ')
                # Will print the parameters with their values
                destination_folder_name = str(image_id)
                extraction(he_slide_path=os.path.join(os.getcwd(),'IMAGE.ndpi'),real_wsi_id = destination_folder_name,gpu_ids = [],path_app='/app',cj=cj,original_image_name=image_name) #path app is absolute path in singularity image of the app files


                image_directory = './sample_staining/slides/'+destination_folder_name+'/'
                image_filenames = os.listdir(image_directory)

                image_paths = [image_directory + i for i in image_filenames]
                uploader = Cytomine(host, public_key, private_key)

                print(image_paths)
                for filename in image_paths:
                    upload(filename,host,public_key,private_key,upload_host,project_id)

if __name__ == "__main__":
    
    print('Starting')
    print(sys.argv[1:])
    main(sys.argv[1:])
