from cytomine import CytomineJob
from cytomine.models import ImageInstance
from extraction_utils import extraction
from cytomine import Cytomine
from cytomine.models import StorageCollection, Project, UploadedFile
import os

project_id = 137597


def main(argv):
    with CytomineJob.from_cli(argv) as cj:
                host = cj._parameters.cytomine_host
                public_key = cj._parameters.cytomine_public_key
                private_key = cj._parameters.cytomine_private_key
                image_id = cj._parameters.image_id


                image = ImageInstance().fetch(image_id)
                image_dl = image.download(dest_pattern="IMAGE.ndpi",override=True)
                if not image_dl:
                    print('Error fetching the image. Did you provide a valide image ID ? ')
                # Will print the parameters with their values
                destination_folder_name = str(image_id)
                extraction(he_slide_path='IMAGE.ndpi',real_wsi_id = destination_folder_name,gpu_ids = [])


                image_directory = './sample_staining/slides/'+destination_folder_name+'/'
                image_filenames = os.listdir(image_directory)

                image_paths = [image_directory + i for i in image_filenames]
                uploader = Cytomine(host, public_key, private_key)

                print(image_paths)
                for filename in image_paths:
                    # Create a new image instance for each image file

                    #id_storage = int(str(image_id) + str(k))

                    project = Project().fetch(project_id)

                    storages = StorageCollection().fetch()
                    my_storage = next(filter(lambda storage: storage.user == uploader.current_user.id, storages))
                    if not my_storage:
                                raise ValueError("Storage not found")
                    id_storage = my_storage.id
                    uploaded_file = uploader.upload_image(upload_host=host,
                                                              filename=filename,
                                                              id_storage=id_storage,
                                                              id_project=project_id)
if __name__ == "__main__":
    import sys
    print('Starting')
    print(sys.argv[1:])
    main(sys.argv[1:])