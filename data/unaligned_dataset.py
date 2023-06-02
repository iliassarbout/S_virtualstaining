import os.path, glob
import torchvision.transforms as transforms
from data.base_dataset import BaseDataset, get_transform, get_params
from data.image_folder import make_dataset
from PIL import Image
from natsort import natsorted
import random
import numpy as np

class UnalignedDataset(BaseDataset):
    def __init__(self, opt):
        super(UnalignedDataset, self).__init__()
        self.opt = opt

        datapath = os.path.join(opt.dataroot, opt.phase+'*')
        self.dirs = natsorted(glob.glob(datapath))

        self.paths = [natsorted(make_dataset(d)) for d in self.dirs]
        self.sizes = [len(p) for p in self.paths]

        print(self.sizes )

    def load_image(self, dom, idx):
        path = self.paths[dom][idx]
        AB_img = Image.open(path).convert('RGB')
        
        # split AB image into A and B
        w, h = AB_img.size
        w2 = int(w / 2)
        A = AB_img.crop((0, 0, w2, h))
        B = AB_img.crop((w2, 0, w, h))

        # apply the same transform to both A and B
        transform_params = get_params(self.opt, A.size)

        A_transform = get_transform(self.opt, transform_params)
        B_transform = get_transform(self.opt, transform_params)

        A = A_transform(A)
        B = B_transform(B)

        return A, B, path

    def __getitem__(self, index):
        he_a = True
        if not self.opt.isTrain:
            if self.opt.serial_test:
                for d,s in enumerate(self.sizes):
                    if index < s:
                        DA = d; break
                    index -= s
                index_A = index
            else:
                DA = index % len(self.dirs)
                index_A = random.randint(0, self.sizes[DA] - 1)
        else:
            # Choose two of our domains to perform a pass on
            DA = random.randint(0, len(self.dirs))

            if DA == len(self.dirs):
                DA = random.randint(0, len(self.dirs)-1)
                he_a = False
            
            index_A = random.randint(0, self.sizes[DA] - 1)
        
        if he_a:
            A_img, B_img, A_path = self.load_image(DA, index_A)
            bundle = {'A': A_img, 'DA': len(self.dirs), 'path': A_path}
        
        else:
            B_img, A_img, A_path = self.load_image(DA, index_A)
            bundle = {'A': A_img, 'DA': DA, 'path': A_path}

        if self.opt.isTrain:
            if he_a:
                bundle.update( {'B': B_img, 'DB': DA, 'path_B': A_path} )
            else:
                bundle.update( {'B': B_img, 'DB': len(self.dirs), 'path_B': A_path} )

        return bundle

    def __len__(self):
        if self.opt.isTrain:
            return max(self.sizes)
        return sum(self.sizes)

    def name(self):
        return 'UnalignedDataset'
