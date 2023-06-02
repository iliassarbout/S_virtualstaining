from scipy.ndimage import binary_fill_holes, binary_dilation, binary_erosion
from skimage.measure import label, regionprops
import torchvision.transforms as transforms
from models.combogan_model_pred import ComboGANModel
import torch
from imantics import Polygons, Mask, BBox
from matplotlib.patches import Rectangle
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from openslide import OpenSlide
from collections import Counter
from openslide import lowlevel
from natsort import natsorted
from PIL import Image
from glob import glob
import pandas as pd
import numpy as np
from tqdm import trange
import math
import cv2
import os
import pyvips
import shutil

from utils import *

#GPUS
gpu_ids = []

# Patch size in pixels (x40) 
patch_size = 2048

# Choosing the magnification level
slide_dim_lvl = 5

# Slide ID
real_wsi_id = 'sample_vs'

# Path to all HE slides version 'HE+CD8'
he_slide_path  = './HE.ndpi'

# Path to save data
path_to_save_patches = os.path.join('sample_staining', 'he_patches', real_wsi_id)
os.makedirs(path_to_save_patches ,exist_ok=True)

# Path to save data
save_WSIs = os.path.join('sample_staining', 'slides', real_wsi_id)
os.makedirs(save_WSIs ,exist_ok=True)

os.makedirs(os.path.join('sample_staining', 'GAN', real_wsi_id) ,exist_ok=True) 

# Opening the slides
he_slide  = OpenSlide(he_slide_path)
low_slide = lowlevel.open(he_slide_path)
keys = lowlevel.get_property_names(low_slide)

# Getting slides level dimensions
he_slide_levels = he_slide.level_dimensions


scale_factor = math.ceil(he_slide_levels[0][0]/he_slide_levels[slide_dim_lvl][0])
new_patch_size = math.ceil(patch_size/ scale_factor)

# Getting the thumbnail for the slides
he_thm  = he_slide.read_region((0, 0), slide_dim_lvl, he_slide_levels[slide_dim_lvl])

clean_label_image, polygons = cluster_wsi(he_thm, plot=False)


for poly_id in range(len(polygons.points)):
    tmp_bn_img = np.zeros(np.shape(clean_label_image))
    rr, cc = np.where(clean_label_image==int(len(polygons.points)-poly_id))
    tmp_bn_img[rr,cc] = 1

    # plt.plot(polygons.points[poly_id][:,0], polygons.points[poly_id][:,1], label='Slide region: '+str(poly_id))
    xmin, xmax = np.min(polygons.points[poly_id][:,0]), np.max(polygons.points[poly_id][:,0])
    ymin, ymax = np.min(polygons.points[poly_id][:,1]), np.max(polygons.points[poly_id][:,1])
    # plt.gca().add_patch(Rectangle((xmin, ymin), xmax-xmin, ymax-ymin,edgecolor='gray', facecolor='none', lw=1, ls="--"))

    x_patches = math.ceil((xmax - xmin)/ new_patch_size)
    y_patches = math.ceil((ymax - ymin)/ new_patch_size)

    for y_id in range(1, math.ceil(y_patches)+1):
        for x_id in range(1, math.ceil(x_patches)+1):
            tmp_x_corr, tmp_y_corr = xmin+((x_id-1)*new_patch_size), ymin+((y_id-1)*new_patch_size)
            if (tmp_bn_img[tmp_y_corr:tmp_y_corr+new_patch_size, tmp_x_corr:tmp_x_corr+new_patch_size].sum()/ new_patch_size**2) >0.05:

                path_he_thm  = he_slide.read_region((int(tmp_x_corr*scale_factor), int(tmp_y_corr*scale_factor)), 0, (patch_size, patch_size))

                path_he_thm.save(os.path.join(path_to_save_patches, f'tile_{patch_size}_{int(tmp_x_corr*scale_factor)}_{int(tmp_y_corr*scale_factor)}.tif'))
                # print(x_id-1,y_id-1)


model = ComboGANModel(which_epoch=885, save_dir=os.path.join('./checkpoints', 'stain_aligned_comGan-512'),gpu_ids = gpu_ids)

def get_transform():
    transform_list = []
    transform_list.append(transforms.Resize(512, Image.BICUBIC))
    transform_list += [transforms.ToTensor(),
                       transforms.Normalize((0.5, 0.5, 0.5),
                                            (0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list)


# Get tiles paths
path_tiles = natsorted(glob(os.path.join(path_to_save_patches, '*.tif')))

transform = get_transform()
for path in path_tiles:
    he_img = Image.open(path).convert('RGB')

    he_img = transform(he_img)

    bundle = {'A': torch.tensor(np.expand_dims(he_img, axis=0)), 'DA':
              torch.tensor([8]), 'path': [path]}
    model.set_input(bundle)
    model.test()
    visuals = model.get_current_visuals(testing=True)
    
    short_path = os.path.basename(path)
    name = os.path.splitext(short_path)[0]

    for label, image_numpy in visuals.items():
        image_name = '%s_%s.tif' % (name, label)
        save_path = os.path.join('sample_staining', 'GAN', real_wsi_id,image_name)
        Image.fromarray(image_numpy).save(save_path)

    

for stain_id_fake in trange(0, 9):


    # Get tiles paths
    path_tiles = natsorted(glob(os.path.join('sample_staining', 'GAN', real_wsi_id, '*'+str(stain_id_fake)+'.tif')))

    # Getting the x rows and y columns
    _, patch_size, x_rows, y_colms, stain, modality =os.path.splitext(os.path.basename(path_tiles[len(path_tiles)-1]))[0].split('_')

    x_mins, y_mins = [], []
    for path in path_tiles:
        # Getting the x rows and y columns
        _, patch_size, x_min, y_min, stain, modality =os.path.splitext(os.path.basename(path))[0].split('_')

        x_mins.append(int(x_min))
        y_mins.append(int(y_min))


    all_slide = np.ones((np.max(y_mins)-np.min(y_mins)+int(patch_size), np.max(x_mins)-np.min(x_mins)+int(patch_size), 3))*255

    for path in path_tiles:
        # Getting the x rows and y columns
        _, patch_size, x_min, y_min, stain, modality =os.path.splitext(os.path.basename(path))[0].split('_')

        x_min      = int(x_min) - np.min(x_mins)
        y_min      = int(y_min) - np.min(y_mins)
        patch_size = int(patch_size)

        tmp_array = np.array(Image.open(path).resize((2048, 2048)))[:,:,0:3]


        all_slide[y_min:y_min+patch_size, x_min:x_min+patch_size, :] = tmp_array

    pyramid = pyvips.Image.new_from_array(all_slide)

    del all_slide
    # Add metadata
    # metadata = {'ResolutionUnit': 'micrometers',
    #             'openslide.mpp-x': float(lowlevel.get_property_value(low_slide,keys[108])),
    #             'openslide.mpp-y': float(lowlevel.get_property_value(low_slide,keys[109])),
    #             'openslide.objective-power': float(lowlevel.get_property_value(low_slide,keys[110])),
    #             }shutil
    # tiff.Model: C13210
    # tiff.ResolutionUnit: centimeter
    # tiff.Software: NDP.scan 3.2.15
    # tiff.XResolution: 45344
    # tiff.YResolution: 45344
    # slide-associated-images: macro

    # pyramid.set_type(pyvips.GValue.gint_type, "openslide.objective-power", metadata['openslide.objective-power'])
    # pyramid.set_type(pyvips.GValue.gdouble_type,    "openslide.mpp-x", metadata['openslide.mpp-x'])
    # pyramid.set_type(pyvips.GValue.gdouble_type,    "openslide.mpp-y", metadata['openslide.mpp-y'])
    # pyramid.set_type(pyvips.GValue.gint_type,    "tiff.XResolution", 45344)
    # pyramid.set_type(pyvips.GValue.gint_type,    "tiff.YResolution", 45344)
    # pyramid.set_type(pyvips.GValue.gstr_type,    "tiff.ResolutionUnit", "centimeter")
    # pyramid.set_type(pyvips.GValue.gstr_type,    "tiff.Model", "C13210")
    # pyramid.set_type(pyvips.GValue.gstr_type,    "tiff.Make", "Hamamatsu")
    # pyramid.set_type(pyvips.GValue.gint_type, "hamamatsu.SourceLens",40)

    pyramid.tiffsave(os.path.join(save_WSIs, 'WSI_'+stain+'_'+modality+'.tif'), 
                compression="jpeg", 
                Q=100, 
                tile=True, 
                tile_width=512, 
                tile_height=512, 
                pyramid=True)
    
    
    del pyramid



try:
    shutil.rmtree(path_to_save_patches)
    shutil.rmtree(os.path.join('sample_staining', 'GAN', real_wsi_id))
    print("Temp folders has been deleted.")
except OSError as e:
    print(f"Error: {e.strerror}")
