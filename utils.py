from scipy.ndimage import binary_fill_holes, binary_dilation, binary_erosion
from skimage.measure import label, regionprops
from imantics import Polygons, Mask, BBox
from sklearn.cluster import KMeans
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
import cv2



def reg_slides(reg_he_thm, reg_stain_thm, plot=True):
    def get_centroid_difference(im1_gray, im2_gray):
        def centeroidnp(arr):
            count = (arr == 1).sum()
            # x_center, y_center = 
            return np.argwhere(arr==1).sum(0)/count
        
        bn_1 = np.array(im1_gray<(3*np.max(im1_gray)/4))*1
        bn_2 = np.array(im2_gray<(3*np.max(im1_gray)/4))*1

        y1 ,x1 = centeroidnp(bn_1)
        y2 ,x2 = centeroidnp(bn_2)

        return x2-x1, y2-y1

    im1_gray, im2_gray = np.array(reg_he_thm.convert('L')), np.array(reg_stain_thm.convert('L'))

    rr, cc = np.where(im1_gray == 0)
    im1_gray[rr, cc] = 255

    rr, cc = np.where(im2_gray == 0)
    im2_gray[rr, cc] = 255

    # Find size of image1
    sz = im1_gray.shape
    
    # Define the motion model
    warp_mode = cv2.MOTION_TRANSLATION # MOTION_AFFINE #MOTION_EUCLIDEAN  #MOTION_HOMOGRAPHY   #MOTION_TRANSLATION
    
    # Define 2x3 or 3x3 matrices and initialize the matrix to identity
    if warp_mode == cv2.MOTION_HOMOGRAPHY : warp_matrix = np.eye(3, 3, dtype=np.float32)
    else : warp_matrix = np.eye(2, 3, dtype=np.float32)
    
    # Initialization to help the registration speed
    x_init, y_init =  get_centroid_difference(im1_gray, im2_gray)
    warp_matrix[0][2], warp_matrix[1][2] = x_init, y_init

    # Specify the number of iterations.
    number_of_iterations = 5000
    
    # Specify the threshold of the increment
    # in the correlation coefficient between two iterations
    termination_eps = 1e-10
    
    # Define termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations,  termination_eps)
    
    # Run the ECC algorithm. The results are stored in warp_matrix.
    (cc, warp_matrix) = cv2.findTransformECC (im1_gray,im2_gray,warp_matrix, warp_mode, criteria)
    
    if warp_mode == cv2.MOTION_HOMOGRAPHY :
        # Use warpPerspective for Homography
        im2_aligned = cv2.warpPerspective (np.array(reg_stain_thm), warp_matrix, (sz[1],sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
    else :
        # Use warpAffine for Translation, Euclidean and Affine
        im2_aligned = cv2.warpAffine(np.array(reg_stain_thm), warp_matrix, (sz[1],sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP);
    
    # Defining the X and Y shifts
    reg_x_shift, reg_y_shift = warp_matrix[0][2], warp_matrix[1][2]

    if plot:
        plt.figure()
        plt.imshow(np.array(reg_stain_thm))
        plt.show()

        # Show final results
        plt.figure()
        plt.imshow(np.array(reg_he_thm))
        plt.show()

        plt.figure()
        plt.imshow(im2_aligned)
        plt.show()

        print('X init : ',x_init,' , Y init : ',y_init)
        print('Transformation matrix : \n',warp_matrix)


    # plt.figure()
    # plt.imshow((0.5*im2_aligned)+(0.5*np.array(he_thm)))
    # plt.show()
    return reg_x_shift, reg_y_shift


def cluster_wsi(cluster_he_thm, plot=True):

    def is_img_RGBA(img):
        if   np.shape(img)[2] == 3: return False
        elif np.shape(img)[2] == 4: return True

    # Converting PIL thumbnails of the slides into numpy arrays (RGBA to RGB)
    if is_img_RGBA(cluster_he_thm): np_he_thm = np.array(cluster_he_thm)[:,:,0:3]
    else: np_he_thm = np.array(cluster_he_thm)
    rr, cc, dd = np.where(np_he_thm ==0)
    np_he_thm[rr,cc,dd]=255

    # Reshaping the image into 2D numpy img
    (h,w,c) = np_he_thm.shape
    np_he_thm_2D = np_he_thm.reshape(h*w,c)

    # Apply K-Means clustering to remove the slide's background
    kmeans_model = KMeans(n_clusters=2) # (add , n_init='auto' if you use sklearn> 0.24.2)we shall retain only 2 colors (background vs foreground)
    cluster_labels = kmeans_model.fit_predict(np_he_thm_2D)
    labels_count = Counter(cluster_labels)

    # Identification of the cluster RGB colors
    rgb_cols = kmeans_model.cluster_centers_.round(0).astype(int)

    # Creating a quantified image
    img_quant = np.reshape(rgb_cols[cluster_labels],(h,w,c))

    # Selecting the foreground cluster
    foreground_color = np.min(rgb_cols, axis=0)

    # Creating the binary image 
    bn_img = np.zeros((np.shape(img_quant)[0], np.shape(img_quant)[1]))
    rr, cc, _ = np.where(img_quant==foreground_color)
    bn_img[rr, cc]= 1
    
    if plot:
        plt.figure()
        plt.imshow(img_quant)
        plt.title("KMeans (background/foreground)")
        plt.show()

    # Mathematical morphology to clean up the foreground
    # bn_img = binary_dilation(bn_img, iterations=4)
    bn_img = binary_fill_holes(bn_img)
    # bn_img = binary_erosion(bn_img, iterations=4)

    # label image regions
    label_image = label(bn_img)

    # Store the area of all the labeled regions 
    area_list = []
    for region in regionprops(label_image): area_list.append(region.area)

    # Getting the mean area of the labeled regions
    mean_area = np.mean(area_list)

    # Create empty numpy array to store the clean mask
    bn_img = np.zeros((np.shape(img_quant)[0], np.shape(img_quant)[1]))

    # Filleting out the small regions
    for region in regionprops(label_image):
        if region.area>=mean_area:
            rr, cc = np.where(label_image== region.label)
            bn_img[rr, cc] = 1

    # label image regions
    clean_label_image = label(bn_img)

    # plt.figure()
    # plt.imshow(bn_img)
    # plt.title("Background/foreground")
    # plt.show()

    if plot:
        plt.figure()
        plt.imshow(clean_label_image)
        plt.title("Clean label slide regions")
        plt.show()

    polygons = Mask(bn_img).polygons()

    return clean_label_image, polygons

