import os
from pathlib import Path

import numpy as np
import torch 
from cellpose import models
from skimage.io import imread, imsave
from skimage.transform import resize, rescale

def read_files(eval_data_path, train_data_path, img_filename_dapi, img_filename_gfp):

    root_name = img_filename_dapi.split('DAPI Channel')[0]
    potential_seg_name = root_name + '_seg.tiff' #Path(img_filename).stem+'_seg.tiff' #+Path(img_filename).suffix
    potential_class_name = root_name + '_classes.tiff' #Path(img_filename).stem+'_classes.tiff' #+Path(img_filename).suffix

    # check if the image is in the uncurated folder
    if os.path.exists(os.path.join(eval_data_path, img_filename_dapi)):
        in_eval = True
        img_dapi = imread(os.path.join(eval_data_path, img_filename_dapi))
        img_gfp = imread(os.path.join(eval_data_path, img_filename_gfp))

        # check if object segmentation and class masks are in the folder and read them if they are there  
        if os.path.exists(os.path.join(eval_data_path, potential_seg_name)):
            seg = imread(os.path.join(eval_data_path, potential_seg_name))
        else: seg = None
        if os.path.exists(os.path.join(eval_data_path, potential_class_name)):
            classes = imread(os.path.join(eval_data_path, potential_class_name))
        else: classes = None

    # if not it will be in the curated folder
    else: 
        img_dapi = imread(os.path.join(train_data_path, img_filename_dapi))
        img_gfp = imread(os.path.join(train_data_path, img_filename_gfp))

        # check if object segmentation and class masks are in the folder and read them if they are there 
        if os.path.exists(os.path.join(train_data_path, potential_seg_name)):
            seg = imread(os.path.join(train_data_path, potential_seg_name))
        else: seg = None
        if os.path.exists(os.path.join(train_data_path, potential_class_name)):
            classes = imread(os.path.join(train_data_path, potential_class_name))
        else: classes = None

    return img_dapi, img_gfp, seg, classes, in_eval, root_name

def get_channel_files(eval_data_path, train_data_path, cur_selected_img):

    if os.path.exists(os.path.join(eval_data_path, cur_selected_img)):
        in_eval = True
    else: in_eval = False

    if 'GFP Channel' in cur_selected_img:
        gfp_img = cur_selected_img
        name_split = gfp_img.split('GFP Channel')[0]
        if in_eval:
            dapi_img = [file for file in os.listdir(eval_data_path) if name_split in file and 'DAPI Channel' in file][0]
        else:
            dapi_img = [file for file in os.listdir(train_data_path) if name_split in file and 'DAPI Channel' in file][0]

    else: 
        dapi_img = cur_selected_img
        name_split = dapi_img.split('DAPI Channel')[0]
        if in_eval:
            gfp_img = [file for file in os.listdir(eval_data_path) if name_split in file and 'GFP Channel' in file][0]
        else:
            gfp_img = [file for file in os.listdir(train_data_path) if name_split in file and 'GFP Channel' in file][0]

    return dapi_img, gfp_img

def get_predictions(eval_data_path, threshold_gfp, accepted_types):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device=="cuda":
        model = models.Cellpose(gpu=True, model_type="cyto")
    else:
        model = models.Cellpose(gpu=False, model_type="cyto")

    list_files_dapi = [file for file in os.listdir(eval_data_path) if Path(file).suffix in accepted_types and 'DAPI Channel' in file]
    list_files_gfp = [file for file in os.listdir(eval_data_path) if Path(file).suffix in accepted_types and 'GFP Channel' in file]
    list_files_dapi.sort()
    list_files_gfp.sort()

    for idx, img_filename in enumerate(list_files_dapi):
        # don't do this for segmentations in the folder
        #extend to check the prefix also matches an existing image
        #seg_name = Path(self.img_filename).stem+'_seg'+Path(self.img_filename).suffix
        if '_seg' in img_filename or '_classes' in img_filename:  continue

        else:
            img = imread(os.path.join(eval_data_path, img_filename)) #DAPI
            img_gfp = imread(os.path.join(eval_data_path, list_files_gfp[idx]))
            orig_size = img.shape
            # get root filename without DAPI Channel ending
            name_split = img_filename.split('DAPI Channel')[0]
            seg_name = name_split + '_seg.tiff' #Path(img_filename).stem+'_seg.tiff' #+Path(img_filename).suffix
            class_name = name_split + '_classes.tiff' #Path(img_filename).stem+'_classes.tiff' #+Path(img_filename).suffix

            if Path(img_filename).suffix in (".tiff", ".tif") and len(orig_size)==3:
                warn_flag = '3d'
                height, width = orig_size[1], orig_size[2]
                
                max_dim = max(height, width)
                rescale_factor = max_dim/512
                img = rescale(img, 1/rescale_factor, channel_axis=0)
                mask, _, _, _ = model.eval(img, z_axis=0) #for 3D
                mask = merge_2d_labels(mask)
                mask = resize(mask, (orig_size[0], height, width), order=0)
                
                # get labels of object segmentation
                labels = np.unique(mask)[1:]
                class_mask = np.copy(mask)
                class_mask[mask != 0] = 1 # set all objects to class 1
                
                # if the mean of an object in the gfp image is abpve the predefined threshold set it to class 2
                for l in labels:
                    mean_l = np.mean(img_gfp[mask == l])
                    if mean_l > threshold_gfp:
                        class_mask[mask == l] = 2                   
                
                imsave(os.path.join(eval_data_path, seg_name), mask)
                imsave(os.path.join(eval_data_path, class_name), class_mask)
            else: 
                warn_flag = 'exit'

            return warn_flag

def merge_2d_labels(mask):
    inc = 100 
    for idx in range(mask.shape[0]-1):
        slice_cur = mask[idx] # get current slice
        labels_cur = list(np.unique(slice_cur))[1:] # and a list of the object labels in this slice

        mask[idx+1] += inc # increase the label values of the next slices so that we dont have different objects with same labels
        mask[idx+1][mask[idx+1]==inc] = 0 # make sure that background remains black
        inc += 100 # and increase inc by 100 so in the next slice there is no overlap again

        # for each label in the current slices
        for label_cur in labels_cur:
            # get a patch around the object
            x, y = np.where(slice_cur==label_cur)
            max_x, min_x = np.max(x), np.min(x)
            max_y, min_y = np.max(y), np.min(y)
            # and extract this patch in the next slice
            slice_patch_next = mask[idx+1, min_x:max_x, min_y:max_y]
            # get the labels within this patch
            labels_next = np.unique(slice_patch_next)
            
            # if there are none continue
            if labels_next.shape[0]==0: continue
            elif labels_next[0]==0 and labels_next.shape[0]==1: continue
            # if there is only one label(not the background) set it to value of label in current slice
            elif labels_next[0]!=0 and labels_next.shape[0]==1: 
                slice_next = mask[idx+1]
                slice_next[slice_next==labels_next[0]] = label_cur
                mask[idx+1] = slice_next
                continue
            # and if there are multiple
            if labels_next[0]==0 and labels_next.shape[0]>1: 
                labels_next = labels_next[1:]
            # pick the object with the largest area 
            obj_sizes = [np.where(slice_patch_next==l2)[0].shape[0] for l2 in labels_next]
            idx_max = obj_sizes.index(max(obj_sizes))
            # replace the found label in the next slice with the one in the current slice
            label2replace = labels_next[idx_max]
            slice_next = mask[idx+1]
            slice_next[slice_next==label2replace] = label_cur
            mask[idx+1] = slice_next

    # reorder the labels in the mask so they are continuous
    unique = list(np.unique(mask))[1:] 
    for idx, l in enumerate(unique):
        mask[mask==l] = idx+1
    # and convert to unit8 if we have fewer than 256 labels
    if len(unique)<256:
        mask = mask.astype(np.uint8)
    return mask
'''
def changeWindow(w1, w2):
    w1.hide()
    w2.show()

def circularity(perimeter, area):
    """Calculate the circularity of the region

    Parameters
    ----------
    perimeter : float
        the perimeter of the region
    area : float
        the area of the region

    Returns
    -------
    circularity : float
        The circularity of the region as defined by 4*pi*area / perimeter^2
    """
    circularity = 4 * np.pi * area / (perimeter ** 2)

    return circularity

def make_bbox(bbox_extents):
    """Get the coordinates of the corners of a
    bounding box from the extents

    Parameters
    ----------
    bbox_extents : list (4xN)
        List of the extents of the bounding boxes for each of the N regions.
        Should be ordered: [min_row, min_column, max_row, max_column]

    Returns
    -------
    bbox_rect : np.ndarray
        The corners of the bounding box. Can be input directly into a
        napari Shapes layer.
    """
    minr = bbox_extents[0]
    minc = bbox_extents[1]
    maxr = bbox_extents[2]
    maxc = bbox_extents[3]

    bbox_rect = np.array(
        [[minr, minc], [maxr, minc], [maxr, maxc], [minr, maxc]]
    )
    bbox_rect = np.moveaxis(bbox_rect, 2, 0)

    return bbox_rect
'''

'''
# to convert segmentation of classes into bboxes 

print('bbbbbb', classes.shape)
properties = regionprops_table(classes, properties=('label', 'bbox', 'area'))
print('AAAAAA', properties)
# create the bounding box rectangles
bbox_rects = make_bbox([properties[f'bbox-{i}'] for i in range(4)])

# specify the display parameters for the text
text_parameters = {
    'string': 'label: {label}',
    'size': 12,
    'color': 'green',
    'anchor': 'upper_left',
    'translation': [-3, 0],
}

self.viewer.add_shapes(
    bbox_rects,
    face_color='transparent',
    edge_color='green',
    properties=properties,
    text=text_parameters,
    name='bounding box',
)
'''