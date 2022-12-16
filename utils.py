import numpy as np

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