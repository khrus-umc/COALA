import nibabel as nib
import numpy as np


def get_hu_lesions(segmentation, hu_ct):
    lesions = np.unique(segmentation)
    hu_lesions = []
    for lesion in lesions:
        hu_lesion = hu_ct[segmentation == lesion]
        hu_lesions.append(np.mean(hu_lesion))

    return hu_lesions


def remove_lesions(seg_path, scan_path, hu):
    segmentation = nib.load(seg_path).get_fdata()

    ct = nib.load(scan_path).get_fdata()
    rescale_slope = ct.header.get_zooms()[0]
    rescale_intercept = ct.header.get_zooms()[1]
    hu_ct = (ct * rescale_slope) + rescale_intercept

    if hu_ct.shape != segmentation.shape:
        raise ValueError("CT scan and lesion mask must have the same dimensions")

    hu_lesions = get_hu_lesions(segmentation, hu_ct)
    lesions_to_remove = [lesion for lesion in hu_lesions if lesion == 0]