import cv2
import os
import numpy as np
import pandas as pd
from skimage import morphology
import openslide
import warnings
from PIL import Image

import configure


def otsu_filter(channel, gaussian_blur=True):
    """Otsu filter."""
    if gaussian_blur:
        channel = cv2.GaussianBlur(channel, (5, 5), 0)
    channel = channel.reshape((channel.shape[0], channel.shape[1]))

    return cv2.threshold(
        channel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]


def tissue_cutout(tissue_slide, tissue_contours, slide):
    # https://stackoverflow.com/a/28759496
    crop_mask = np.zeros_like(tissue_slide)  # Create mask where white is what we want, black otherwise
    cv2.drawContours(crop_mask, tissue_contours, -1, 255, -1)  # Draw filled contour in mask
    tissue_only = np.zeros_like(slide)  # Extract out the object and place into output image
    tissue_only[crop_mask == 255] = slide[crop_mask == 255]
    return tissue_only


def draw_tissue_polygons(mask, polygons, polygon_type,
                         line_thickness=None):
    """
    Plot as numpy array detected tissue.
    Modeled WSIPRE github package

    Parameters
    ----------
    mask: numpy array
        This is the original image represented as 0's for a starting canvas
    polygons: numpy array
        These are the identified tissue regions
    polygon_type: str ("line" | "area")
        The desired display type for the tissue regions
    polygon_type: int
        If the polygon_type=="line" then this parameter sets thickness

    Returns
    -------
    Nunmpy array of tissue mask plotted
    """

    tissue_color = 1

    for poly in polygons:
        if polygon_type == 'line':
            mask = cv2.polylines(
                mask, [poly], True, tissue_color, line_thickness)
        elif polygon_type == 'area':
            if line_thickness is not None:
                warnings.warn('"line_thickness" is only used if ' +
                              '"polygon_type" is "line".')

            mask = cv2.fillPoly(mask, [poly], tissue_color)
        else:
            raise ValueError(
                'Accepted "polygon_type" values are "line" or "area".')

    return mask


def getSubImage(rect, src_img):
    width = int(rect[1][0])
    height = int(rect[1][1])
    box = cv2.boxPoints(rect)

    src_pts = box.astype("float32")
    dst_pts = np.array([[0, height - 1],
                        [0, 0],
                        [width - 1, 0],
                        [width - 1, height - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped = cv2.warpPerspective(src_img, M, (width, height))
    return warped


def detect_tissue(wsi, sensitivity=3000, downsampling_factor=4):
    """
    Find RoIs containing tissue in WSI.
    Generate mask locating tissue in an WSI. Inspired by method used by
    Wang et al. [1]_.
    .. [1] Dayong Wang, Aditya Khosla, Rishab Gargeya, Humayun Irshad, Andrew
    H. Beck, "Deep Learning for Identifying Metastatic Breast Cancer",
    arXiv:1606.05718

    Parameters
    ----------
    wsi: OpenSlide/AnnotatedOpenSlide class instance
        The whole-slide image (WSI) to detect tissue in.
    downsampling_factor: int
        The desired factor to downsample the image by, since full WSIs will
        not fit in memory. The image's closest level downsample is found
        and used.
    sensitivity: int
        The desired sensitivty of the model to detect tissue. The baseline is set
        at 5000 and should be adjusted down to capture more potential issue and
        adjusted up to be more agressive with trimming the slide.

    Returns
    -------
    -Binary mask as numpy 2D array,
    -RGB slide image (in the used downsampling level, in case the user is visualizing output examples),
    -Downsampling factor.
    """

    # Get a downsample of the whole slide image (to fit in memory)
    downsampling_factor = min(
        wsi.level_downsamples, key=lambda x: abs(x - downsampling_factor))
    level = wsi.level_downsamples.index(downsampling_factor)

    slide = wsi.read_region((0, 0), level, wsi.level_dimensions[level])
    slide = np.array(slide)[:, :, :3]

    # Convert from RGB to HSV color space
    slide_hsv = cv2.cvtColor(slide, cv2.COLOR_BGR2HSV)

    # Compute optimal threshold values in each channel using Otsu algorithm
    _, saturation, _ = np.split(slide_hsv, 3, axis=2)

    mask = otsu_filter(saturation, gaussian_blur=True)

    # Make mask boolean
    mask = mask != 0

    mask = morphology.remove_small_holes(mask, area_threshold=sensitivity)
    mask = morphology.remove_small_objects(mask, min_size=sensitivity)

    mask = mask.astype(np.uint8)
    mask_contours, tier = cv2.findContours(
        mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    return mask_contours, tier, slide, downsampling_factor


#def detect_and_crop(image, sensitivity=3000)

if __name__ == "__main__":
    df_train = pd.read_csv(configure.TRAIN_DF)

    # for image_id in df_train['image_id'].values.tolist():
    #     input_image = os.path.join(configure.TRAIN_IMAGE_PATH,
    #                                f"{image_id}.tiff")
    #     output_image = os.path.join(configure.TISSUE_DETECTION_TRAIN_IMAGE_PATH,
    #                                 f"{image_id}.png")
    #     detect_and_crop(input_image, output_image)

    input_image = os.path.join(configure.TRAIN_IMAGE_PATH,
                               "0032bfa835ce0f43a92ae0bbab6871cb.tiff")
    output_image = os.path.join("test.png")
    detect_and_crop(input_image, output_image, sensitivity=3000,
                    downsample_rate=16)
