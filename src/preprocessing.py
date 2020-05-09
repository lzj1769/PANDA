import numpy as np
import pandas as pd
import skimage.io
from skimage import morphology
import cv2
import warnings
import os

import configure


def otsu_filter(channel, gaussian_blur=True):
    """Otsu filter."""

    if gaussian_blur:
        channel = cv2.GaussianBlur(channel, (5, 5), 0)
    channel = channel.reshape((channel.shape[0], channel.shape[1]))

    return cv2.threshold(channel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]


def detect_tissue(input_slide, sensitivity=3000):
    """
    Description
    ----------
    Find RoIs containing tissue in WSI.
    Generate mask locating tissue in an WSI. Inspired by method used by
    Wang et al. [1]_.
    .. [1] Dayong Wang, Aditya Khosla, Rishab Gargeya, Humayun Irshad, Andrew
    H. Beck, "Deep Learning for Identifying Metastatic Breast Cancer",
    arXiv:1606.05718
    Credit: Github-wsipre

    Parameters
    ----------
    input_slide: numpy array
        Slide to detect tissue on.
    sensitivity: int
        The desired sensitivty of the model to detect tissue. The baseline is set
        at 3000 and should be adjusted down to capture more potential issue and
        adjusted up to be more agressive with trimming the slide.

    Returns (3)
    -------
    -Tissue binary mask as numpy 2D array,
    -Tiers investigated
    """

    # Convert from RGB to HSV color space
    slide_hsv = cv2.cvtColor(input_slide, cv2.COLOR_BGR2HSV)
    # Compute optimal threshold values in each channel using Otsu algorithm
    _, saturation, _ = np.split(slide_hsv, 3, axis=2)

    mask = otsu_filter(saturation, gaussian_blur=True)
    # Make mask boolean
    mask = mask != 0

    mask = morphology.remove_small_holes(mask, area_threshold=sensitivity)
    mask = morphology.remove_small_objects(mask, min_size=sensitivity)
    mask = mask.astype(np.uint8)
    mask_contours, tier = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    return mask_contours, tier


def draw_tissue_polygons(input_slide, tissue_contours, plot_type, line_thickness=None):
    """
    Description
    ----------
    Plot Tissue Contours as numpy array on.
    Credit: Github-wsipre

    Parameters
    ----------
    input_slide: numpy array
        Slide to draw contours onto
    tissue_contours: numpy array
        These are the identified tissue regions as cv2 contours
    plot_type: str ("line" | "area")
        The desired display type for the tissue regions
    line_thickness: int
        If the polygon_type=="line" then this parameter sets thickness

    Returns (1)
    -------
    - Numpy array of tissue mask plotted
    """

    tissue_color = 1

    for cnt in tissue_contours:
        if plot_type == "line":
            output_slide = cv2.polylines(input_slide, [cnt], True, tissue_color, line_thickness)
        elif plot_type == "area":
            if line_thickness is not None:
                warnings.warn(
                    '"line_thickness" is only used if ' + '"polygon_type" is "line".'
                )

            output_slide = cv2.fillPoly(input_slide, [cnt], tissue_color)
        else:
            raise ValueError('Accepted "polygon_type" values are "line" or "area".')

    return output_slide


def tissue_cutout(input_slide, tissue_contours):
    """
    Description
    ----------
    Set all parts of the in_slide to black except for those
    within the provided tissue contours
    Credit: https://stackoverflow.com/a/28759496

    Parameters
    ----------
    input_slide: numpy array
            Slide to cut non-tissue backgound out
    tissue_contours: numpy array
            These are the identified tissue regions as cv2 contours

    Returns (1)
    -------
    - Numpy array of slide with non-tissue set to black
    """

    # Get intermediate slide
    base_slide_mask = np.zeros(input_slide.shape[:2])

    # Create mask where white is what we want, black otherwise
    crop_mask = np.zeros_like(base_slide_mask)

    # Draw filled contour in mask
    cv2.drawContours(crop_mask, tissue_contours, -1, 255, -1)

    # Extract out the object and place into output image
    tissue_only_slide = np.zeros_like(input_slide)
    tissue_only_slide[crop_mask == 255] = input_slide[crop_mask == 255]

    return tissue_only_slide


def getSubImage(input_slide, rect):
    """
    Description
    ----------
    Take a cv2 rectagle object and remove its contents from
    a source image.
    Credit: https://stackoverflow.com/a/48553593

    Parameters
    ----------
    input_slide: numpy array
            Slide to pull subimage off
    rect: cv2 rect
        cv2 rectagle object with a shape of-
            ((center_x,center_y), (hight,width), angle)

    Returns (1)
    -------
    - Numpy array of rectalge data cut from input slide
    """

    width = int(rect[1][0])
    height = int(rect[1][1])
    box = cv2.boxPoints(rect)

    src_pts = box.astype("float32")
    dst_pts = np.array(
        [[0, height - 1], [0, 0], [width - 1, 0], [width - 1, height - 1]],
        dtype="float32",
    )
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    output_slide = cv2.warpPerspective(input_slide, M, (width, height))
    return output_slide


def color_cut(in_slide, color=[255, 255, 255]):
    """
    Description
    ----------
    Take a input image and remove all rows or columns that
    are only made of the input color [R,G,B]. The default color
    to cut from image is white.

    Parameters
    ----------
    input_slide: numpy array
        Slide to cut white cols/rows
    color: list
        List of [R,G,B] pixels to cut from the input slide

    Returns (1)
    -------
    - Numpy array of input_slide with white removed
    """
    # Remove by row
    row_not_blank = [row.all() for row in ~np.all(in_slide == color, axis=1)]
    output_slide = in_slide[row_not_blank, :]

    # Remove by col
    col_not_blank = [col.all() for col in ~np.all(output_slide == color, axis=0)]
    output_slide = output_slide[:, col_not_blank]
    return output_slide


def fast_detect_and_crop(image_location, sensitivity=3000,
                         input_level=2, output_level=1):
    """
    Description
    ----------
    This method performs the pipeline as described in the notebook:
    https://www.kaggle.com/dannellyz/panda-tissue-detect-scaling-bounding-boxes-fast

    Parameters
    ----------
    image_location:str
        Location of the slide image to process
    sensitivity:int
        The desired sensitivty of the model to detect tissue. The baseline is set
        at 3000 and should be adjusted down to capture more potential issue and
        adjusted up to be more agressive with trimming the slide.
    input_level: int
        The level at which to downsample the slide. This can be referenced in
        reverse order to access the lowest resoltuion items first.
        [-1] = lowest resolution
        [0] = highest resolution
    output_level: int
        The level at which the final slide should sample at. This can be referenced in
        reverse order to access the lowest resoltuion items first.
        [-1] = lowest resolution
        [0] = highest resolution
    Returns (4)
    -------
    - Numpy array of final produciton(prod) slide
    - Percent memory reduciton from original slide
    """
    # Open Small Slide
    wsi_small = skimage.io.MultiImage(image_location)[input_level]

    # Get returns from detect_tissue() ons mall image
    tissue_contours, tier = detect_tissue(wsi_small, sensitivity)

    # Get minimal bounding rectangle for all tissue contours
    if len(tissue_contours) == 0:
        img_id = image_location.split("/")[-1]
        print(f"No Tissue Contours - ID: {img_id}")
        return None

    # Open Big Slide
    wsi_big = skimage.io.MultiImage(image_location)[output_level]

    # Get small boudning rect and scale
    bounding_rect_small = cv2.minAreaRect(np.concatenate(tissue_contours))

    # Scale Rectagle to larger image
    scale = int(wsi_big.shape[0] / wsi_small.shape[0])
    scaled_rect = (
        (bounding_rect_small[0][0] * scale, bounding_rect_small[0][1] * scale),
        (bounding_rect_small[1][0] * scale, bounding_rect_small[1][1] * scale),
        bounding_rect_small[2],
    )
    # Crop bigger image with getSubImage()
    scaled_crop = getSubImage(wsi_big, scaled_rect)

    # Cut out white
    white_cut = color_cut(scaled_crop)

    # Get returns from detect_tissue() on small image
    tissue_contours_big, tier_big = detect_tissue(white_cut, sensitivity)
    prod_slide = tissue_cutout(white_cut, tissue_contours_big)

    return prod_slide


def tile(img, num_tiles=16, tile_size=512):
    """
    Description
    __________
    Tilizer module made by @iafoss that can be found in the notebook:
    https://www.kaggle.com/iafoss/panda-concat-tile-pooling-starter-inference
    Takes a base image and returns the N tiles with the largest differnce
    from a white backgound each with a given square size of input-sz.

    Parameters
    __________
    base_image: numpy array
        Image array to split into tiles and plot
    N: int
        This is the number of tiles to split the image into
    sz: int
        This is the size for each side of the square tiles

    Returns
    __________
    - List of size N with each item being a numpy array tile.
    """

    # Get the shape of the input image
    shape = img.shape

    # Find the padding such that the image divides evenly by the desired size
    pad0, pad1 = (tile_size - shape[0] % tile_size) % tile_size, (tile_size - shape[1] % tile_size) % tile_size

    # Pad the image with blank space to reach the above found targets
    img = np.pad(img,
                 [[pad0 // 2, pad0 - pad0 // 2],
                  [pad1 // 2, pad1 - pad1 // 2], [0, 0]],
                 constant_values=0, mode="constant")

    # Reshape and Transpose to get the images into tiles
    img = img.reshape(img.shape[0] // tile_size, tile_size,
                      img.shape[1] // tile_size, tile_size, 3)
    img = img.transpose(0, 2, 1, 3, 4).reshape(-1, tile_size, tile_size, 3)

    # If there are not enough tiles to meet desired N pad again
    if len(img) < num_tiles:
        img = np.pad(img,
                     [[0, num_tiles - len(img)],
                      [0, 0], [0, 0], [0, 0]],
                     constant_values=0, mode="constant")

    # Sort the images by those with the lowest sum (i.e the least white)
    idxs = np.argsort(img.reshape(img.shape[0], -1).sum(-1))[::-1][:num_tiles]

    # Slect by index those returned from the above funtion
    img = img[idxs]

    return img


if __name__ == "__main__":
    df = pd.read_csv(configure.TRAIN_DF)

    images = dict()
    mean, std = [], []
    for image_id in df['image_id'].values.tolist():
        file_path = f'{configure.TRAIN_IMAGE_PATH}/{image_id}.tiff'
        image = fast_detect_and_crop(image_location=file_path)

        if image is None:
            continue

        image = tile(image)

        # from PIL import Image
        # for i in range(16):
        #     img = Image.fromarray(image[i])
        #     img.save(f'{image_id}_{i}.png')

        images[image_id] = np.empty(shape=(16, 128, 128, 3), dtype=np.uint8)
        for i in range(16):
            images[image_id][i] = cv2.resize(image[i], (128, 128))

        mean.append((images[image_id] / 255.0).reshape(-1, 3).mean(0))
        std.append(((images[image_id] / 255.0) ** 2).reshape(-1, 3).mean(0))

    # image stats
    img_avr = np.array(mean).mean(0)
    img_std = np.sqrt(np.array(std).mean(0) - img_avr ** 2)
    print('mean:', img_avr, ', std:', np.sqrt(img_std))

    images['mean'] = img_avr
    images['std'] = img_std

    np.save(os.path.join(configure.DATA_PATH, "train_images_level_1_128_16"),
            images)
