import numpy as np
import pandas as pd
import configure
import skimage.io
import cv2


def crop_white(image: np.ndarray) -> np.ndarray:
    assert image.shape[2] == 3
    assert image.dtype == np.uint8
    ys, = (image.min((1, 2)) != 255).nonzero()
    xs, = (image.min(0).min(1) != 255).nonzero()
    if len(xs) == 0 or len(ys) == 0:
        return image

    return image[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


def tile(img, tile_size=128, num_tiles=12):
    shape = img.shape
    pad0, pad1 = (tile_size - shape[0] % tile_size) % tile_size, \
                 (tile_size - shape[1] % tile_size) % tile_size

    img = np.pad(img, [[pad0 // 2, pad0 - pad0 // 2],
                       [pad1 // 2, pad1 - pad1 // 2],
                       [0, 0]], constant_values=255, mode='constant')

    img = img.reshape(img.shape[0] // tile_size, tile_size,
                      img.shape[1] // tile_size, tile_size, 3)
    img = img.transpose(0, 2, 1, 3, 4).reshape(-1, tile_size, tile_size, 3)
    if len(img) < num_tiles:
        img = np.pad(img, [[0, num_tiles - len(img)],
                           [0, 0], [0, 0], [0, 0]],
                     constant_values=255, mode='constant')

    idxs = np.argsort(img.reshape(img.shape[0], -1).sum(-1))[:num_tiles]
    img = img[idxs]

    return img


if __name__ == "__main__":
    df = pd.read_csv(configure.TRAIN_DF)

    images = dict()
    mean, std = [], []
    for i, image_id in enumerate(df['image_id'].values.tolist()):
        if i % 500 == 0:
            print(f"processed {i} images...")

        file_path = f'{configure.TRAIN_IMAGE_PATH}/{image_id}.tiff'
        image = skimage.io.MultiImage(file_path)[-2]
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = crop_white(image)
        images[image_id] = tile(image)

        mean.append((images[image_id] / 255.0).reshape(-1, 3).mean(0))
        std.append(((images[image_id] / 255.0) ** 2).reshape(-1, 3).mean(0))

    np.save(configure.TRAINING_IMAGE_NPY, images)

    # image stats
    img_avr = np.array(mean).mean(0)
    img_std = np.sqrt(np.array(std).mean(0) - img_avr ** 2)
    print('mean:', img_avr, ', std:', np.sqrt(img_std))
