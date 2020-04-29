import cv2
import skimage.io
from torch.utils.data import Dataset
import utils


class PandaDataset(Dataset):
    def __init__(self, df, data_dir, transform=None,
                 image_width=256, image_height=256):
        self.df = df
        self.data_dir = data_dir
        self.transform = transform
        self.image_width = image_width
        self.image_height = image_height

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        file_name = self.df['image_id'].values[idx]
        file_path = f'{self.data_dir}/{file_name}.tiff'
        image = skimage.io.MultiImage(file_path)

        # remove white background
        image = utils.crop_white(image[-1])
        image = cv2.resize(image, (self.image_width, self.image_height))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']

        label = self.df['isup_grade'].values[idx]

        return image, label

