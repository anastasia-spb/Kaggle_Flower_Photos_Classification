import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
import os


class TestDataset(Dataset):
    def __init__(self, root_dir: str, csv_file: str, transform=None):
        super(Dataset, self).__init__()
        self.transform = transform
        test_images_paths = pd.read_csv(csv_file).Id.values
        self.img_data = self.__read_images(root_dir, test_images_paths)
        self.label = 0

    def __len__(self):
        return len(self.img_data)

    def __read_images(self, root_dir, test_images_paths: np.ndarray):
        img_data = []
        for img_path in test_images_paths:
            full_path = os.path.join(root_dir, img_path)
            with Image.open(full_path).convert('RGB') as sample_image:
                img_data.append((img_path, np.asarray(sample_image)))
        return img_data

    def __getitem__(self, index):
        img_path, image = self.img_data[index]
        if self.transform is not None:
            image = self.transform(image)
        return self.label, img_path, image
