import os
import random

import pandas as pd
from torchvision import transforms

from custom_transformations import PerImageNormalization
from jpeg_dataset import JpegDataset, ImageInfo


def get_categories(root_dir: str, imgs_dir: str):
    data_dir = os.path.join(root_dir, imgs_dir)
    categories_dirs_names = [name for name in os.listdir(data_dir) if
                             os.path.isdir(os.path.join(data_dir, name))]
    return categories_dirs_names


# It might be necessary to increase swap
# before loading whole training dataset: https://askubuntu.com/questions/178712/how-to-increase-swap-space
# with parameters bs=1024 count=136314880.
def create_dataset(expected_img_size: int, root_dir: str, imgs_dir: str, sample_submission_file: str, train=True):
    """
    create_dataset Create train, validation and test datasets

    :param expected_img_size: Size of image expected by model. Input images will be reshaped to a square size.
    :param root_dir: Path to root folder of dataset
    :param imgs_dir: Name of directory, which contains folders with images of every category
    :param sample_submission_file: Name of csv file, which contains paths relative @ref root_dir of images
                                   for model testing
    :param train: If True, then train and validation datasets will be formed
    :return: If @ref train is True, then returns train and validation datasets.
             Otherwise, returns test dataset
    """
    train_image_transformation = transforms.Compose([
        transforms.ToPILImage(),  # Convert a tensor or a ndarray to PIL Image
        transforms.RandomResizedCrop(expected_img_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        PerImageNormalization()])

    validation_image_transformation = transforms.Compose([
        transforms.ToPILImage(),  # Convert a tensor or a ndarray to PIL Image
        transforms.Resize(expected_img_size),
        transforms.CenterCrop(expected_img_size),
        transforms.ToTensor(),
        PerImageNormalization()])

    data_dir = os.path.join(root_dir, imgs_dir)
    categories_dirs_names = get_categories(root_dir, imgs_dir)

    if train:
        # Store all images paths and corresponding labels
        images_info = []
        for index in range(len(categories_dirs_names)):
            current_folder = categories_dirs_names[index]
            print(current_folder)
            current_path = os.path.join(data_dir, current_folder)
            for img_name in os.listdir(current_path):
                images_info.append(ImageInfo(os.path.join(current_path, img_name), index))

        # Shuffle images info
        random.shuffle(images_info)

        # Split to train and train-validation datasets
        train_set_size = int(0.8 * len(images_info))

        return JpegDataset(images_info[:train_set_size], train_image_transformation), \
               JpegDataset(images_info[train_set_size:], validation_image_transformation)
    else:
        path_to_submission_template = os.path.join(root_dir, sample_submission_file)
        test_images_paths = pd.read_csv(path_to_submission_template).Id.values
        test_images_info = [ImageInfo(os.path.join(root_dir, img_path), -1) for img_path in test_images_paths]

        return JpegDataset(test_images_info, validation_image_transformation)
