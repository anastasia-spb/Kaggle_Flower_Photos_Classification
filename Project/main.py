import gc
import os
import random

import numpy as np
import pandas as pd
import torch
from torchvision import transforms

from custom_transformations import PerImageNormalization
from jpeg_dataset import JpegDataset, ImageInfo
from vgg_model_wrapper import VggModelWrapper

# Train model if True. Load model from file otherwise.
TRAIN = False
DATA_ROOT_DIR = '../data/'
IMGS_DIR = 'flower_photos'
SAMPLE_SUBMISSION_FILE = 'sample_submission.csv'


def get_categories(root_dir=DATA_ROOT_DIR, imgs_dir=IMGS_DIR):
    data_dir = os.path.join(root_dir, imgs_dir)
    categories_dirs_names = [name for name in os.listdir(data_dir) if
                             os.path.isdir(os.path.join(data_dir, name))]
    return categories_dirs_names


# It might be necessary to increase swap
# before loading whole training dataset: https://askubuntu.com/questions/178712/how-to-increase-swap-space
# with parameters bs=1024 count=136314880.
def create_dataset(expected_img_size: int, train=True, root_dir=DATA_ROOT_DIR, imgs_dir=IMGS_DIR,
                   sample_submission_file=SAMPLE_SUBMISSION_FILE):
    """
    create_dataset Create train, validation and test datasets

    :param expected_img_size: Size of image expected by model. Input images will be reshaped to a square size.
    :param train: If True, then train and validation datasets will be formed
    :param root_dir: Path to root folder of dataset
    :param imgs_dir: Name of directory, which contains folders with images of every category
    :param sample_submission_file: Name of csv file, which contains paths relative @ref root_dir of images
                                   for model testing
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


if __name__ == '__main__':
    # Cuda maintenance
    gc.collect()
    torch.cuda.empty_cache()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Torch device: ", device)

    categories_names = get_categories(DATA_ROOT_DIR, IMGS_DIR)
    model_wrapper = VggModelWrapper(len(categories_names), device)
    dataset_for_submission = create_dataset(model_wrapper.get_expected_img_size(), False)

    if TRAIN:
        train_dataset, train_validation_dataset = create_dataset(model_wrapper.get_expected_img_size())
        val_acc_history = model_wrapper.train(train_dataset, train_validation_dataset, True)
        submission_results, img_names_column = model_wrapper.predict(dataset_for_submission)
    else:
        submission_results, img_names_column = model_wrapper.predict(dataset_for_submission,
                                                                     './model_from_last_train.pth')

    # Write results into submission file
    sample_submission = pd.read_csv('../data/sample_submission.csv')
    labels_mapping = {'tulips': 'TULIP',
                      'daisy': 'DAISY',
                      'sunflowers': 'SUNFLOWER',
                      'dandelion': 'DANDELION',
                      'roses': 'ROSE'}

    sample_submission['Category'] = [labels_mapping[categories_names[index]] for index in
                                     np.concatenate(submission_results)]
    sample_submission['Id'] = np.concatenate(img_names_column)

    sample_submission.to_csv('flowers_submission.csv', index=False)
