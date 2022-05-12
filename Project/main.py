import random

import torch
import gc
import os
from torchvision import transforms
from jpeg_dataset import ImagesDataset, ImageInfo
from test_dataset import TestDataset
from custom_transformations import PerImageNormalization
import numpy as np
import pandas as pd
from model_launcher import ModelWrapper

# Dataset parameters
CLASS_NUM = 10
IMG_SIZE = 224
IMG_CHANNELS = 3
# Train model if True. Load model from file otherwise.
TRAIN = False


# It might be necessary to increase swap
# before loading whole training dataset: https://askubuntu.com/questions/178712/how-to-increase-swap-space
# with parameters bs=1024 count=136314880.
# It took around 100Gb of Swp
def create_dataset(train=True):
    train_image_transformation = transforms.Compose([
        transforms.ToPILImage(),  # Convert a tensor or a ndarray to PIL Image
        transforms.RandomResizedCrop((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        PerImageNormalization()])

    validation_image_transformation = transforms.Compose([
        transforms.ToPILImage(),  # Convert a tensor or a ndarray to PIL Image
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.CenterCrop((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        PerImageNormalization()])

    data_dir = '../data/flower_photos'
    labels_dirs_names = [name for name in os.listdir(data_dir) if
                         os.path.isdir(os.path.join(data_dir, name))]

    if train:
        # Store all images paths and corresponding labels
        images_info = []
        for index in range(len(labels_dirs_names)):
            current_folder = labels_dirs_names[index]
            print(current_folder)
            current_path = os.path.join(data_dir, current_folder)
            for img_name in os.listdir(current_path):
                images_info.append(ImageInfo(os.path.join(current_path, img_name), index))

        # Shuffle images info
        random.shuffle(images_info)

        # Split to train and train-validation datasets
        train_set_size = int(0.8 * len(images_info))

        return labels_dirs_names, \
               ImagesDataset(images_info[:train_set_size], train_image_transformation), \
               ImagesDataset(images_info[train_set_size:], validation_image_transformation)
    else:
        # @todo: Use ImagesDataset
        return labels_dirs_names, \
               TestDataset('../data/', '../data/sample_submission.csv', validation_image_transformation)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # Cuda maintenance
    gc.collect()
    torch.cuda.empty_cache()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Torch device: ", device)

    categories_names, dataset_for_submission = create_dataset(False)
    model_wrapper = ModelWrapper(len(categories_names), device)

    if TRAIN:
        categories_names, train_dataset, train_validation_dataset = create_dataset()
        model_wrapper.train(train_dataset, train_validation_dataset, True)
        submission_results, img_names_column = model_wrapper.predict(dataset_for_submission)
    else:
        submission_results, img_names_column = model_wrapper.predict(dataset_for_submission, './model_from_last_train.pth')

    # Write results into submission file
    sample_submission = pd.read_csv('../data/sample_submission.csv')
    labels_mapping = {'tulips': 'TULIP',
                      'daisy': 'DAISY',
                      'sunflowers': 'SUNFLOWER',
                      'dandelion': 'DANDELION',
                      'roses': 'ROSE'}

    sample_submission['Category'] = [labels_mapping[categories_names[index]] for index in np.concatenate(submission_results)]
    sample_submission['Id'] = np.concatenate(img_names_column)

    sample_submission.to_csv('flowers_submission.csv', index=False)
