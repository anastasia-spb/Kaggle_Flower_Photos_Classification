import gc
import pathlib

import numpy as np
import pandas as pd
import torch

from dataset_utils import create_dataset, get_categories
from vgg_model_wrapper import VggModelWrapper

# Train model if True. Load model from file otherwise.
TRAIN = True
DATA_ROOT_DIR = '../data/'
IMGS_DIR = 'flower_photos'
SAMPLE_SUBMISSION_FILE = 'sample_submission.csv'

if __name__ == '__main__':
    # Cuda maintenance
    gc.collect()
    torch.cuda.empty_cache()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Torch device: ", device)

    categories_names = get_categories(DATA_ROOT_DIR, IMGS_DIR)
    model_wrapper = VggModelWrapper(len(categories_names), device)
    dataset_for_submission = create_dataset(model_wrapper.get_expected_img_size(), DATA_ROOT_DIR, IMGS_DIR,
                                            SAMPLE_SUBMISSION_FILE, False)

    if TRAIN:
        train_dataset, train_validation_dataset = create_dataset(model_wrapper.get_expected_img_size(), DATA_ROOT_DIR,
                                                                 IMGS_DIR, SAMPLE_SUBMISSION_FILE)
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
    sample_submission['Id'] = [pathlib.Path[p].parts[2:] for p in np.concatenate(img_names_column)]

    sample_submission.to_csv('flowers_submission.csv', index=False)
