from __future__ import division
from __future__ import print_function

import copy

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models


class VggModelWrapper:
    """
    This class loads pretrained VGG11_bn model and
    only update the final layer weights
    """

    def __init__(self, num_classes, device):
        self.device = device
        self.num_classes = num_classes
        # Initialize learning parameters
        self.criterion = nn.CrossEntropyLoss()
        self.num_epochs = 50
        self.batch_size = 64
        self.__initialize_vgg_model()
        self.__create_optimizer()

    def __set_parameter_requires_grad(self):
        """
        Since this class only performs feature extraction,
        we shall set .requires_grad attribute of the parameters
        in the model to False.
        Gradients shall be computed only for the newly initialized layer.
        """
        for param in self.model.parameters():
            param.requires_grad = False

    def __initialize_vgg_model(self):
        # We only update the reshaped layer params
        self.model = models.vgg11_bn(pretrained=True)
        self.__set_parameter_requires_grad()
        num_ftrs = self.model.classifier[6].in_features
        self.model.classifier[6] = nn.Linear(num_ftrs, self.num_classes)
        self.input_size = 224  # VGG11 expects image of size (224, 224)
        self.model = self.model.to(self.device)

    def get_expected_img_size(self):
        return self.input_size

    def __create_optimizer(self):
        """
        Gather the parameters to be optimized/updated.
        In feature extract method, we will only update the parameters
        that we have just initialized.
        """
        params_to_update = []
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                params_to_update.append(param)
        self.optimizer_ft = optim.SGD(params_to_update, lr=0.001, momentum=0.9)

    def train(self, train_dataset, validation_dataset, save_model=False):
        dataloaders = {"train": DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True),
                       "val": DataLoader(validation_dataset, batch_size=self.batch_size, shuffle=True)}

        val_acc_history = []

        best_model_wts = copy.deepcopy(self.model.state_dict())
        best_acc = 0.0

        for epoch in range(self.num_epochs):
            print('Epoch {}/{}'.format(epoch, self.num_epochs - 1))
            print('-' * 10)
            for phase in ['train', 'val']:
                if phase == 'train':
                    self.model.train()  # Set model to training mode
                else:
                    self.model.eval()  # Set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0

                for labels, _, inputs in dataloaders[phase]:
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)

                    # zero the parameter gradients
                    self.optimizer_ft.zero_grad()

                    # forward
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = self.model(inputs)
                        loss = self.criterion(outputs, labels)
                        _, preds = torch.max(outputs, 1)

                        if phase == 'train':
                            loss.backward()
                            self.optimizer_ft.step()

                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                epoch_loss = running_loss / len(dataloaders[phase].dataset)
                epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

                print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

                # deep copy the model
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(self.model.state_dict())
                if phase == 'val':
                    val_acc_history.append(epoch_acc)

            print()

        print('Best val Acc: {:4f}'.format(best_acc))

        # load best model weights
        self.model.load_state_dict(best_model_wts)
        if save_model:
            torch.save(self.model.state_dict(), './model_from_last_train.pth')

        return val_acc_history

    def predict(self, torch_dataset, model_from_file=''):
        if model_from_file:
            self.model.load_state_dict(torch.load(model_from_file))

        submission_dl = DataLoader(torch_dataset, batch_size=self.batch_size, shuffle=False)

        submission_results = []
        img_names_column = []
        for batch_idx, (_, img_names, data) in enumerate(submission_dl):
            ## Forward Pass
            scores = self.model(data.cuda())
            softmax = torch.exp(scores).cpu()
            prob = list(softmax.detach().numpy())
            predictions = np.argmax(prob, axis=1)
            # Store results
            submission_results.append(predictions)
            img_names_column.append(img_names)

        return submission_results, img_names_column
