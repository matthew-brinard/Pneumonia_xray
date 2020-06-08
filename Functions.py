import time
import copy
from typing import Tuple, List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch.utils.data import DataLoader, Dataset
from NeuralNet import Net

sns.set()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train_model(model: Net, dataloader_dict: dict, criterion: torch.nn.modules.loss, optimizer: torch.optim,
                num_epochs: int = 20) -> Tuple[Net, List[torch.Tensor], List[torch.Tensor]]:
    """
    This function trains the neural network model.
    :param model: Neural Network model that will be trained on the datasets.
    :type model: Net
    :param dataloader_dict: Dictionary object that contains a training dataset and a validation dataset.
    :type dataloader_dict: dictionary
    :param criterion: Loss function.
    :type criterion: torch.nn.modules.loss.
    :param optimizer: Optimization function.
    :type optimizer: torch.optim.
    :param num_epochs: Number of epochs to train for. Default is 20.
    :type num_epochs: int
    :return: model, train_acc_history, val_acc_history
    """
    since = time.time()
    train_acc_history = []
    val_acc_history = []
    best_model = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_epoch = 0
    for epoch in range(num_epochs):
        print('\nEpoch {} of {}'.format(epoch + 1, num_epochs))
        print('-' * 40)
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
            running_loss = 0.0
            running_corrects = 0.0
            for inputs, labels in dataloader_dict[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    _, predicted = torch.max(outputs, 1)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(predicted == labels.data)
            epoch_loss = running_loss / (len(dataloader_dict[phase]) * dataloader_dict[phase].batch_size)
            epoch_acc = running_corrects / (len(dataloader_dict[phase]) * dataloader_dict[phase].batch_size)
            print('{} Accuracy: {:,.2f}% Loss: {:,.2f}'.format(phase, (epoch_acc * 100), (epoch_loss * 100)))
            if phase == 'train':
                train_acc_history.append(epoch_acc)
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model = copy.deepcopy(model.state_dict())
                best_epoch = epoch + 1
            if phase == 'val':
                val_acc_history.append(epoch_acc)
    total_time = time.time() - since
    print('Training finished in {:.0f}m {:.0f}s'.format(total_time // 60, total_time % 60))
    print('Best validation accuracy: {:,.2f}% at epoch {:.0f}'.format(best_acc * 100, best_epoch))
    return model, train_acc_history, val_acc_history


def test_model(model: Net, test_loader: DataLoader) -> float:
    """
    Tests the trained neural network model against the test dataset.
    :param model: Trained neural network model that will be tested against the test dataset.
    :type model: Net
    :param test_loader: Dataloader object containing test dataset.
    :type test_loader: torch.utils.data.dataloader.Dataloader.
    :return test_accuracy: float signifying overall accuracy of the model on the test dataset.
    """
    with torch.no_grad():
        model.eval()
        running_corrects = 0.0
        for data, labels in test_loader:
            data = data.to(device)
            labels = labels.to(device)
            output = model(data)
            _, predicted = torch.max(output, 1)
            running_corrects += torch.sum(predicted == labels.data)
    test_accuracy = running_corrects / (len(test_loader) * test_loader.batch_size)
    return test_accuracy


def class_accuracy(model: Net, test_loader: DataLoader, dataset: Dataset) -> pd.DataFrame:
    """
    Tests the trained neural network model against the test dataset. Returns per class accuracy metrics.
    :param model: Trained neural network model that will be tested against the test dataset.
    :type model: Net
    :param test_loader: Dataloader object containing test dataset.
    :type test_loader: torch.utils.data.dataloader.Dataloader.
    :param dataset: Dataset object containing test images.
    :type dataset: torch.utils.data.dataset
    :return class_data: pandas DataFrame object containing per class accuracy.
    """
    classes = dataset.classes
    class_correct = list(0. for i in range(len(classes)))
    class_total = list(0. for i in range(len(classes)))
    with torch.no_grad():
        model.eval()
        for data, labels in test_loader:
            data = data.to(device)
            labels = labels.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(c.size(0)):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1
    data_tuples = list(zip(classes, class_correct, class_total))
    class_data = pd.DataFrame(data_tuples, columns=['Class', 'Class Correct', 'Class Total'])
    class_data['Class Accuracy'] = class_data['Class Correct'] / class_data['Class Total']
    return class_data


def plot_history(t_hist: List[torch.Tensor], v_hist: List[torch.Tensor]) -> None:
    """
    Plots a graph of the training and validation accuracy vs. epoch.
    :param t_hist: Training accuracy history.
    :type t_hist: list of torch.Tensor objects.
    :param v_hist: Validation accuracy history
    :type v_hist: list of torch.Tensor objects
    :return: None
    """
    train_hist = [h.cpu().numpy() for h in t_hist]
    val_hist = [h.cpu().numpy() for h in v_hist]
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.plot(range(1, (len(train_hist) + 1)), train_hist, color='blue', label='Training')
    plt.plot(range(1, (len(val_hist) + 1)), val_hist, color='orange', label='Validation')
    plt.legend()
    plt.ylim((0, 1.0))
    plt.xticks((np.arange(1, len(train_hist) + 1, 4)))
    plt.show()
    return None


def plot_class_accuracy(class_data: pd.DataFrame) -> None:
    top_plot = sns.barplot(x=class_data['Class'], y=class_data['Class Total'], color='#9b59b6')
    bot_plot = sns.barplot(x=class_data['Class'], y=class_data['Class Correct'], color='#3498db')
    top_bar = plt.Rectangle((0, 0), 1, 1, fc='#9b59b6', edgecolor='none')
    bot_bar = plt.Rectangle((0, 0), 1, 1, fc='#3498db', edgecolor='none')
    legend = plt.legend([bot_bar, top_bar], ['Correct', 'Total'], loc=4, ncol=4, prop={'size': 12}) #fix legend
    legend.draw_frame(False)
    sns.despine(left=True)
    bot_plot.set_ylabel("# of Images")
    bot_plot.set_xlabel("Class")
    bot_plot.set_title("Model Per Class Accuracy")
    plt.show()
    return None
