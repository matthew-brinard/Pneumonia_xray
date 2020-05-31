import time
import copy
import numpy as np
import matplotlib.pyplot as plt
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train_model(model, dataloader_dict, criterion, optimizer, num_epochs=20):
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
            epoch_loss = running_loss / len(dataloader_dict[phase].dataset)
            epoch_acc = running_corrects / len(dataloader_dict[phase].dataset)
            print('{} Accuracy: {:,.2f}% Loss: {:,.2f}'.format(phase, (epoch_acc * 100), (epoch_loss * 100)))
            if phase == 'train':
                train_acc_history.append(epoch_acc)
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_acc_history.append(epoch_acc)
    total_time = time.time() - since
    print('Training finished in {:.0f}m {:.0f}s'.format(total_time // 60, total_time % 60))
    print('Best validation accuracy: {:,.2f}%'.format(best_acc * 100))
    return model, train_acc_history, val_acc_history


def test_model(model, test_loader):
    """
    Tests the trained neural network model against the test dataset.
    :param model: Trained neural network model that will be tested against the test dataset.
    :type model: Net
    :param test_loader: Dataloader object containing test dataset.
    :type test_loader: torch.utils.data.dataloader.Dataloader.
    :return: test_accuracy
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
    test_accuracy = running_corrects / len(test_loader.dataset)
    return test_accuracy


def plot_history(t_hist, v_hist):
    """
    Plots a graph of the training and validation accuracy vs. epoch.
    :param t_hist: Training accuracy history.
    :type t_hist: list
    :param v_hist: Validation accuracy history
    :type v_hist: list
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
    plt.xticks((np.arange(1, len(train_hist) + 1, 1)))
    plt.show()
    return None
