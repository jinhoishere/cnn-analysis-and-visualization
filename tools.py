from collections import defaultdict
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np


class simpleCNN(nn.Module):
    def __init__(self):
        super(simpleCNN, self).__init__()

        self.conv_1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, stride=1, padding=1) 
        self.relu_1 = nn.ReLU()
        self.max_pool_1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0) 

        self.conv_2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1) 
        self.relu_2 = nn.ReLU()
        self.max_pool_2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0) 

        self.fc_1 = nn.Linear(16 * 7 * 7, 64)
        self.relu_3 = nn.ReLU()

        self.fc_2 = nn.Linear(64, 10)


    def forward(self, x):
        # (B, C, H, W) in conv layers
        x = self.conv_1(x) # (8, 28, 28)
        x = self.relu_1(x)
        x = self.max_pool_1(x) # (8, 14, 14)

        x = self.conv_2(x) # (16, 14, 14)
        x = self.relu_2(x)
        x = self.max_pool_2(x) # (16, 7, 7)

        # flatten before fc layers
        x = x.view(x.size(0), 16 * 7 * 7)

        # (B, N) in fc layers
        x = self.fc_1(x) # (64,)
        x = self.relu_3(x) 
        x = self.fc_2(x) # (10,)
        
        return x
    

def show_samples(dataset, n):
    """
    Display random sample images from the dataset. Only square numbers are supported.

    :param dataset: NOT dataloader, but dataset object
    :param n: the number of total images to show
    """

    row = (int) (n ** 0.5) # edit here to modify number of rows
    column = (int) (n ** 0.5) # edit here to modify number of columns

    plt.figure(figsize=(column, row)) # column, row

    for i in range(n):
        random_index = random.randint(0, len(dataset) - 1) # 0 - 59999
        current_image, current_label = dataset[random_index]
        plt.subplot(row, column, i+1) # row, column, index
        plt.imshow(current_image.squeeze(0), cmap='gray') # (C, H, W) -> (H, W)
        plt.title(f'Label: {current_label}')
        plt.axis('off')

    plt.suptitle(f"Sample MNIST Images ({n})", fontsize=16)
    plt.tight_layout()
    plt.show()


def train(model, train_loader, loss_function, optimizer, epochs):
    """
    Train the model and return the model and training losses over epochs.

    :param model: (nn.Module), the CNN model to train
    :param train_loader: (DataLoader), training data loader
    :param loss_function: loss function to optimize
    :param optimizer: optimizer for training
    :param epochs: (int), number of epochs to train

    :return: model (nn.Module) and list of training losses over epochs (list)
    """ 

    train_losses = []

    for epoch in range(epochs):
        model.train()
        batch_loss = 0.0
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            batch_loss += loss.item()

        avg_loss = batch_loss / len(train_loader)
        train_losses.append(avg_loss)
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}')

    return model, train_losses


def plot_training_loss(train_losses):
    """ 
    Plot training loss over epochs

    :param train_losses: (list), list of training losses over epochs
    """

    epochs = np.arange(1, len(train_losses) + 1)

    _, ax = plt.subplots()
    ax.plot(epochs, train_losses)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Training Loss")
    ax.set_xlim(1, epochs[-1])
    ax.xaxis.set_major_locator(
        plt.MaxNLocator(integer=True)
    )
    plt.title('Training Loss over Epoch')
    plt.grid(True)
    plt.show()



def average_activation_for_label(model: nn.Module, layer_name: str, label: int, samples_by_label: defaultdict, activations: dict) -> torch.Tensor:
    """
    Get average activation for one label

    :param model: (nn.Module), the trained CNN model
    :param layer_name: (str), 'conv_1' or 'conv_2'
    :param label: (int), digit label 0-9
    :param samples_by_label: (class 'collections.defaultdict'), key: label, value: list of images
    :param activations: (dict), dictionary to store activations captured by hooks

    * Return average activations for a given label and layer
        - Return data type: torch.Tensor (C, H, W)
        - Return data shape: conv_1: (16, 28, 28), conv_2: (32, 14, 14)
    """
    model.eval()
    collected = []

    for image in samples_by_label[label]:
        _ = model(image.unsqueeze(0)) # remove batch dimension
        collected.append(activations[layer_name])

    return torch.mean(torch.cat(collected, dim=0), dim=0)


def plot_avg_conv_feature_maps(feature_maps, title, max_maps=32):
    """
    Visualize averaged feature maps in convolutional layers.

    :param feature_maps: (torch.Tensor), shape: (num_feature_maps, H, W)
    :param title: (str), title of the plot
    :param max_maps: (int), maximum number of feature maps to display, default is 32
    """
    
    num_maps = min(feature_maps.shape[0], max_maps)
    # print(f"Plotting {num_maps} feature maps. feature_maps.shape: {feature_maps.shape}, max_maps: {max_maps}")
    cols = 4
    rows = (num_maps + cols - 1) // cols

    plt.figure(figsize=(cols * 2, rows * 2))
    for i in range(num_maps):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(feature_maps[i].cpu(), cmap='gray')
        plt.title(f'Channel {i+1}')
        plt.axis('off')

    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.show()


def plot_avg_fc_activations(avg_activations, title):
    """
    Visualize averaged activations in fully connected layers with a bar plot.

    :param avg_activations: (torch.Tensor), shape: (num_neurons,)
    :param title: (str), title of the plot
    """
    
    x = np.arange(len(avg_activations))
    y = avg_activations.cpu().numpy()

    plt.figure(figsize=(10, 3))
    plt.bar(x, y)
    plt.xticks(x)

    # plt.figure(figsize=(8, 4))
    # plt.plot(activations)
    # plt.xticks(np.arange(len(avg_activations)))
    
    plt.xlabel("Neuron Index")
    plt.title(title)
    plt.ylabel("Average Activation")
    plt.grid(True)
    plt.show()