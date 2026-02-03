"""
Docstring for main

Q: My conv1 has 16 channels and why do 16 feature maps look different after convolution? 
   I know convolution is finding a pattern of image, 
   but why do 16 convolution operations give 16 different feature maps?
A: https://chatgpt.com/s/t_697993080dcc8191a851971266641d83

"""

# import libraries
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
from collections import defaultdict
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import tools

# Define data transform
mnist_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307), (0.3081)) # MNIST
    # transforms.Normalize((0.5), (0.5)) # Fashion MNIST
])

# Load MNIST dataset
train_dataset = datasets.MNIST(root='./data', train=True, transform=mnist_transform, download=True) # 60,000
test_dataset = datasets.MNIST(root='./data', train=False, transform=mnist_transform, download=True) # 10,000
train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)

# Load dataset onto DataLoader
test_loader = DataLoader(dataset=test_dataset, batch_size=1000, shuffle=False)

# Load show_samples() function to plot sample images
tools.show_samples(train_dataset, n=16)

# Define a model as simpleCNN() from tools.py
model = tools.simpleCNN()

# Define hyperparameters
loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
epochs = 5

# # Train the model
# model, train_losses = tools.train(model, train_loader, loss_function, optimizer, epochs)
# tools.plot_training_loss(train_losses)

# # Save the trained model
# save_path = './data/figures/simpleCNN/simple_cnn_mnist.pth'
# torch.save(model.state_dict(), save_path)

# Load the saved model
save_path = './data/figures/simpleCNN/simple_cnn_mnist.pth'
state_dict = torch.load(save_path, map_location=torch.device('cpu'))
model.load_state_dict(state_dict)

# Evaluate the model on test data
all_pred = []
all_labels = []
model.eval()
with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        all_pred.extend(predicted.numpy())
        all_labels.extend(labels.numpy())
        

# # Plot confusion matrix
# cm = confusion_matrix(all_labels, all_pred)
# plt.figure(figsize=(8, 6))
# sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
#             xticklabels=range(10), yticklabels=range(10))
# plt.title("Confusion Matrix")
# plt.xlabel("Predicted Label")
# plt.ylabel("True Label")
# plt.imshow(cm, cmap='viridis')
# plt.show()


# Register hooks to capture pre-activations
activations = {}

def get_activation(name):
    def hook(model, input, output):
        """
        Explanation and background about the forward hooks: https://web.stanford.edu/~nanbhas/blog/forward-hooks-pytorch/
        Save the output activation of a layer during forward pass.
        """
        # hook fires before activation function (on convolution)
        activations[name] = output.detach()
    return hook


# Register forward hooks on the convolutional layers (pre-actviation)
h_conv_1 = model.conv_1.register_forward_hook(get_activation('conv_1'))
h_conv_2 = model.conv_2.register_forward_hook(get_activation('conv_2'))

# Register forward hooks on the ReLU layers after conv layers (post-activation)
h_relu_1 = model.relu_1.register_forward_hook(get_activation('relu_1'))
h_relu_2 = model.relu_2.register_forward_hook(get_activation('relu_2'))

# Register forward hooks on the fully connected layers (pre-activation)
h_fc_1 = model.fc_1.register_forward_hook(get_activation('fc_1'))
h_fc_2 = model.fc_2.register_forward_hook(get_activation('fc_2'))

# Register forward hoooks on the ReLU layers after the first fc layer (post-activations)
h_relu_3 = model.relu_3.register_forward_hook(get_activation('relu_3'))


# Collect images by label
samples_by_label = defaultdict(list)
for image, label in test_dataset:
    samples_by_label[label].append(image)


# Limit number of samples per label for efficiency
max_samples = 500
for label in samples_by_label:
    samples_by_label[label] = samples_by_label[label][:max_samples]


# Run it for digis 0 to 9 in conv_1 and relu_1
for digit in range(10):
    avg_conv_1 = tools.average_activation_for_label(
        model, 'conv_1', digit, samples_by_label, activations
    )
    tools.plot_avg_conv_feature_maps(
        avg_conv_1, 
        title=f'Conv_1 (8, 28, 28): Averaged {max_samples} Feature Maps for {digit}'
    )

    avg_relu_1 = tools.average_activation_for_label(
        model, 'relu_1', digit, samples_by_label, activations
    )
    tools.plot_avg_conv_feature_maps(
        avg_relu_1, title=f'ReLU_1 (8, 28, 28): Averaged {max_samples} Feature Maps for {digit}'
    )


# Run it for digis 0 to 9 in conv_2 and relu_2
for digit in range(10):
    avg_conv_2 = tools.average_activation_for_label(
        model, 'conv_2', digit, samples_by_label, activations
    )
    tools.plot_avg_conv_feature_maps(
        avg_conv_2, title=f'Conv_2 (16, 14, 14): Averaged {max_samples} Feature Maps for {digit}'
    )

    avg_relu_2 = tools.average_activation_for_label(
        model, 'relu_2', digit, samples_by_label, activations
    )
    tools.plot_avg_conv_feature_maps(
        avg_relu_2, title=f'ReLU_2 (16, 14, 14): Averaged {max_samples} Feature Maps for {digit}'
    )


# Run it for digits 0 to 9 in fc_1 and relu_3
for digit in range(10):
    avg_fc_1 = tools.average_activation_for_label(
        model, 'fc_1', digit, samples_by_label, activations
    )
    tools.plot_avg_fc_activations(
        avg_fc_1, title=f'FC_1 (64,): Averaged {max_samples} Activations before ReLU - {digit}'
    )

    avg_relu_3 = tools.average_activation_for_label(
        model, 'relu_3', digit, samples_by_label, activations
    )
    tools.plot_avg_fc_activations(
        avg_relu_3, title=f'FC_1 (64,): Averaged {max_samples} Activations after ReLU - {digit}'
    )


# Run it for digits 0 to 9 in fc_2
for digit in range(10):
    avg_fc_2 = tools.average_activation_for_label(
        model, 'fc_2', digit, samples_by_label, activations
    )
    tools.plot_avg_fc_activations(
        avg_fc_2, title=f'FC_2 (10,): Averaged {max_samples} Feature Maps for {digit}'
    )

# Run softmax function for digits 0 to 9 after fc_2
softmax = nn.Softmax(dim=1)
for digit in range(10):
    avg_fc_2 = tools.average_activation_for_label(
        model, 'fc_2', digit, samples_by_label, activations
    )
    avg_fc_2_softmax = softmax(avg_fc_2.unsqueeze(0)).squeeze(0)
    tools.plot_avg_fc_activations(
        avg_fc_2_softmax, title=f'FC_2 Softmax (10,): Averaged {max_samples} Feature Maps for {digit}'
    )


# Detach the hooks after the forward pass
h_conv_1.remove()
h_relu_1.remove()
h_conv_2.remove()
h_relu_2.remove()
h_fc_1.remove()
h_relu_3.remove()
h_fc_2.remove()