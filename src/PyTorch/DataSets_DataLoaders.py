# https://docs.pytorch.org/tutorials/beginner/basics/data_tutorial.html

# Loading a Dataset

import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt

# We load the FashionMNIST Dataset with the following parameters:
#       root -- is the path where the train/test data is stored,
#       train -- specifies training or test dataset,
#       download=True -- downloads the data from the internet if it’s not available at root.
#       transform & target_transform -- specify the feature and label transformations
training_data = datasets.FashionMNIST(
    root="src/PyTorch/data",
    train=True,
    download=True,
    transform=ToTensor()
)

test_data = datasets.FashionMNIST(
    root="src/PyTorch/data",
    train=False,
    download=True,
    transform=ToTensor()
)


# Iterating and Visualizing the Dataset

# We can index Datasets manually like a list: training_data[index].
# We use matplotlib to visualize some samples in our training data.
labels_map = {
    0: "T-Shirt",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle Boot",
}
figure = plt.figure(figsize=(8, 8))
cols, rows = 3, 3
for i in range(1, cols * rows + 1):
    sample_idx = torch.randint(len(training_data), size=(1,)).item()
    img, label = training_data[sample_idx]
    figure.add_subplot(rows, cols, i)
    plt.title(labels_map[label])
    plt.axis("off")
    plt.imshow(img.squeeze(), cmap="gray")
plt.show()


# Preparing your data for training with DataLoaders
#
# The Dataset retrieves our dataset’s features and labels one sample at a time. While training a model,
# we typically want to pass samples in “minibatches”, reshuffle the data at every epoch to reduce model overfitting,
# and use Python’s multiprocessing to speed up data retrieval.
# DataLoader is an iterable that abstracts this complexity for us in an easy API.

from torch.utils.data import DataLoader

train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)

# Iterate through the DataLoader
#
# We have loaded that dataset into the DataLoader and can iterate through the dataset as needed.
# Each iteration below returns a batch of train_features and train_labels (containing batch_size=64 features and labels respectively).
# Because we specified shuffle=True, after we iterate over all batches the data is shuffled (for finer-grained control
# over the data loading order, take a look at Samplers).

# Display image and label.
train_features, train_labels = next(iter(train_dataloader))
print(f"Feature batch shape: {train_features.size()}")
print(f"Labels batch shape: {train_labels.size()}")
img = train_features[0].squeeze()
label = train_labels[0]
plt.imshow(img, cmap="gray")
plt.show()
print(f"Label: {label}")