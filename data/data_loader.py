import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

"MNIST", "CIFAR10", ""


def load_MNIST(
    dataset="MNIST",
    train_size=None,
    test_size=None,
    val_size=None,
    batch_size=32,
    shuffle=False,
    download=True,
    root="./data",
):
    """
    Load the MNIST dataset and create train, validation, and test DataLoaders.

    Parameters:
    - train_size (int, optional): Number of training samples to use. If None, use all available training data.
    - test_size (int, optional): Number of test samples to use. If None, use all available test data.
    - val_size (int, optional): Number of samples to reserve for validation from the training set.
    - batch_size (int): Batch size for the DataLoaders.
    - download (bool): Whether to download the data if not present.
    - root (str): Root directory for the dataset.

    Returns:
    - train_loader: DataLoader for training data.
    - val_loader: DataLoader for validation data (or None if val_size is not provided).
    - test_loader: DataLoader for test data.
    """
    # Define a simple transform; add more if necessary.
    transform = transforms.ToTensor()

    # Load the full training and test datasets
    if dataset == "MNIST":
        full_train_dataset = datasets.MNIST(
            root=root, train=True, download=download, transform=transform
        )
        full_test_dataset = datasets.MNIST(
            root=root, train=False, download=download, transform=transform
        )
    elif dataset == "CIFAR10":
        full_test_dataset = datasets.CIFAR10(
            root=root, train=False, download=download, transform=transform
        )
        full_train_dataset = datasets.CIFAR10(
            root=root, train=True, download=download, transform=transform
        )
    elif dataset == "TinyImageNET":
        full_train_dataset = datasets.ImageFolder(
            root=root + "/tiny-imagenet-200/train", transform=transform
        )
        full_test_dataset = datasets.ImageFolder(
            root=root + "/tiny-imagenet-200/val", transform=transform
        )

    # Subset the training dataset if a train_size is provided
    if train_size is not None:
        if train_size > len(full_train_dataset):
            raise ValueError("train_size exceeds available training samples.")
        train_dataset = torch.utils.data.Subset(
            full_train_dataset, list(range(train_size))
        )
    else:
        train_dataset = full_train_dataset

    # Subset the test dataset if a test_size is provided
    if test_size is not None:
        if test_size > len(full_test_dataset):
            raise ValueError("test_size exceeds available test samples.")
        test_dataset = torch.utils.data.Subset(
            full_test_dataset, list(range(test_size))
        )
    else:
        test_dataset = full_test_dataset

    # Create a validation split from the training dataset if requested
    if val_size is not None and val_size > 0:
        if val_size >= len(train_dataset):
            raise ValueError(
                "val_size must be smaller than the number of training samples."
            )
        train_length = len(train_dataset) - val_size
        train_dataset, val_dataset = random_split(
            train_dataset, [train_length, val_size]
        )
    else:
        val_dataset = None

    # Create DataLoaders for each split
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle)
    val_loader = (
        DataLoader(val_dataset, batch_size=batch_size, shuffle=shuffle)
        if val_dataset is not None
        else None
    )

    return train_loader, val_loader, test_loader


# Example usage:
if __name__ == "__main__":
    train_loader, val_loader, test_loader = load_MNIST(
        train_size=50000, test_size=10000, val_size=10000, batch_size=64
    )

    # Check one batch from the training loader
    for images, labels in train_loader:
        print("Train batch:", images.shape, labels.shape)
        break

    if val_loader is not None:
        for images, labels in val_loader:
            print("Validation batch:", images.shape, labels.shape)
            break

    for images, labels in test_loader:
        print("Test batch:", images.shape, labels.shape)
        break
