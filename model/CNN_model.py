import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):

    def __init__(
        self,
        conv_nodes_1,
        conv_nodes_2,
        kernel_size_1,
        kernel_size_2,
        maxpool_size,  # Used when using standard max pooling
        pooling_strategy,  # "max" for standard max pooling, "mac" for global MAC pooling
        dropout_rate,
        fc_nodes,
        input_shape=(3, 32, 32),  # Default for MNIST; change for other datasets
    ):
        super(CNN, self).__init__()
        self.input_shape = input_shape

        # Convolutional layers
        self.conv1 = nn.Conv2d(
            in_channels=input_shape[0],
            out_channels=conv_nodes_1,
            kernel_size=kernel_size_1,
            padding=(kernel_size_1 - 1) // 2,
        )
        self.conv2 = nn.Conv2d(
            in_channels=conv_nodes_1,
            out_channels=conv_nodes_2,
            kernel_size=kernel_size_2,
            padding=(kernel_size_2 - 1) // 2,
        )

        # Choose pooling layer based on pooling_strategy
        self.pooling_strategy = pooling_strategy.lower()
        if self.pooling_strategy == "mac":
            # Global MAC pooling: output size will be (batch, channels, 1, 1)
            self.pool = nn.AdaptiveMaxPool2d((1, 1))
        else:
            # Standard max pooling with fixed kernel size and stride
            self.pool = nn.MaxPool2d(kernel_size=maxpool_size, stride=maxpool_size)

        # Dropout for regularization
        self.dropout = nn.Dropout(dropout_rate)

        # Dynamically determine the flattened feature size for the first FC layer.
        dummy_input = torch.rand(1, *input_shape)
        fc_input_dim = self._get_conv_output(dummy_input)
        self.fc1 = nn.Linear(fc_input_dim, fc_nodes)
        self.fc2 = nn.Linear(
            fc_nodes, 10
        )  # Adjust if you have a different number of classes

    def _get_conv_output(self, x):
        """Pass a dummy input through conv layers and pooling to calculate flattened dimension."""
        with torch.no_grad():
            x = F.relu(self.conv1(x))
            x = F.relu(self.conv2(x))
            x = self.pool(x)
            return x.view(1, -1).shape[1]

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.dropout(x)
        x = x.view(x.size(0), -1)  # Flatten dynamically
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


def train(model, device, train_loader, test_loader, optimizer, num_epochs):
    """
    Trains the model for a given number of epochs.

    Args:
        model (torch.nn.Module): The model to train.
        device (torch.device): Device on which to perform training (CPU or GPU).
        train_loader (DataLoader): DataLoader for the training dataset.
        optimizer (torch.optim.Optimizer): Optimizer for updating the model's parameters.
        num_epochs (int): Number of epochs to train.
        log_interval (int, optional): Interval (in batches) to log training status.
        test_loader (DataLoader): The model is evaluated on test data after training.
    """

    def evaluate(model, device, test_loader):
        """
        Evaluates the model on test data and returns the accuracy.

        Args:
            model (torch.nn.Module): The trained model.
            device (torch.device): Device on which to perform evaluation.
            test_loader (DataLoader): DataLoader for the test dataset.

        Returns:
            float: The accuracy of the model on the test dataset.
        """
        model.eval()  # Set the model to evaluation mode
        correct = 0
        loss = 0

        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                # Sum up batch loss (using sum to later compute the average loss)
                loss += F.cross_entropy(output, target, reduction="sum").item()

                # Get the index of the max log-probability (predicted class)
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        accuracy = correct / len(test_loader.dataset)
        loss = loss / len(test_loader.dataset)

        return accuracy, loss

    model.train()  # Set the model to training mode
    for epoch in range(1, num_epochs + 1):

        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()  # Clear previous gradients
            output = model(data)  # Forward pass
            loss = F.cross_entropy(output, target)  # Compute loss
            loss.backward()  # Backward pass (compute gradients)
            optimizer.step()  # Update model parameters

    # If a test_loader is provided, evaluate the model after each epoch
    accuracy, loss = evaluate(model, device, test_loader)
    return accuracy, loss
