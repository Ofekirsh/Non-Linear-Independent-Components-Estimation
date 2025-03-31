import pickle
import matplotlib.pyplot as plt
import torch


def plot_train_test_loss(train_loss, test_loss, name):
    """
    Plots training and testing loss.

    Args:
        train_loss (list of float): List of training loss values.
        test_loss (list of float): List of testing loss values.

    Returns:
        None
    """
    if len(train_loss) != len(test_loss):
        raise ValueError("Train and test loss lists must have the same length.")

    epochs = range(1, len(train_loss) + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_loss, label='Train Loss', marker='o', color='blue')
    plt.plot(epochs, test_loss, label='Test Loss', marker='s', color='orange')

    plt.title(f'Train vs Test Loss {name}')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"plots/{name}", dpi=300)
    plt.show()


if __name__ == '__main__':
    # Load the file and print its content
    file_path1 = "loss_lists/loss_train_fashion-mnist_adaptive.pkl"

    with open(file_path1, 'rb') as file:
        train_loss = pickle.load(file)

    # Load the file and print its content
    file_path2 = "loss_lists/loss_test_fashion-mnist_adaptive.pkl"

    with open(file_path2, 'rb') as file:
        test_loss = pickle.load(file)

    plot_train_test_loss(train_loss, test_loss, "Fashion-MNIST Adaptive")