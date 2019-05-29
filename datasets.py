from torchvision.datasets import MNIST, CIFAR10


def dataset_getter():

    train_ds = MNIST(
        data_path, 
        train=True, 
        download=True,
        transform=T.Compose([
            T.RandomAffine(5, translate=(0.05, 0.05), scale=(0.9, 1.1)),
            T.ToTensor(),
            T.Normalize(*mnist_stats)
        ])
    )

    valid_ds = MNIST(
        data_path, 
        train=False, 
        transform=T.Compose([
            T.ToTensor(),
            T.Normalize(*mnist_stats)
        ])
    )
    return train_ds, valid_ds