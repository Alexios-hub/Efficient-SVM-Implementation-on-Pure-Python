import numpy as np
import matplotlib.pyplot as plt
from SVM import SVM


def generate_linearly_separable_data(w, b, n, d, separation=0.1):
    """
    Generate linearly separable data with larger separation.

    Parameters:
    n (int): Number of samples.
    d (int): Number of features.
    separation (float): Amount of separation to be added between the two classes.

    Returns:
    X (numpy.ndarray): Generated samples.
    y (numpy.ndarray): Labels of the samples.
    """

    # Initialize samples array
    X = np.zeros((n, d))

    # Create labels and adjust the samples based on the hyperplane
    labels = np.zeros(n)
    for i in range(n):
        # Randomly generate a point
        point = np.random.randn(d)

        # Assign label based on the side of the hyperplane
        label = np.sign(np.dot(point, w) + b)

        # Adjust the point to maintain separation
        point += separation * label * w

        # Store the adjusted point and label
        X[i, :] = point
        labels[i] = label

    return X, labels


# Function to plot a hyperplane given weights and bias
def plot_hyperplane(w, b, color, label):
    x = np.linspace(-3, 3, 10)
    y = (-w[0] * x - b) / w[1]
    plt.plot(x, y, color=color, label=label)


def downsample_data(X, y, downsample_ratio):
    """
    Downsample the dataset while maintaining the class balance.

    Parameters:
    X (numpy.ndarray): Input features.
    y (numpy.ndarray): Labels.
    downsample_ratio (float): Proportion of the data to keep.

    Returns:
    Downsampled X and y.
    """
    # Separate by class
    pos_indices = np.where(y == 1)[0]
    neg_indices = np.where(y == -1)[0]

    # Calculate the number of samples to keep for each class
    n_pos_samples = int(len(pos_indices) * downsample_ratio)
    n_neg_samples = int(len(neg_indices) * downsample_ratio)

    # Downsample each class separately
    pos_samples = np.random.choice(pos_indices, n_pos_samples, replace=False)
    neg_samples = np.random.choice(neg_indices, n_neg_samples, replace=False)

    # Combine and shuffle
    downsampled_indices = np.hstack((pos_samples, neg_samples))
    np.random.shuffle(downsampled_indices)

    return X[downsampled_indices], y[downsampled_indices]


def remove_distant_points(X, y, model, threshold=1):
    """
    Remove data points that are far from the decision boundary.

    Parameters:
    X (numpy.ndarray): Input features.
    y (numpy.ndarray): Labels.
    model (SVM): Trained SVM model.
    threshold (float): Distance threshold to decide if a point is far.

    Returns:
    Refined X and y.
    """
    _, scores = model.predict(X)
    margin_distances = np.abs(scores)
    close_indices = margin_distances < threshold

    return X[close_indices], y[close_indices]


def train(x_train, y_train, kkt_thr=1e-3, max_iter=1e3, distance_threshold=1):
    """
    train a linear seperateble dataset and return a SVM model

    Parameters:
    x_train:[N,d]
    y_train:[N,]
    kkt_thr: kkt tolerance,default set to 1e-3
    max_iter:max iterations,default set to 1e3

    Returns:
    a SVM model
    """
    N = float(x_train.shape[0])
    downsample_ratio = 1e3 / N
    if downsample_ratio < 1:
        while x_train.shape[0] * downsample_ratio > 1e4:
            downsample_ratio = downsample_ratio * downsample_ratio
        print(f"Downsampling dataset with ratio:{downsample_ratio}")
        x_train_downsampled, y_train_downsampled = downsample_data(
            x_train, y_train, downsample_ratio
        )
        print(f"training on the downsampled dataset")
        # Fit the SVM model
        model = SVM(
            kkt_thr=kkt_thr,
            max_iter=max_iter,
            x_train=x_train_downsampled,
            y_train=y_train_downsampled,
        )
        model.fit()
        print(f"filtering original dataset")
        x_train_filtered, y_train_filtered = remove_distant_points(
            x_train, y_train, model, distance_threshold
        )
        print(f"{x_train.shape[0]-x_train_filtered.shape[0]} samples removed")
        print(f"training on the filtered dataset")
        model = SVM(
            kkt_thr=kkt_thr,
            max_iter=max_iter,
            x_train=x_train_filtered,
            y_train=y_train_filtered,
        )
        model.fit()
    else:
        model = SVM(
            kkt_thr=kkt_thr, max_iter=max_iter, x_train=x_train, y_train=y_train
        )
        model.fit()
    return model
