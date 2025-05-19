import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import auc

def compute_recall_at_k(labels, dists, k='k'):
    """
    Compute the recall at k retrievals for each sample in the dataset.
    Args:
        labels: numpy array of shape (n_samples,) containing the labels of the samples
        dists: numpy array of shape (n_samples, n_samples) containing the distances between samples
        k: int or 'k', if int, the number of nearest neighbors to consider, if 'k', the number of nearest neighbors is
           determined by the number of samples with the same label as the query sample
    Returns:
        prec_at_k: numpy array of shape (n_samples,) containing the precision at k for each sample
        rec_at_k: numpy array of shape (n_samples,) containing the recall at k for each sample
    """
    prec_at_k = np.zeros(len(labels))
    rec_at_k = np.zeros(len(labels))

    for i in range(len(labels)):
        if k == 'k':
            _k = np.sum(labels == labels[i]) - 1
        else:
            _k = k
        # get the indices of the k nearest neighbors
        idx = np.argsort(dists[i])[:_k]
        # get the labels of the k nearest neighbors
        nn_labels = labels[idx]
        # count the number of relevant retrieved samples
        n_relevant_retrieved = np.sum(nn_labels == labels[i])
        # count the number of relevant samples
        n_relevant = np.sum(labels == labels[i]) - 1
        # compute the precision at k
        prec_at_k[i] =  n_relevant_retrieved / _k
        # compute the recall at k
        rec_at_k[i] = n_relevant_retrieved / n_relevant
    return prec_at_k, rec_at_k

def plot_precision_recall_curve(x, y, filename):
    """
    Plot the precision-recall curve.
    Args:
        x: numpy array of shape (n_samples,) containing the recall values
        y: numpy array of shape (n_samples,) containing the precision values
        filename: string, path to save the plot
    Returns:
        None
    """
    idx = np.argsort(x)
    x = x[idx]
    y = y[idx]
    x = [0] + list(x) + [1]
    y = [1] + list(y) + [0]
    area = auc(x, y)
    plt.figure(figsize=(10, 5))
    plt.plot(x, y, label=f"Area under the curve: {area:.2f}", marker="o", linestyle="--", linewidth=2)
    plt.xlabel("Recall", fontsize=15)
    plt.ylabel("Precision", fontsize=15)
    plt.legend(fontsize=15)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    return area

def retrieve_filenames(query, labels, filenames, dists):
    """
    Retrieve the filenames of the k nearest neighbors of a query sample.
    """
    filenames = np.array(filenames)
    
    i = np.where(filenames == query)[0][0]
    
    _k = np.sum(labels == labels[i]) - 1
    # get the indices of the k nearest neighbors
    idx = np.argsort(dists[i])[:_k]
    # get the labels of the k nearest neighbors
    nn_labels = labels[idx]
    # get the filenames of the k nearest neighbors
    retrieved_filenames  = [filenames[j] for j in idx]
    return retrieved_filenames
