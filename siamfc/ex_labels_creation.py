import numpy as np

def _create_labels(size):

    def logistic_labels(x, y, r_pos, r_neg):
        dist = np.abs(x) + np.abs(y)  # block distance
        labels = np.where(dist <= r_pos,
                          np.ones_like(x),
                          np.where(dist < r_neg,
                                   np.ones_like(x) * 0.5,
                                   np.zeros_like(x)))
        return labels

    # distances along x- and y-axis
    n, c, h, w = 1, 1, 17, 17
    x = np.arange(w) - (w - 1) / 2
    y = np.arange(h) - (h - 1) / 2
    x, y = np.meshgrid(x, y)

    # create logistic labels
    r_pos = 16 / 8
    r_neg = 0 / 8
    labels = logistic_labels(x, y, r_pos, r_neg)

    # repeat to size
    labels = labels.reshape((1, 1, h, w))
    labels = np.tile(labels, (n, c, 1, 1))

    return labels

if __name__ == "__main__":
    labels = _create_labels(17)
    a = 0