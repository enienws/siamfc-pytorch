import pickle
import numpy as np


if __name__ == "__main__":
    with open("/opt/project/resnet_v4.pickle", "rb") as f:
        features = pickle.load(f)

    # features = [x.cpu().numpy() for x in features]
    features = np.stack(np.array(features), axis=0)
    features = np.reshape(features, (-1, 1, 15, 15))
    mean = np.mean(features)
    variance = np.std(features)
    features = (features - mean) / variance
    a = 0