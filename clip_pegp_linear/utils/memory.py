import torch
import numpy as np


def get_representation_matrix(data_loader, device):
    count = 1
    representation = []
    for tasks, inputs, targets in data_loader:
        inputs = inputs.to(device, non_blocking=True)
        representation.append(inputs)
        count += 1
        if count > 16:
            representation = torch.cat(representation)
            break
    return representation


def update_memory(representations, threshold, features=None):
    for i in range(len(representations)):
        representation = representations[i]
        feature = features[i]
        representation = np.matmul(representation.T, representation)
        if feature is None:
            U, S, Vh = np.linalg.svd(representation.astype(np.float32), full_matrices=False)
            sval_total = (S ** 2).sum()
            sval_ratio = (S ** 2) / sval_total
            r = np.sum(np.cumsum(sval_ratio) < threshold)
            feature = U[:, 0:r]
        else:
            U1, S1, Vh1 = np.linalg.svd(representation.astype(np.float32), full_matrices=False)
            sval_total = (S1 ** 2).sum()
            # Projected Representation
            act_hat = representation - np.dot(np.dot(feature, feature.transpose()), representation)
            U, S, Vh = np.linalg.svd(act_hat.astype(np.float32), full_matrices=False)
            # criteria
            sval_hat = (S ** 2).sum()
            sval_ratio = (S ** 2) / sval_total
            accumulated_sval = (sval_total - sval_hat) / sval_total
            r = 0
            for ii in range(sval_ratio.shape[0]):
                if accumulated_sval < threshold:
                    accumulated_sval += sval_ratio[ii]
                    r += 1
                else:
                    break
            if r != 0:
                U = np.hstack((feature, U[:, 0:r]))
                if U.shape[1] > U.shape[0]:
                    feature = U[:, 0:U.shape[0]]
                else:
                    feature = U
        features[i] = feature

        print('-'*40)
        print('Gradient Constraints Summary', feature.shape)
        print('-'*40)

    return features
