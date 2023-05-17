"""
Code from https://github.com/liyuan9988/LinGapE.
"""

import os
import gzip
import numpy as np


def encode_line(line):
    user_id = line.index("|user")
    user_raw_feature = line[user_id + 1:user_id + 7]
    user_feature = np.zeros(6)
    for a in user_raw_feature:
        a = a.split(":")
        if(int(a[0]) == 7):
            break
        user_feature[int(a[0]) - 1] = float(a[1])
    article_id = line[1]
    article_feature_id = line.index("|" + article_id)
    article_raw_feature = line[article_feature_id + 1:article_feature_id + 7]
    article_feature = np.zeros(6)
    for a in article_raw_feature:
        a = a.split(":")
        if(int(a[0]) == 7):
            break
        else:
            article_feature[int(a[0]) - 1] = float(a[1])
    feature = np.outer(user_feature, article_feature)
    return np.reshape(feature,36)


def PCA_SVD(X, num_components):
    mean_X = np.mean(X, axis=0)
    normalized_X = X - mean_X
    U, S, Vt = np.linalg.svd(normalized_X, full_matrices=False)

    Vt_subset = Vt[:num_components, :]
    X_reduced = X @ Vt_subset.T

    return X_reduced, S


if __name__ == "__main__":
    days = [f'0{i + 1}' for i in range(7)]
    d_new = 24

    if not os.path.exists("yahoo_features"):
        os.makedirs("yahoo_features")
    if not os.path.exists("yahoo_features_pca"):
        os.makedirs("yahoo_features_pca")
    if not os.path.exists("yahoo_targets"):
        os.makedirs("yahoo_targets")

    for day in days:
        f = gzip.open(f"ydata-fp-td-clicks-v1_0.200905{day}.gz")
        nrow = 0
        for i in f:
            nrow += 1
        f.close()

        f = gzip.open(f"ydata-fp-td-clicks-v1_0.200905{day}.gz")
        X = np.empty((nrow, 36))
        y = np.empty(nrow)
        for i, line in enumerate(f):
            line = line.decode()
            line = line.rstrip()
            line = line.split(" ")
            X[i] = encode_line(line)
            y[i] = int(line[2])
        f.close()

        keep_idx = np.where(X[:, 0] > 0)[0]
        X = X[keep_idx]
        y = y[keep_idx]

        pca_X, singular_values = PCA_SVD(X, d_new)
        print(f"Singluar values for day {day}: {singular_values}")

        np.save(f"yahoo_features/yahoo_features_{day}.npy",X)
        np.save(f"yahoo_targets/yahoo_targets_{day}.npy",y)
        np.save(f"yahoo_features_pca/yahoo_features_{day}_pca.npy", pca_X)