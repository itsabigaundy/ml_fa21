import matplotlib.pyplot as plt
import numpy as np
import utils

from scipy.spatial.distance import cdist
from sklearn import random_projection
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA, FastICA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture


def plotSilhouetteScore(scores):
    plt.plot(range(2, len(scores) + 2), scores, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Silhouette Score')
    plt.show()


def chooseKMeans(data, hi=10):
    distortions = []
    inertias = []
    silhouettes = []

    for k in range(1, hi + 1):
        model = KMeans(k)
        model.fit(data)

        distortions.append(sum(np.min(cdist(data, model.cluster_centers_, 'euclidean'), axis=1)) / data.shape[0])
        inertias.append(model.inertia_)
        if k >= 2:
            silhouettes.append(silhouette_score(data, model.labels_))
    
    for name, data in zip(('Distortion', 'Inertia'), (distortions, inertias)):
        plt.plot(range(1, hi + 1), data, 'bx-')
        plt.xlabel('k')
        plt.ylabel(name)
        plt.show()

    plotSilhouetteScore(silhouettes)


def chooseEM(data, hi=10):
    aics = []
    bics = []
    silhouettes = []

    for k in range(1, hi + 1):
        print(k)
        temp_aic = []
        temp_bic = []
        temp_scores = []
        for i in range(50):
            model = GaussianMixture(k, reg_covar=1e-5)
            labels = model.fit_predict(data)

            temp_aic.append(model.aic(data))
            temp_bic.append(model.bic(data))
            if k >= 2:
                temp_scores.append(silhouette_score(data, labels))

        aics.append(np.mean(temp_aic))
        bics.append(np.mean(temp_bic))
        if k >= 2:
            silhouettes.append(np.mean(temp_scores))

    for name, data in zip(('AIC', 'BIC'), (aics, bics)):
        plt.plot(range(1, hi + 1), data, 'bx-')
        plt.xlabel('k')
        plt.ylabel(name)
        plt.show()

    plotSilhouetteScore(silhouettes)

    print(np.argmin(aics) + 1, np.argmin(bics) + 1, np.argmax(silhouettes) + 2)


# kmeans and expectation maximization
# em = GaussianMixture()

#pca, ica, randomized projections, lda
# pca = PCA()
# ica = FastICA()
# rp = random_projection()
# lda = LinearDiscriminantAnalysis()


churn_data, churn_labels = utils.getChurnData()
stroke_data, stroke_labels = utils.getStrokeData()

# chooseKMeans(churn_data)
# chooseKMeans(stroke_data)

chooseEM(churn_data, 50)
chooseEM(stroke_data, 50)

# visualizer = TSNE()
# churn_tsne = visualizer.fit_transform(churn_data)
# stroke_tsne = visualizer.fit_transform(stroke_data)

# kmeans = KMeans(4)
# churn_predict = kmeans.fit_predict(churn_data)

# plt.scatter(churn_tsne[:,0], churn_tsne[:,1], c=churn_predict)
# plt.show()

# kmeans = KMeans(2)
# stroke_predict = kmeans.fit_predict(stroke_data)

# plt.scatter(stroke_tsne[:,0], stroke_tsne[:,1], c=stroke_predict)
# plt.show()