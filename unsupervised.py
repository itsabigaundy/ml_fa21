import matplotlib.pyplot as plt
import numpy as np
import utils

from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA, FastICA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.random_projection import SparseRandomProjection


def plotSilhouetteScore(name: str, cluster: str, scores: np.array):
    plt.plot(range(2, len(scores) + 2), scores, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Silhouette Score')
    plt.savefig('unsupervised_images/' + name + '_' + cluster + '_' + 'sil.png')
    plt.clf()


def chooseKMeans(X, problem_name: str, hi: int=10):
    distortions = []
    inertias = []
    silhouettes = []

    for k in range(1, hi + 1):
        model = KMeans(k)
        model.fit(X)

        distortions.append(sum(np.min(cdist(X, model.cluster_centers_, 'euclidean'), axis=1)) / X.shape[0])
        inertias.append(model.inertia_)
        if k >= 2:
            silhouettes.append(silhouette_score(X, model.labels_))
    
    for name, X in zip(('Distortion', 'Inertia'), (distortions, inertias)):
        plt.plot(range(1, hi + 1), X, 'bx-')
        plt.xlabel('k')
        plt.ylabel(name)
        plt.savefig('unsupervised_images/' + problem_name + '_k_' + name + '.png')
        plt.clf()

    plotSilhouetteScore(problem_name, 'k', silhouettes)


def chooseEM(X, problem_name: str, hi: int=10):
    aics = []
    bics = []
    silhouettes = []

    for k in range(1, hi + 1):
        print(k)
        temp_aic = []
        temp_bic = []
        temp_scores = []
        for i in range(50):
            model = GaussianMixture(k) #, reg_covar=1e-5)
            labels = model.fit_predict(X)

            temp_aic.append(model.aic(X))
            temp_bic.append(model.bic(X))
            if k >= 2:
                temp_scores.append(silhouette_score(X, labels))

        aics.append(np.mean(temp_aic))
        bics.append(np.mean(temp_bic))
        if k >= 2:
            silhouettes.append(np.mean(temp_scores))

    for name, X in zip(('AIC', 'BIC'), (aics, bics)):
        plt.plot(range(1, hi + 1), X, 'bx-')
        plt.xlabel('k')
        plt.ylabel(name)
        plt.savefig('unsupervised_images/' + problem_name + '_em_' + name + '.png')
        plt.clf()

    plotSilhouetteScore(problem_name, 'em', silhouettes)

    print(np.argmin(aics) + 1, np.argmin(bics) + 1, np.argmax(silhouettes) + 2)


def visualizeClusters(X, k: int, cluster_cls, problem_name: str, cluster_name: str):
    visualizer = TSNE()
    X_tsne = visualizer.fit_transform(X)

    model = cluster_cls(k)
    clusters = model.fit_predict(X)

    plt.scatter(X_tsne[:,0], X_tsne[:,1], c=clusters)
    plt.savefig('unsupervised_images/' + '_'.join([problem_name, cluster_name, 'TSNE']) + '.png')
    plt.clf()


def dimensionReduction(X):
    methods = {
        'pca' : PCA(n_components=.95, svd_solver='full'),
        'ica' : FastICA(),
        'srp' : SparseRandomProjection(),
        'lda' : LinearDiscriminantAnalysis(),
    }
    
    return {key : methods[key].fit_transform(X) for key in methods}


# kmeans and expectation maximization
# em = GaussianMixture()

#pca, ica, randomized projections, lda
# pca = PCA()
# ica = FastICA()
# rp = random_projection()
# lda = LinearDiscriminantAnalysis()


# chooseKMeans(churn_data)
# chooseKMeans(stroke_data)

# chooseEM(churn_data, 50)
# chooseEM(stroke_data, 50)

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


if __name__ == "__main__":
    szeged = 'szeged'
    epi = 'epi'
    szeged_data = utils.getSzegedData()
    epi_data = utils.getEpicuriousData()

    med = szeged_data['Temperature (C)'].median()
    szeged_X = szeged_data.drop( [ 'Temperature (C)', ], axis=1)
    szeged_y = np.array([x >= med for x in szeged_data['Temperature (C)']])

    epi_X = epi_data.drop( [ 'healthy', ], axis=1)
    epi_y = epi_data['healthy']

    # chooseKMeans(szeged_X, szeged)
    # chooseKMeans(epi_X, epi)

    # chooseEM(szeged_X, szeged)
    # chooseEM(epi_X, epi)

    visualizeClusters(szeged_X, 2, KMeans, szeged, 'k')
    visualizeClusters(epi_X, 4, KMeans, epi, 'k')

    szeged_red = dimensionReduction(szeged_X)
    epi_red = dimensionReduction(epi_X)

    for key in szeged_red:
        chooseKMeans(szeged_red[key], szeged + '_' + key)
        chooseKMeans(epi_red[key], epi + '_' + key)