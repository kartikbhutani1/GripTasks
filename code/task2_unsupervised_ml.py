import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans


def load_data(path):
    iris = pd.read_csv(path)
    iris_features = iris.iloc[:, [1, 2, 3, 4]]
    iris_labels = iris.iloc[:, 5]
    print(iris_features.head())
    return iris_features, iris_labels


def perform_wcss(X, y):
    wcss = []
    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i, init='k-means++',
                        max_iter=300, n_init=10, random_state=0)
        kmeans.fit(X)
        wcss.append(kmeans.inertia_)

    plt.plot(range(1, 11), wcss)
    plt.title('The elbow method')
    plt.xlabel('Number of clusters')
    plt.ylabel('WCSS')  # Within cluster sum of squares
    plt.savefig(f'{output_dir}task2_wcss_method.png')
    plt.close()
    # plt.show()


def apply_kmeans(X, y, clusters):
    kmeans = KMeans(n_clusters=clusters, init='k-means++',
                    max_iter=300, n_init=10, random_state=0)
    y_kmeans = kmeans.fit_predict(X)
    return kmeans, y_kmeans


def visualize_clusters(X, kmeans, y_kmeans):
    plt.scatter(X.iloc[y_kmeans == 0, 0], X.iloc[y_kmeans == 0, 1],
                s=100, c='red', label='Iris-setosa')
    plt.scatter(X.iloc[y_kmeans == 1, 0], X.iloc[y_kmeans == 1, 1],
                s=100, c='blue', label='Iris-versicolour')
    plt.scatter(X.iloc[y_kmeans == 2, 0], X.iloc[y_kmeans == 2, 1],
                s=100, c='green', label='Iris-virginica')

    # Plotting the centroids of the clusters
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
                s=100, c='yellow', label='Centroids')

    plt.legend()
    plt.savefig(f'{output_dir}task2_kmeans_clustering.png')
    # plt.show()


def iris_optimal_centres_task():
    path = "./datasets/Iris.csv"
    X, y = load_data(path)
    perform_wcss(X, y)
#     Choosing the number of clusters as 3
    clusters = 3
    kmeans, y_kmeans = apply_kmeans(X, y, clusters)
    visualize_clusters(X, kmeans, y_kmeans)


if __name__ == "__main__":
    output_dir = './outputs/'
    iris_optimal_centres_task()
