from kmeans import get_kmeans, assign_clusters
from sklearn.metrics import silhouette_score
from symnmf import symnmf_clustering
import pandas
import sys

def main(*args):
    k = int(args[0])
    df = pandas.read_csv(args[1], header=None)

    X = df.to_numpy()
    centroids_kmeans = get_kmeans(X, k, formatted=False)
    kmeans_cluster_assign = assign_clusters(X, centroids_kmeans)
    symnmf_cluster_assign, H = symnmf_clustering(X, k, use_c_module=True)

    score_symnmf = silhouette_score(X, symnmf_cluster_assign)
    score_kmeans = silhouette_score(X, kmeans_cluster_assign)

    print(f'nmf: {score_symnmf:.4f}')
    print(f'kmeans: {score_kmeans:.4f}')
    

if __name__ == "__main__":
    main(*sys.argv[1:])
