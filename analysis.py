from kmeans import get_kmeans, assign_clusters
from sklearn.metrics import silhouette_score
from symnmf import symnmf_clustering, get_clusters_from_H
import pandas
import sys

def main(*args):
    try:
        k = int(args[0])
        df = pandas.read_csv(args[1], header=None)

        X = df.to_numpy()
        
        centroids_kmeans = get_kmeans(X, k, formatted=False)

        kmeans_cluster_assign = assign_clusters(X, centroids_kmeans)
        H = symnmf_clustering(X, k)
        symnmf_cluster_assign = get_clusters_from_H(H)
    
        score_symnmf = silhouette_score(X, symnmf_cluster_assign)
        score_kmeans = silhouette_score(X, kmeans_cluster_assign)

        print(f'nmf: {score_symnmf:.4f}')
        print(f'kmeans: {score_kmeans:.4f}')

    except Exception as e:
        print("An Error Has Occurred")
        exit(1)
    

if __name__ == "__main__":
    main(*sys.argv[1:])
