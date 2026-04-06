import numpy as np
import sys
import pandas
import symnmfmodule

def init_H(W : list[list], k):
    W_np = np.array(W)
    m = np.mean(W_np)
    n = W_np.shape[0]
    np.random.seed(1234)
    H = np.random.uniform(0,2*((m/k)**0.5), (n,k))
    return H

def get_clusters_from_H(H: np.ndarray) -> np.ndarray:
    # argmax along the row gives the cluster index
    clusters = np.argmax(H, axis=1)
    return clusters

def print_formatted(mat):
    print('\n'.join([','.join([f"{x:.4f}" for x in line]) for line in mat]))

def symnmf_clustering(X: np.ndarray, k: int, goal='symnmf'):
    try:
        n, d = X.shape

        # --- Step 1: Similarity matrix ---
        A = symnmfmodule.sym(X.tolist(), n, d)
        if goal == 'sym':
            return A
        
        # --- Step 2: Diagonal Degree matrix ---
        D = symnmfmodule.ddg(A, n)
        if goal == 'ddg':
            return D

        # --- Step 3: Normalized similarity matrix ---
        W = symnmfmodule.norm(A, D, n)
        if goal == 'norm':
            return W

        # --- Step 4: Initialize H ---
        H_init = init_H(W, k)

        # --- Step 5: Run SymNMF ---
        H = symnmfmodule.symnmf(H_init.tolist(), W, n, k)
        
        if goal == 'symnmf':
            return H
        else:
            # goal is invalid
            print("An Error Has Occurred")
            exit(1)
    except:
        print("An Error Has Occurred")
        exit(1)
    
def main(*args):
    try:
        k = int(args[0])
        goal = args[1]
        df = pandas.read_csv(args[2], header=None)
        X = df.to_numpy()
    
        res_Matrix = symnmf_clustering(X, k, goal)
        print_formatted(res_Matrix)
    
    except Exception as e:
        print("An Error Has Occurred")
        exit(1)

if __name__ == "__main__":
    main(*sys.argv[1:])