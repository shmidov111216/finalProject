import numpy as np
import sys
import pandas
import symnmfmodule

def init_H(W : list[list], k):
    """
    Initialize the H matrix for SymNMF.

    Parameters
    ----------
    W : list[list]
        Normalized similarity matrix.
    k : int
        Number of clusters.

    Returns
    -------
    np.ndarray
        Initialized H matrix (n x k).
    """
    W_np = np.array(W)
    m = np.mean(W_np)
    n = W_np.shape[0]
    np.random.seed(1234)
    H = np.random.uniform(0,2*((m/k)**0.5), (n,k))
    return H

def get_clusters_from_H(H: np.ndarray) -> np.ndarray:
    """
    Assign clusters using the maximum value in each row of H.

    Parameters
    ----------
    H : np.ndarray
        SymNMF result matrix.

    Returns
    -------
    np.ndarray
        Cluster indices.
    """
    clusters = np.argmax(H, axis=1)
    return clusters

def print_formatted(mat):
    """
    Print a matrix with comma-separated values.

    Parameters
    ----------
    mat : array-like
        Matrix to print.
    """
    print('\n'.join([','.join([f"{x:.4f}" for x in line]) for line in mat]))

def symnmf_clustering(X: np.ndarray, k: int, goal='symnmf'):
    """
    Run SymNMF or return an intermediate matrix.

    Parameters
    ----------
    X : np.ndarray
        Data matrix.
    k : int
        Number of clusters.
    goal : str
        {'sym', 'ddg', 'norm', 'symnmf'}.

    Returns
    -------
    array-like
        Result matrix.
    """ 
    try:
        n, d = X.shape

        A = symnmfmodule.sym(X.tolist(), n, d) # Step 1: Similarity matrix
        if goal == 'sym':
            return A 
        D = symnmfmodule.ddg(A, n) # Step 2: Diagonal Degree matrix
        if goal == 'ddg':
            return D
        W = symnmfmodule.norm(A, D, n) # Step 3: Normalized similarity matrix
        if goal == 'norm':
            return W
        H_init = init_H(W, k) # Step 4: Initialize H
        H = symnmfmodule.symnmf(H_init.tolist(), W, n, k) # Step 5: Run SymNMF
        if goal == 'symnmf':
            return H
    except Exception as e:
        print('An Error Has Occurred')
        exit(1)
    
def main(*args):
    """
    Run the program from command-line arguments.

    Parameters
    ----------
    args : tuple
        (k, goal, input_file)
    """
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