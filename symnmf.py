import numpy as np
import sys
import pandas
import symnmfmodule
# calculate similarity matrix A (nxn) given input X (nxd)
def get_similarity_matrix(X : np.ndarray):
    dist_matrix = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=2)
    A = -((dist_matrix)**2)/2
    A = np.exp(A)
    np.fill_diagonal(A, 0)
    return A

def get_diagonal_degree_matrix(A : np.ndarray):
    D = np.diag(np.sum(A, axis=0))

    return D

def get_normalized_similarity_matrix(A : np.ndarray, D : np.ndarray):
    D_inv_sqrt = np.diag(1.0 / np.sqrt(np.diag(D)))
    W = D_inv_sqrt@A@D_inv_sqrt
    return W

def init_H(W : np.ndarray, k):
    m = np.mean(W)
    n = W.shape[0]
    np.random.seed(1234)
    H = np.random.uniform(0,2*((m/k)**0.5), (n,k))
    return H

def get_clusters_from_H(H: np.ndarray) -> np.ndarray:
    # argmax along the row gives the cluster index
    clusters = np.argmax(H, axis=1)
    return clusters

def symnmf_numpy_cstyle(W, H_init, beta=0.5, epsilon=1e-4, max_iter=300):
    """
    SymNMF following the C module update order exactly
    Args:
        W: (n,n) symmetric similarity matrix
        H_init: (n,k) initial nonnegative matrix
        beta: damping factor
        epsilon: convergence threshold
        max_iter: max number of iterations
    Returns:
        H: (n,k) nonnegative factor matrix
    """
    H = H_init.copy()
    n, k = H.shape

    for _ in range(max_iter):
        # Step 1: W_H = W @ H
        W_H = W @ H

        # Step 2: H_Ht_H = H @ H.T @ H
        H_Ht_H = H @ H.T @ H

        # Step 3: elementwise reciprocal of H_Ht_H (avoid div by zero)
        H_Ht_H = 1.0 / H_Ht_H

        # Step 4: elementwise multiply with W_H
        W_H *= H_Ht_H

        # Step 5: damping
        W_H = beta * W_H 


        # Step 6: final H update
        H_new = H * (W_H + 1-beta)

        # Step 7: convergence check
        if np.linalg.norm(H_new - H, 'fro')**2 < epsilon:
            H = H_new
            break

        H = H_new


    return H


def print_formatted(mat):
    print('\n'.join([','.join([f"{x:.4f}" for x in line]) for line in mat]))

def symnmf_clustering(X: np.ndarray, k: int, use_c_module=False):
    n, d = X.shape

    # --- Step 1: Similarity matrix ---
    if use_c_module:
        A = np.array(symnmfmodule.sym(X.tolist(), n, d))
    else:
        A = get_similarity_matrix(X)

    # --- Step 2: Degree matrix ---
    if use_c_module:
        D = np.array(symnmfmodule.ddg(A.tolist(), n))
    else:
        D = get_diagonal_degree_matrix(A)

    # --- Step 3: Normalized similarity ---
    if use_c_module:
        W = np.array(symnmfmodule.norm(A.tolist(), D.tolist(), n))
    else:
        W = get_normalized_similarity_matrix(A, D)

    # --- Step 4: Initialize H ---
    H_init = init_H(W, k)

    # --- Step 5: Run SymNMF ---
    if use_c_module:
        H = np.array(symnmfmodule.symnmf(H_init.tolist(), W.tolist(), n, k))
    else:
        H = symnmf_numpy_cstyle(W, H_init)

    # --- Step 6: Extract clusters ---
    clusters = get_clusters_from_H(H)

    return clusters, H

def main(*args):
    k = int(args[0])
    goal = args[1]
    df = pandas.read_csv(args[2], header=None)

    X = df.to_numpy()

    n = X.shape[0]
    d = X.shape[1]
    
    A = symnmfmodule.sym(X.tolist(), n, d)
    A_py = get_similarity_matrix(X)
    if goal == 'sym':
        print_formatted(A)
        return
    
    D = symnmfmodule.ddg(A, n)
    D_py = get_diagonal_degree_matrix(A_py)
    
    if goal == 'ddg':
        print_formatted(D)
        return
    
    W = symnmfmodule.norm(A, D, n)
    W_py = get_normalized_similarity_matrix(A_py, D_py)
    if goal == 'norm':
        print_formatted(W)
        return
    
    H = init_H(np.array(W), k)
    H_copy = H.copy()
    W_copy = W.copy()

    try:
        H_res = symnmfmodule.symnmf(H.tolist(),W, n, k)
    
    except Exception as e:
        print(e)
        exit(1)
    
    
    print('\n'.join([','.join([f"{x:.4f}" for x in line]) for line in H_res]))

    print('***********************************\nnumpy algo\n***********************************')
    H_res_test = symnmf_numpy_cstyle(W_py, H_copy)
    print('\n'.join([','.join([f"{x:.4f}" for x in line]) for line in H_res_test]))


if __name__ == "__main__":
    main(*sys.argv[1:])
#add try and execpt
#add try and execpt
#add try and execpt
#add try and execpt
#add try and execpt
#add try and execpt
#add try and execpt
#add try and execpt
#add try and execpt
#add try and execpt
#add try and execpt
#add try and execpt
#add try and execpt
#add try and execpt
#add try and execpt
#add try and execpt
#add try and execpt
#add try and execpt
#add try and execpt
#add try and execpt
#add try and execpt
#add try and execpt
#add try and execpt
#add try and execpt
#add try and execpt
#add try and execpt
#add try and execpt
#add try and execpt