import numpy as np
import sys


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
    D_inv_sqrt = np.linalg.inv(np.sqrt(D))
    W = D_inv_sqrt@A@D_inv_sqrt
    return W

def init_H(W : np.ndarray, k):
    m = np.mean(W)
    n = W.shape[0]
    np.random.seed(1234)
    H = np.random.uniform(0,2*((m/k)**0.5), (n,k))
    return H

def main(*args):
    X = np.array([[0, 1],
              [0, 0],
              [0,1]])
    A = get_similarity_matrix(X)
    print(A)
    print("eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee")
    D = get_diagonal_degree_matrix(A)
    print(D)
    print("pppppppppppppppppppppppppppppppppppppppppppppppppppppppppp")
    W = get_normalized_similarity_matrix(A,D)
    print(W)
    print("pppppppppppppppppppppppppppppppppppppppppppppppppppppppppp")
    print(init_H(W, 2))

if __name__ == "__main__":
    main(*sys.argv[1:])
