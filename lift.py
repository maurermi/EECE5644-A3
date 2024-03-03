import numpy as np

def lift(x):
    i_dim = len(x)

    # As defined in assignment, the lifting function is of the form
    # R^d -> R^[d*(d-1)/2 + 2d]
    # NOTE: This means we are not including the affine bias term
    # currently
    # res_dim = int(i_dim*(i_dim-1)/2 + i_dim)

    x_prime = x
    for i in range(i_dim):
        for j in range(i, i_dim):
            x_prime = np.append(x_prime, x[i]*x[j])
    return x_prime

def liftDataset(X):
    # Lift the entire dataset
    # X is of shape (N, d)
    # X' is of shape (N, d*(d-1)/2 + 2d)
    X_prime = np.array([lift(x) for x in X])
    return X_prime

