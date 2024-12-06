import numpy as np
import torch
from torch.distributions import Binomial, Categorical, Normal

from mixmil.model import MixMIL

def simulate_data(likelihood="binomial", N=100, Q=5, K=3, P=1, n_trials=2, max_instances=100, seed=42):
    """
    Simulate data for testing MixMIL with three likelihoods:
    - binomial
    - categorical
    - normal

    Parameters:
    -----------
    likelihood : str
        One of {"binomial", "categorical", "normal"}.
    N : int
        Number of bags.
    Q : int
        Latent dimension for random effects.
    K : int
        Number of fixed effects.
    P : int
        Number of outputs (for categorical, number of classes).
    n_trials : int
        Number of trials for binomial likelihood (ignored for categorical/normal).
    max_instances : int
        Maximum number of instances per bag (instances per bag will be random up to this number).
    seed : int
        Random seed for reproducibility.

    Returns:
    --------
    Xs : list of np.ndarray
        A list of arrays, each (M_i x Q) representing the instances of bag i.
    F : torch.Tensor
        Fixed effects design matrix (N x K).
    Y : torch.Tensor
        Targets (N x P) for binomial/normal or (N x 1) for categorical if P>2,
        or (N,) for categorical if P=2 (we'll keep it as (N,1) for consistency).
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Fixed effects design matrix: N x K
    F = torch.tensor(np.random.randn(N, K), dtype=torch.float32)

    # True parameters
    alpha = np.random.randn(K, P) * 0.5   # fixed effects coefficients: (K, P)
    mu_z = np.random.randn(Q, P) * 0.5    # mean random effects loadings: (Q, P)
    # norm_mu_z for dimension-consistent scaling
    # We'll normalize mu_z to avoid large scale differences
    mu_z_t = torch.tensor(mu_z, dtype=torch.float32)
    norm_mu_z = mu_z_t / (mu_z_t.norm() + 1e-6)

    # Generate bags with variable number of instances
    Xs = []
    for i in range(N):
        M_i = np.random.randint(3, max_instances)  # between 3 and max_instances
        X_i = np.random.randn(M_i, Q)  # instances for bag i
        Xs.append(X_i)

    alpha_t = torch.tensor(alpha, dtype=torch.float32)

    Y_list = []
    for i, X_i in enumerate(Xs):
        X_t = torch.tensor(X_i, dtype=torch.float32)  # (M_i, Q)

        # Compute instance weights w_i:
        # w_logits: (M_i, P) = X_i @ mu_z_t
        w_logits = X_t @ mu_z_t
        w = torch.softmax(w_logits, dim=0)  # (M_i, P)

        # Compute Xi_mu = X_i @ norm_mu_z: (M_i, P)
        Xi_mu = X_t @ norm_mu_z

        # u_i = sum over instances of w * Xi_mu: (P,)
        u_i = (w * Xi_mu).sum(dim=0)  # (P,)

        # Compute logits: (1, P) = F_i alpha + u_i
        Fi = F[i:i+1, :]  # (1, K)
        logits_i = Fi @ alpha_t + u_i  # (1, P)

        if likelihood == "binomial":
            # Y_i ~ Binomial(n_trials, p), p = sigmoid(logits_i)
            dist = Binomial(total_count=n_trials, logits=logits_i)
            Yi = dist.sample()  # (1, P)

        elif likelihood == "categorical":
            # For categorical, if P=1, we need to make it a 2-class problem:
            if P == 1:
                logits_cat = torch.cat([-logits_i, logits_i], dim=1)  # (1, 2)
                dist = Categorical(logits=logits_cat)
                Yi = dist.sample()  # shape (1,)
                Yi = Yi.unsqueeze(-1)  # (1,1)
            else:
                # P classes directly
                dist = Categorical(logits=logits_i)
                Yi = dist.sample().unsqueeze(-1)  # (1,1) for consistency

        elif likelihood == "normal":
            # Y_i ~ Normal(logits_i, scale=1.0)
            dist = Normal(loc=logits_i, scale=1.0)
            Yi = dist.sample()  # (1, P)
        else:
            raise ValueError("Unknown likelihood.")

        Y_list.append(Yi)

    Y = torch.cat(Y_list, dim=0)  # (N, P) or (N,1) depending on P

    return Xs, F, Y


# Example usage:
if __name__ == "__main__":
    # Test binomial
    Xs_bin, F_bin, Y_bin = simulate_data(likelihood="binomial", N=10, Q=3, K=2, P=1, n_trials=2)
    print("Binomial Y:", Y_bin)

    # Test categorical (P=2 classes)
    Xs_cat, F_cat, Y_cat = simulate_data(likelihood="categorical", N=10, Q=3, K=2, P=2)
    print("Categorical Y:", Y_cat)

    # Test normal
    Xs, F, Y = simulate_data(likelihood="normal", N=10, Q=3, K=2, P=1)
    print("Normal Y:", Y)
    Xs = [torch.tensor(x, dtype=torch.float32) for x in Xs]  # list of tensors
    if not isinstance(F, torch.Tensor):
        F = torch.tensor(F, dtype=torch.float32)
    if not isinstance(Y, torch.Tensor):
        Y = torch.tensor(Y, dtype=torch.float32)

    model = MixMIL.init_with_mean_model(Xs, F, Y, likelihood='normal')

    history = model.train(Xs, F, Y, n_epochs=500, batch_size=8, lr=1e-3, verbose=True)


