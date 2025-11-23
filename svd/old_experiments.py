import os
import numpy as np
from scipy.linalg import svd, qr, interpolative
try: 
    from sklearn.utils.extmath import randomized_svd
    from sklearn.kernel_approximation import Nystroem
except ImportError:
    print("sklearn not installed. Installing using pip install scikit-learn")
    import subprocess
    subprocess.check_call(["pip", "install", "scikit-learn"])
    from sklearn.utils.extmath import randomized_svd
    from sklearn.kernel_approximation import Nystroem
from PIL import Image

try:
    import pymanopt
    from pymanopt.manifolds import Grassmann, FixedRankEmbedded
    from pymanopt.optimizers import ConjugateGradient
    import autograd.numpy as anp
except ImportError:
    print("pymanopt not installed. Installing using pip install pymanopt autograd")
    import subprocess
    subprocess.check_call(["pip", "install", "pymanopt"])
    subprocess.check_call(["pip", "install", "autograd"])
    import pymanopt
    from pymanopt.manifolds import Grassmann, FixedRankEmbedded
    from pymanopt.optimizers import ConjugateGradient
    import autograd.numpy as anp       

try:
    import torch
except:
    import subprocess
    subprocess.check_call(["pip", "install", "pytorch"])
    import torch
    
# Set seed for reproducibility
np.random.seed(42)

# ---------------------------
# 1. Synthetic Low-Rank Matrix
# ---------------------------
def generate_low_rank_matrix(m, n, r, noise_level=0.0):
    U = np.random.randn(m, r)
    V = np.random.randn(n, r)
    A = U @ V.T
    noise = noise_level * np.random.randn(m, n)
    return A + noise

# ---------------------------
# 2. Real Image as Matrix
# ---------------------------
def load_image_as_matrix(path, size=(512, 512)):
    img = Image.open(path).convert('L')  # Convert to grayscale
    img = img.resize(size)
    return np.array(img, dtype=np.float64)

# ---------------------------
# 3. Approximation Methods
# ---------------------------

def svd_approx(A, rank=None, max_iter=100, tol=1e-10):
    """
    Compute SVD decomposition from scratch using power iteration method.
    
    Parameters:
    - A: input matrix of shape (m, n)
    - rank: number of singular values/vectors to compute (default: min(m,n))
    - max_iter: maximum iterations for power method convergence
    - tol: tolerance for convergence
    
    Returns:
    - U: left singular vectors
    - S: singular values
    - VT: right singular vectors transposed
    """
    A = np.array(A, dtype=float)  # ensure A is a numpy array with float dtype
    m, n = A.shape
    
    if rank is None:
        rank = min(m, n)
    else:
        rank = min(rank, min(m, n))
    
    # Initialize matrices to store results
    U = np.zeros((m, rank))
    S = np.zeros(rank)
    VT = np.zeros((rank, n))
    
    # Make a copy of A to work with (we'll be deflating it)
    A_work = A.copy()
    
    for k in range(rank):
        # Initialize random vector for power iteration
        v = np.random.randn(n)
        v = v / np.linalg.norm(v)
        
        # Power iteration to find dominant right singular vector
        for _ in range(max_iter):
            # v ← A^T A v (power iteration for dominant eigenvector)
            v_new = A_work.T @ (A_work @ v)
            
            # Normalize
            v_new_norm = np.linalg.norm(v_new)
            
            # Handle the case of zero vector (can happen with deflation)
            if v_new_norm < tol:
                v_new = np.random.randn(n)
                v_new = v_new / np.linalg.norm(v_new)
            else:
                v_new = v_new / v_new_norm
            
            # Check convergence
            if np.linalg.norm(v_new - v) < tol:
                v = v_new
                break
            
            v = v_new
        
        # Compute singular value and left singular vector
        u = A_work @ v
        sigma = np.linalg.norm(u)
        
        # Handle near-zero singular values (numerical stability)
        if sigma > tol:
            u = u / sigma
        else:
            # If singular value is effectively zero, generate a random orthogonal vector
            u = np.random.randn(m)
            u = u / np.linalg.norm(u)
        
        # Store results
        U[:, k] = u
        S[k] = sigma
        VT[k, :] = v
        
        # Deflate the matrix: A_k+1 = A_k - sigma_k * u_k * v_k^T
        A_work = A_work - sigma * np.outer(u, v)
        
        # Reorthogonalize if numerical errors accumulate (optional)
        if k > 0 and k % 10 == 0:  # every 10 iterations
            # Reorthogonalize U
            for i in range(k):
                U[:, k] = U[:, k] - np.dot(U[:, k], U[:, i]) * U[:, i]
            U[:, k] = U[:, k] / np.linalg.norm(U[:, k])
            
            # Reorthogonalize VT
            for i in range(k):
                VT[k, :] = VT[k, :] - np.dot(VT[k, :], VT[i, :]) * VT[i, :]
            VT[k, :] = VT[k, :] / np.linalg.norm(VT[k, :])
    
    return U @ np.diag(S) @ VT

def scipy_svd_approx(A, rank):
    U, S, VT = svd(A, full_matrices=False)
    return U[:, :rank] @ np.diag(S[:rank]) @ VT[:rank, :]

# Manual implementation of randomized SVD (Algorithm 4.3 from Halko et al.)
def rsvd_approx(A, rank, oversample=10, n_power_iter=2):
    """
    Implementation of the randomized SVD algorithm.
    
    Parameters:
    - A: input matrix
    - rank: target rank
    - oversample: oversampling parameter (default: 10)
    - n_power_iter: number of power iterations (default: 2)
    """
    m, n = A.shape
    r = min(rank + oversample, min(m, n))
    
    # Step 1: Generate random Gaussian matrix
    Omega = np.random.randn(n, r)
    
    # Step 2: Compute sampling matrix Y = A*Omega
    Y = A @ Omega
    
    # Step 3: Optional power iterations to increase accuracy
    for _ in range(n_power_iter):
        Y = A @ (A.T @ Y)
    
    # Step 4: Compute orthogonal basis Q via QR decomposition
    Q, _ = np.linalg.qr(Y, mode='reduced')
    
    # Step 5: Compute small matrix B = Q^T * A 
    B = Q.T @ A
    
    # Step 6: SVD of small matrix B
    U_B, S, Vt = np.linalg.svd(B, full_matrices=False)
    
    # Step 7: Recover left singular vectors
    U = Q @ U_B
    
    # Return only the desired rank
    return U[:, :rank] @ np.diag(S[:rank]) @ Vt[:rank, :]

def sklearn_rsvd_approx(A, rank):    
    U, S, VT = randomized_svd(A, n_components=rank)
    return U @ np.diag(S) @ VT


def cur_approx(A, rank, oversample=2, leverage=True):
    """
    CUR with optional oversampling and leverage‐score sampling.
      rank       = target CUR rank
      oversample = pick c = min(n, oversample*rank) cols (and similarly rows)
      leverage   = if True, use top‐rank SVD to form sampling probs
                   else, use plain column/row norms.
    """
    m, n = A.shape
    c = min(n, oversample * rank)
    r = min(m, oversample * rank)

    # 1) compute sampling probabilities
    if leverage:
        # top‐rank SVD
        U_r, S_r, Vt_r = svd(A, full_matrices=False)
        U_r = U_r[:, :rank]
        Vt_r = Vt_r[:rank, :]
        p_cols = np.sum(Vt_r**2, axis=0) / rank
        p_rows = np.sum(U_r**2, axis=1) / rank
    else:
        col_norms = np.linalg.norm(A, axis=0)
        p_cols = col_norms / col_norms.sum()
        row_norms = np.linalg.norm(A, axis=1)
        p_rows = row_norms / row_norms.sum()

    # 2) sample & form C
    cols = np.random.choice(n, size=c, replace=False, p=p_cols)
    C = A[:, cols] #/ np.sqrt(c * p_cols[cols])[None, :]

    # 3) sample & form R
    rows = np.random.choice(m, size=r, replace=False, p=p_rows)
    R = A[rows, :] #/ np.sqrt(r * p_rows[rows])[:, None]

    # 4) intersection & weight
    W = A[np.ix_(rows, cols)]
    # re‐normalize W
    # W = W / (np.sqrt(r*p_rows[rows])[:,None] * np.sqrt(c*p_cols[cols])[None,:])
    U = np.linalg.pinv(W)

    # 5) reconstruct
    return C @ U @ R

def id_approx(A, rank):
    # Interpolative decomposition via pivoted QR + reconstruction
    # 1) pick the top‐rank pivot columns
    Q, R, P = qr(A, mode='economic', pivoting=True)
    cols = P[:rank]
    C = A[:, cols]
    # 2) solve for X in A ≈ C @ X
    X = np.linalg.pinv(C) @ A
    # 3) assemble the full‐size low‐rank approx
    return C @ X

# SciPy interpolative decomposition
def scipy_id_approx(A, rank):
    # Calculate ID decomposition
    idx, proj = interpolative.interp_decomp(A, rank)
    # Reconstruct approximation
    B = interpolative.reconstruct_matrix_from_id(A[:, idx[:rank]], idx, proj)
    return B

def nystrom_approx(A, rank, symmetrize='AAT'):
    """
    Approximates the *kernel* K via Nyström, where
    symmetrize = 'AAT'   → K = A @ A.T
                = 'sym'   → K = (A + A.T)/2
    Returns the low-rank approximation of K.
    """
    # 1) form symmetric K
    if symmetrize == 'AAT':
        K = A @ A.T
    elif symmetrize == 'sym':
        K = 0.5 * (A + A.T)
    else:
        raise ValueError("symmetrize must be 'AAT' or 'sym'")

    # 2) sample columns of K
    n = K.shape[1]
    cols = np.random.choice(n, size=rank, replace=False)

    # 3) extract C and W from K
    C = K[:, cols]
    W = K[np.ix_(cols, cols)]

    # 4) Nyström reconstruction of K
    W_pinv = np.linalg.pinv(W)
    K_approx = C @ W_pinv @ C.T

    return K_approx

def manifold_approx(
        A,
        rank,
        n_steps=200,
        lr=5e-3,
        K=5,
        tol=1e-7
    ):
    """
    Low-rank matrix approximation using Riemannian optimization on the manifold of fixed-rank matrices.
    Parameters:
    -----------
    A : array-like - Input matrix to approximate
    rank : int - Target rank for the approximation
    n_steps : int, optional - Number of optimization steps (default: 200)
    lr : float, optional - Learning rate for optimizer (default: 5e-3)
    K : int, optional - Frequency of retraction to the manifold (default: 5)
    tol : float, optional - Tolerance for early stopping (default: 1e-7)
    Returns: array-like - Low-rank approximation of A
    """
    # Input validation
    if rank <= 0 or rank > min(A.shape):
        raise ValueError(f"Rank must be between 1 and min(A.shape)={min(A.shape)}")
    m, n = A.shape
    r = rank 
    
    # Convert numpy array to torch tensor
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    A_t = torch.tensor(A, dtype=torch.float32, device=device)
    
    # Initialize factors using SVD for better starting point
    try:
        # Use randomized SVD for initialization if matrix is large
        if max(m, n) > 1000:
            U, S, VT = randomized_svd(A, n_components=r)
        else:
            U, S, VT = svd(A, full_matrices=False)
        
        # Take only the top r components
        U_r = U[:, :r]
        S_r = S[:r]
        VT_r = VT[:r, :]
        
        # Initialize B and C using SVD factors
        sqrt_S = np.sqrt(S_r)
        B_init = U_r * sqrt_S[None, :]
        C_init = VT_r.T * sqrt_S[None, :]
        
        B = torch.tensor(B_init, dtype=torch.float32, device=device, requires_grad=True)
        C = torch.tensor(C_init, dtype=torch.float32, device=device, requires_grad=True)
    except:
        # Fallback to random initialization if SVD fails
        B = torch.randn(m, r, device=device, requires_grad=True)
        C = torch.randn(n, r, device=device, requires_grad=True)
    
    # Use Adam optimizer instead of SGD for better convergence
    optimizer = torch.optim.Adam([B, C], lr=lr, weight_decay=1e-5)
    
    prev_loss = float('inf')
    patience = 0
    max_patience = 5  # Allow a few iterations without improvement
    
    for step in range(n_steps):
        optimizer.zero_grad()        
        # Current approximation
        A_hat = B @ C.T
        # Frobenius-norm loss
        loss = torch.norm(A_hat - A_t, p='fro')**2
        
        # Detect explosion early
        if not torch.isfinite(loss):
            print(f"Loss went non-finite at step {step}")
            break
            
        loss.backward()        
        torch.nn.utils.clip_grad_norm_([B, C], max_norm=10.0)
        optimizer.step()
        
        # Do Riemannian-style step (retracting to the manifold)
        if (step + 1) % K == 0:
            with torch.no_grad():
                # QR on B and C
                Qb, Rb = torch.linalg.qr(B, mode='reduced')
                Qc, Rc = torch.linalg.qr(C, mode='reduced')
                
                # Form small matrix S = Rb @ Rc^T
                S = Rb @ Rc.T
                
                # Check for non-finite values and replace them
                if not torch.all(torch.isfinite(S)):
                    S = torch.where(torch.isfinite(S), S, torch.zeros_like(S))
                
                # Add small regularization for numerical stability
                S = S + 1e-8 * torch.eye(r, device=device)
                
                # SVD of the small core
                U_s, S_s, Vh_s = torch.linalg.svd(S, full_matrices=False)
                sqrtS = torch.diag(torch.sqrt(S_s))
                B.copy_(Qb @ U_s @ sqrtS)
                C.copy_(Qc @ Vh_s.T @ sqrtS)
        
        # Early stopping with patience
        curr = loss.item()
        improvement = prev_loss - curr
        if improvement < tol:
            patience += 1
            if patience >= max_patience:
                break
        else:
            patience = 0
            
        prev_loss = curr
    
    # Return the approximation
    return (B @ C.T).detach().cpu().numpy()


def create_cost_and_derivates(manifold, matrix, backend):
    euclidean_gradient = None

    if backend == "autograd":

        @pymanopt.function.autograd(manifold)
        def cost(u, s, vt):
            X = u @ anp.diag(s) @ vt
            return anp.linalg.norm(X - matrix) ** 2
    else:
        raise ValueError(f"Unsupported backend '{backend}'")

    return cost, euclidean_gradient


def pymanopt_manifold_approx(A, rank):
    """
    Low-rank matrix approximation using pymanopt for manifold optimization.
    This uses the Grassmann manifold for finding the optimal subspaces.
    """            
    m, n = A.shape
    
    # Define the manifold: Grassmann manifold
    manifold = FixedRankEmbedded(m, n, rank)    
    # manifold = FixedRankEmbedded(m, n, rank)
        
    cost, euclidean_gradient = create_cost_and_derivates(
        manifold, A, 'autograd'
    ) 
    # Create problem
    problem = pymanopt.Problem(manifold=manifold, cost=cost, euclidean_gradient=euclidean_gradient)
    
    # Choose solver (ConjugateGradient is typically more effective than SteepestDescent)
    optimizer = ConjugateGradient(
        verbosity=0,
        beta_rule="PolakRibiere"  # This rule often performs well
    )
    
    # Solve the optimization problem
    u, s, vt = optimizer.run(problem).point
    
    # Return low-rank approximation
    return u @ anp.diag(s) @ vt

# ---------------------------
# 4. Error Metrics
# ---------------------------
def rel_fro_error(A, A_approx):
    return np.linalg.norm(A - A_approx, ord='fro') / np.linalg.norm(A, ord='fro')

# ---------------------------
# Run Experiments
# ---------------------------

if __name__ == '__main__':
    import time
    import matplotlib.pyplot as plt    
    import seaborn as sns
    import pandas as pd
    
    # 1) parameters
    synth_ranks = [5, 10, 20, 50]    
    img_ranks = [10, 20, 30]   # pick one rank for image illustrations
    
    methods = {
        'SVD': svd_approx,
        'Scipy-SVD': scipy_svd_approx,
        'rSVD': rsvd_approx,
        'Sklearn-rSVD': sklearn_rsvd_approx,
        'CUR': cur_approx,
        'ID': id_approx,
        'Scipy-ID': scipy_id_approx,
        'nystrom': nystrom_approx,
        'manifold': manifold_approx,
        'Pymanopt-manifold': pymanopt_manifold_approx,
    }
    # 2) prepare storage
    syn_errs_results = {name: [] for name in methods}
    syn_rn_results = {name: [] for name in methods}

    # 3) loop over ranks on synthetic data
    # 3a) loop over ranks on synthetic data
    for r in synth_ranks:
        A_syn = generate_low_rank_matrix(500, 500, r, noise_level=0.01)
        for name, method in methods.items():
            t0 = time.perf_counter()
            if 'nystrom' in name:
                # Nyström approximates K = A A^T
                K    = A_syn @ A_syn.T
                K_app = method(A_syn, r)
                err  = rel_fro_error(K, K_app)
            elif name == 'CUR':
                K_app = method(A_syn, r, oversample=2, leverage=True)
                err   = rel_fro_error(A_syn, K_app)
            else:
                A_app = method(A_syn, r)
                err   = rel_fro_error(A_syn, A_app)
            t1 = time.perf_counter()
            syn_errs_results[name].append(err)
            syn_rn_results[name].append(t1 - t0)

    # 3b) loop over matrix dimensions using fixed rank
    dimensions = [500, 1000, 3000, 5000]
    fixed_rank = 20  # Use a moderate rank for dimension scaling tests
    dim_errs_results = {name: [] for name in methods}
    dim_rn_results = {name: [] for name in methods}

    for dim in dimensions:
        print(f"Testing dimension {dim}x{dim} with rank {fixed_rank}...")
        A_syn = generate_low_rank_matrix(dim, dim, fixed_rank, noise_level=0.01)
        for name, method in methods.items():
            t0 = time.perf_counter()
            if 'nystrom' in name:
                # Nyström approximates K = A A^T
                K    = A_syn @ A_syn.T
                K_app = method(A_syn, fixed_rank)
                err  = rel_fro_error(K, K_app)
            elif name == 'CUR':
                K_app = method(A_syn, fixed_rank, oversample=2, leverage=True)
                err   = rel_fro_error(A_syn, K_app)
            else:
                A_app = method(A_syn, fixed_rank)
                err   = rel_fro_error(A_syn, A_app)
            t1 = time.perf_counter()
            dim_errs_results[name].append(err)
            dim_rn_results[name].append(t1 - t0)
            print(f"  {name}: Error={err:.4f}, Time={t1-t0:.3f}s")

    # 4) print summary tables
    # Print rank comparison table
    print("\n--- Rank Comparison ---")
    header = ['rank'] + list(methods.keys())
    print('\t'.join(header))
    for i, r in enumerate(synth_ranks):
        row = [str(r)]
        for name in methods:
            e = syn_errs_results[name][i]
            t = syn_rn_results[name][i]
            row.append(f"{e:.4f}/{t:.3f}s")
        print('\t'.join(row))
    
    # Print dimension comparison table
    print("\n--- Dimension Comparison ---")
    header = ['dimension'] + list(methods.keys())
    print('\t'.join(header))
    for i, dim in enumerate(dimensions):
        row = [f"{dim}x{dim}"]
        for name in methods:
            e = dim_errs_results[name][i]
            t = dim_rn_results[name][i]
            if e is not None and t is not None:
                row.append(f"{e:.4f}/{t:.3f}s")
            else:
                row.append("failed")
        print('\t'.join(row))
    
    
    # Create dataframes for dimension results
    df_err_dim = pd.DataFrame(dim_errs_results, index=dimensions)
    df_rt_dim = pd.DataFrame(dim_rn_results, index=dimensions)

    df_err_rank = pd.DataFrame(syn_errs_results, index=synth_ranks)
    df_rt_rank = pd.DataFrame(syn_rn_results, index=synth_ranks)

    # Heat-maps for dimension results
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    sns.heatmap(df_err_dim.T, annot=True, fmt=".4f", cmap="YlGnBu", cbar_kws={'label':'Error'})
    plt.title("Relative Error vs. Dimension")
    plt.xlabel("Dimension"); plt.ylabel("Method")

    plt.subplot(1,2,2)
    sns.heatmap(df_rt_dim.T, annot=True, fmt=".3f", cmap="YlOrRd", cbar_kws={'label':'Time (s)'})
    plt.title("Runtime vs. Dimension")
    plt.xlabel("Dimension"); plt.ylabel("")

    plt.tight_layout()
    plt.savefig("img/dimension_heatmaps.png", dpi=200)
    plt.close()

    # Heat-maps for dimenRanksion results
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    sns.heatmap(df_err_rank.T, annot=True, fmt=".4f", cmap="YlGnBu", cbar_kws={'label':'Error'})
    plt.title("Relative Error vs. Rank")
    plt.xlabel("Rank"); plt.ylabel("Method")

    plt.subplot(1,2,2)
    sns.heatmap(df_rt_rank.T, annot=True, fmt=".3f", cmap="YlOrRd", cbar_kws={'label':'Time (s)'})
    plt.title("Runtime vs. Rank")
    plt.xlabel("Rank"); plt.ylabel("")

    plt.tight_layout()
    plt.savefig("img/rank_heatmaps.png", dpi=200)
    plt.close()
    
    # Create visualizations to compare methods across ranks
    plt.figure(figsize=(12, 10))
    
    # 1. Error comparison by rank (line plot)
    plt.subplot(2, 2, 1)
    for name, result in syn_errs_results.items():
        plt.plot(synth_ranks, result, marker='o', label=name)
    plt.xlabel('Rank')
    plt.ylabel('Relative Frobenius Error')
    plt.title('Error vs Rank')
    plt.grid(True)
    plt.legend(loc='best', fontsize=8)
    
    # 2. Time comparison by rank (line plot)
    plt.subplot(2, 2, 2)
    for name, result in syn_rn_results.items():
        plt.plot(synth_ranks, result, marker='o', label=name)
    plt.xlabel('Rank')
    plt.ylabel('Time (s)')
    plt.title('Execution Time vs Rank')
    plt.grid(True)
    
    # 3. Error comparison by dimension (line plot)
    plt.subplot(2, 2, 3)
    for name, result in dim_errs_results.items():
        valid_points = [(d, e) for d, e in zip(dimensions, result) if e is not None]
        if valid_points:
            dims, errs = zip(*valid_points)
            plt.plot(dims, errs, marker='o', label=name)
    plt.xlabel('Matrix Dimension')
    plt.ylabel('Relative Frobenius Error')
    plt.title(f'Error vs Dimension (rank={fixed_rank})')
    plt.grid(True)    

    # 4. Time comparison by dimension (line plot)
    plt.subplot(2, 2, 4)
    for name, result in dim_rn_results.items():
        valid_points = [(d, t) for d, t in zip(dimensions, result) if t is not None]
        if valid_points:
            dims, times = zip(*valid_points)
            plt.plot(dims, times, marker='o', label=name)
    plt.xlabel('Matrix Dimension')
    plt.ylabel('Time (s)')
    plt.title(f'Execution Time vs Dimension (rank={fixed_rank})')
    plt.yscale('log')  # Log scale for better visibility
    plt.grid(True)    
        
    plt.tight_layout()
    plt.savefig('img/method_comparison.png', dpi=300, bbox_inches='tight')
    
    # Create a separate figure for dimension scaling
    plt.figure(figsize=(15, 6))
    
    # 1. Error comparison by dimension
    plt.subplot(1, 2, 1)
    for name, result in dim_errs_results.items():
        valid_points = [(d, e) for d, e in zip(dimensions, result) if e is not None]
        if valid_points:
            dims, errs = zip(*valid_points)
            plt.plot(dims, errs, marker='o', label=name)
    plt.xlabel('Matrix Dimension')
    plt.ylabel('Relative Frobenius Error')
    plt.title(f'Error vs Dimension (rank={fixed_rank})')
    plt.grid(True)
    plt.legend()
    
    # 2. Time comparison by dimension with log scale for time
    plt.subplot(1, 2, 2)
    for name, result in dim_rn_results.items():
        valid_points = [(d, t) for d, t in zip(dimensions, result) if t is not None]
        if valid_points:
            dims, times = zip(*valid_points)
            plt.plot(dims, times, marker='o', label=name)
    plt.xlabel('Matrix Dimension')
    plt.ylabel('Time (s)')
    plt.title(f'Execution Time vs Dimension (rank={fixed_rank})')
    plt.yscale('log')  # Log scale for better visibility
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('img/dimension_scaling.png', dpi=300, bbox_inches='tight')

    # 5) visualize image approximations at ranks [10,20,50]
    img_path = 'img/sample.jpg'
    if os.path.exists(img_path):
        A_img = load_image_as_matrix(img_path)
    else:
        A_img = generate_low_rank_matrix(512, 512,  max(img_ranks))

    # only methods that approximate A directly
    methods_img = {k:methods[k] for k in methods if k!='nystrom' and '-' not in k}

    nrows, ncols = len(img_ranks), len(methods_img) + 1
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(4*ncols, 3*nrows),
                             squeeze=False)

    for i, rk in enumerate(img_ranks):
        # original in first column
        ax = axes[i,0]
        ax.imshow(A_img, cmap='gray')
        ax.set_title(f'Original\n(>={rk})')
        ax.axis('off')

        # approximations
        for j, (name, method) in enumerate(methods_img.items(), start=1):
            if name == 'CUR':
                K_app = method(A_img, rk, oversample=2, leverage=True)
            else:
                A_ap = method(A_img, rk)
            ax = axes[i,j]
            ax.imshow(A_ap, cmap='gray')
            ax.set_title(f'{name}\nrank={rk}')
            ax.axis('off')

    plt.tight_layout()
    # Save the figure
    plt.savefig('img/low_rank_approximation_results.png', dpi=300, bbox_inches='tight')
    plt.show()  