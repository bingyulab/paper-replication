import os
import numpy as np
from scipy.linalg import svd
import time
import matplotlib.pyplot as plt
import pandas as pd
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from svd.svd_njit import golub_kahan_svd_numba

# Set seed for reproducibility
np.random.seed(42)

# Create output directory
os.makedirs('imgs/svd', exist_ok=True)

# Science plots styling
def set_science_style():
    import scienceplots
    plt.style.use(['science', 'ieee'])
    # Ensure we do not call external LaTeX even if the style sets it
    plt.rcParams['text.usetex'] = False

set_science_style()
# ---------------------------
# CORE ALGORITHMS
# ---------------------------

def apply_householder_sequence(vectors, matrix, side='left'):
    """
    Applies a sequence of Householder transformations to a matrix.
    
    Args:
        vectors (list): A list of (start_index, householder_vector) tuples.
        matrix (np.ndarray): The matrix to be transformed.
    """
    # Note: Householder transformations must be applied from right to left (last to first)
    for k_start, v in reversed(vectors):
        if side == 'left':
            # Left multiplication: H*M = (I - 2vv^T)M = M - 2v(v^T M)
            sub_matrix = matrix[k_start:, :]
            # Ensure sub_matrix is not empty before applying
            if sub_matrix.shape[0] > 0:
                matrix[k_start:, :] -= 2 * np.outer(v, v @ sub_matrix)
        elif side == 'right':
            # Right multiplication: M*H = M(I - 2vv^T) = M - 2(Mv)v^T
            sub_matrix = matrix[:, k_start:]
            # Ensure sub_matrix is not empty before applying
            if sub_matrix.shape[1] > 0:
                 matrix[:, k_start:] -= 2 * np.outer(sub_matrix @ v, v)
    return matrix

def householder_bidiagonalization(A):
    A_work = A.copy()
    m, n = A.shape    
    
    # Store Householder vectors for U and V
    p_vectors = []
    q_vectors = []
    
    # Step 1: Bidiagonalization (A = P * J * Q.T)
    for k in range(n):
        # Column reflection (forms P)
        x_col = A_work[k:, k]
        x_norm = np.linalg.norm(x_col)
        # Skip if already zero
        if x_norm > 1e-14:
            alpha = -np.sign(x_col[0]) * x_norm if x_col[0] != 0 else -x_norm
            v = x_col.copy()
            v[0] -= alpha
            v_norm = np.linalg.norm(v)
            if v_norm > 1e-14:
                v = v / v_norm
                p_vectors.append((k, v))
                
                # Apply to remaining columns
                sub = A_work[k:, k:]
                proj = v @ sub
                sub -= 2 * np.outer(v, proj)

        # Row reflection (forms Q)
        if k < n - 1:
            x_row = A_work[k, k+1:]
            norm_x = np.linalg.norm(x_row)
            if norm_x > 1e-14:
                beta = -np.sign(x_row[0]) * norm_x if x_row[0] != 0 else -norm_x
                v = x_row.copy()
                v[0] -= beta
                v_norm = np.linalg.norm(v)
                if v_norm > 1e-14:
                    v = v / v_norm
                    q_vectors.append((k+1, v))
                    
                    # Apply to remaining rows
                    sub = A_work[k:, k+1:]
                    proj = np.dot(sub, v)
                    sub -= 2 * np.outer(proj, v)
    # Extract bidiagonal matrix J
    J = np.diag(np.diag(A_work, k = 0)) + np.diag(np.diag(A_work, k=1), k=1)
    
    return J, p_vectors, q_vectors
    
def golub_kahan_svd(A):
    """
    Performs SVD using the Golub-Kahan bidiagonalization algorithm.
    This implementation follows the paper's methodology.
    $O(mn^2)$
    Returns:
        U (np.ndarray): Left singular vectors.
        S (np.ndarray): Singular values.
        VT (np.ndarray): Transposed right singular vectors.
    """    
    m, n = A.shape 
    
    # Handle m < n case by transposing
    if m < n:
        # A = U S Vt  =>  A.T = V S Ut
        # We compute SVD of A.T, then swap U and V and transpose them
        V, S, Ut = golub_kahan_svd_numba(A.T, full_matrices=full_matrices)
        return Ut.T, S, V.T
    
    J, p_vectors, q_vectors = householder_bidiagonalization(A)

    # Step 2: Compute SVD of the bidiagonal matrix J
    U_J, S, VT_J = svd(J, full_matrices=False)

    # Step 3: Reconstruct full U and V by applying the stored Householder transformations
    U = apply_householder_sequence(p_vectors, U_J, 'left')
    VT = apply_householder_sequence(q_vectors, VT_J, 'right')

    return U, S, VT


def compute_sv_from_bidiagonal(J, method='JTJ'):
    """
    Computing singular values from bidiagonal matrix
    method='JTJ': Uses eigenvalues of J^T @ J (fast but can lose accuracy for small singular values).
    method='2n': Uses eigenvalues of a related 2n x 2n symmetric tridiagonal matrix (more accurate).
    """
    n = J.shape[0]
    
    if method == 'JTJ':
        # Construct J^T J (tridiagonal symmetric)
        JTJ = J.T @ J
        eigvals = np.linalg.eigvalsh(JTJ)
        singular_values = np.sqrt(np.maximum(eigvals, 0))
    elif method == '2n':
        # Construct 2n x 2n matrix (equation 3.2)
        T = np.zeros((2 * n, 2 * n))
        alpha = np.diag(J)
        beta = np.diag(J, k=1)
        
        T[np.arange(0, 2*n-1, 2), np.arange(1, 2*n, 2)] = alpha
        T[np.arange(1, 2*n, 2), np.arange(0, 2*n-1, 2)] = alpha
        if n > 1:
            T[np.arange(1, 2*n-2, 2), np.arange(2, 2*n-1, 2)] = beta
            T[np.arange(2, 2*n-1, 2), np.arange(1, 2*n-2, 2)] = beta
            
        eigvals = np.linalg.eigvalsh(T)
        singular_values = np.abs(eigvals[n:]) # The positive eigenvalues are the singular values
    else:
        raise ValueError("method must be 'JTJ' or '2n'")
    
    return np.sort(singular_values)[::-1]

def randomized_svd(A, rank, oversample=10, n_power_iter=2):
    """
    Performs Randomized SVD. A modern, efficient alternative for low-rank approximation.
     
    Parameters:
    - A: input matrix
    - rank: target rank
    - oversample: oversampling parameter (default: 10)
    - n_power_iter: number of power iterations (default: 2)
    """
    m, n = A.shape
    r = min(rank + oversample, min(m, n))
    
    # 1. Random projection
    Omega = np.random.randn(n, r)
    Y = A @ Omega
    
    # 2. Power iterations for improved accuracy
    for _ in range(n_power_iter):
        Y = A @ (A.T @ Y)
    
    # 3. Form an orthonormal basis Q
    Q, _ = np.linalg.qr(Y, mode='reduced')
    
    # 4. Project A onto the smaller subspace
    B = Q.T @ A
    
    # 5. SVD of the small matrix B
    U_B, S, VT = svd(B, full_matrices=False)
    
    # 6. Reconstruct the full left singular vectors
    U = Q @ U_B

    return U[:, :rank], S[:rank], VT[:rank, :]

def pseudoinverse_svd(A, rcond=1e-15):
    """
    Computes the pseudo-inverse via SVD: A+ = V @ Σ+ @ U.T
    This is the most numerically stable method.
    """
    U, S, VT = svd(A, full_matrices=False)
    
    # Invert non-zero singular values, cutting off at a relative condition number
    cutoff = rcond * np.max(S)
    S_inv = np.where(S > cutoff, 1.0 / S, 0)
    
    return VT.T @ np.diag(S_inv) @ U.T

def pseudoinverse_normal_eq(A):
    """
    Computes the pseudo-inverse via normal equations: (A.T @ A)^(-1) @ A.T
    Known to fail for ill-conditioned or rank-deficient matrices.
    """
    ATA = A.T @ A
    # This will raise a LinAlgError if ATA is singular
    return np.linalg.inv(ATA) @ A.T

# ---------------------------
# TEST MATRIX GENERATORS
# ---------------------------

def generate_low_rank_matrix(m, n, rank):
    """Generates a matrix of size m x n with rank exactly rank."""
    U = np.random.randn(m, rank)
    V = np.random.randn(n, rank)
    return U @ V.T

def generate_controlled_sv_matrix(m, n, sv_type='geometric', condition=1e6):
    """
    Generate matrix with controlled singular value distribution
    """
    k = min(m, n)
    
    if sv_type == 'geometric':
        # σ_i = κ^(-i/k) for condition number κ
        sv = np.array([condition ** (-i/k) for i in range(k)])
    elif sv_type == 'clustered':
        # Two clusters of singular values
        sv = np.concatenate([np.ones(k//2), 0.01 * np.ones(k - k//2)])
    elif sv_type == 'gap':
        # Large gap after first few
        sv = np.concatenate([np.ones(3), 1e-8 * np.ones(k - 3)])
    elif sv_type == 'uniform':
        sv = np.linspace(1, 1/condition, k)
    else:
        raise ValueError(f"Unknown sv_type: {sv_type}")
    
    # Construct matrix with these singular values
    U = np.linalg.qr(np.random.randn(m, k))[0]
    V = np.linalg.qr(np.random.randn(n, k))[0]
    return U @ np.diag(sv) @ V.T

# ---------------------------
# ERROR METRICS
# ---------------------------

def relative_error(A, A_approx):
    """||A - A_approx||_F / ||A||_F"""
    return np.linalg.norm(A - A_approx, 'fro') / np.linalg.norm(A, 'fro')

def orthogonality_error(U):
    """||U^T U - I||_F"""
    n = U.shape[1]
    return np.linalg.norm(U.T @ U - np.eye(n), 'fro')

def reconstruction_error(A, U, S, VT):
    """Calculates ||A - U Σ V^T||_F / ||A||_F"""
    A_recon = U @ np.diag(S) @ VT
    return relative_error(A, A_recon)

# ---------------------------
# EXPERIMENT 1: BIDIAGONALIZATION
# ---------------------------

def experiment_1_golub_kahan_validation():
    """
    GOAL: Validate our implementation of the paper's Golub-Kahan SVD algorithm.
    METHOD: Compare its speed and accuracy against SciPy's battle-tested SVD.
    METRICS:
    - Runtime: How does our implementation's speed scale?
    - Accuracy: Does it reconstruct the original matrix correctly?
    - Orthogonality: Are the computed singular vectors orthogonal?
    """
    print("\n" + "="*60)
    print("EXPERIMENT 1: GOLUB-KAHAN SVD VALIDATION")
    print("="*60)

    sizes = [50, 100, 200, 500, 1000, 2000]
    results = []
    
    # Add a "warm-up" call to compile the Numba functions
    print("Warming up Numba JIT compiler (this may take a second)...")
    A_warmup = np.random.randn(10, 5)
    golub_kahan_svd_numba(A_warmup)
    print("Warm-up complete. Starting benchmark.")
    print("=" * 90)
    
    for n in sizes:
        A = generate_controlled_sv_matrix(n, n, condition=1e4)
        
        # Method 1: Our Golub-Kahan implementation
        t0 = time.perf_counter()
        U_gk, S_gk, VT_gk = golub_kahan_svd(A)
        t_gk = time.perf_counter() - t0
        
        # Method 2: SciPy SVD (baseline)
        t0 = time.perf_counter()
        U_sp, S_sp, VT_sp = svd(A, full_matrices=False, lapack_driver='gesdd')
        t_sp = time.perf_counter() - t0
        
        # Method 2: Golub-Kahan implementation (njit)
        t0 = time.perf_counter()
        U_nb, S_nb, VT_nb = golub_kahan_svd_numba(A, full_matrices=False)
        t_nb = time.perf_counter() - t0
        
        recon_error = reconstruction_error(A, U_gk, S_gk, VT_gk)
        recon_error_sci = reconstruction_error(A, U_sp, S_sp, VT_sp)
        recon_error_nb = reconstruction_error(A, U_nb, S_nb, VT_nb)
        sv_error = np.linalg.norm(S_gk - S_sp) / np.linalg.norm(S_sp)
        sv_error_nb = np.linalg.norm(S_gk - S_nb) / np.linalg.norm(S_nb)
        u_ortho_custom_error = orthogonality_error(U_gk)
        v_ortho_custom_error = orthogonality_error(VT_gk.T)
        u_ortho_scipy_error = orthogonality_error(U_sp)
        v_ortho_scipy_error = orthogonality_error(VT_sp.T)        
        u_ortho_scipy_error = orthogonality_error(U_sp)
        v_ortho_scipy_error = orthogonality_error(VT_sp.T)
        u_ortho_njit_error = orthogonality_error(U_nb)
        v_ortho_njit_error = orthogonality_error(VT_nb.T)
        results.append({
            'size': n,
            'Time (Golub-Kahan)': t_gk,
            'Time (SciPy)': t_sp,
            'Time (Numba)': t_nb,
            'Reconstruction Error (Golub-Kahan)': recon_error,
            'Reconstruction Error (SciPy)': recon_error_sci,
            'Reconstruction Error (Numba)': recon_error_nb,
            'Singular Value Error': sv_error,
            'Singular Value Error (Numba)': sv_error_nb,
            'Custom U Orthogonality Error': u_ortho_custom_error,
            'Custom V Orthogonality Error': v_ortho_custom_error,
            'Scipy U Orthogonality Error': u_ortho_scipy_error,
            'Scipy V Orthogonality Error': v_ortho_scipy_error,
            'Numba U Orthogonality Error': u_ortho_njit_error,
            'Numba V Orthogonality Error': v_ortho_njit_error
        })
        print(f"Size {n}x{n}: GK Time={t_gk:.4f}s, SciPy Time={t_sp:.4f}s, GK Numba Time={t_nb:.4f}s, "
              f"Reconstruction Error (Golub-Kahan)={recon_error:.2e}, ,Reconstruction Error (SciPy)={recon_error_sci:.2e}, "
              f"Reconstruction Error (Numba)={recon_error_nb:.2e}, "
              f"SV Error={sv_error:.2e}, SV Error (Numba)={sv_error_nb:.2e}, "
              f"U Orthogonality={u_ortho_custom_error:.2e}, V Orthogonality={v_ortho_custom_error:.2e}, "
              f"U Orthogonality (SciPy)={u_ortho_scipy_error:.2e}, V Orthogonality (SciPy)={v_ortho_scipy_error:.2e}, "
              f"U Orthogonality (Numba)={u_ortho_njit_error:.2e}, V Orthogonality (Numba)={v_ortho_njit_error:.2e}")

    df = pd.DataFrame(results)
    
    # Plot results
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    
    # Time complexity
    axes[0].plot(df['size'], df['Time (Golub-Kahan)'], 'o-', label='Golub-Kahan (Ours)', markersize=6)
    axes[0].plot(df['size'], df['Time (SciPy)'], 's-', label='SciPy (LAPACK)', markersize=6)
    axes[0].plot(df['size'], df['Time (Numba)'], '^-', label='Golub-Kahan (Numba)', markersize=6)
    axes[0].set_xlabel('Matrix Size (n x n)')
    axes[0].set_ylabel('Time (s)')
    axes[0].set_title('Computation Time')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy metrics
    axes[1].semilogy(df['size'], df['Reconstruction Error (Golub-Kahan)'], 'o-', label='Reconstruction (Golub-Kahan)', markersize=6)
    axes[1].semilogy(df['size'], df['Reconstruction Error (SciPy)'], 's-', label='Reconstruction (SciPy)', markersize=6)
    axes[1].semilogy(df['size'], df['Reconstruction Error (Numba)'], '^-', label='Reconstruction (Numba)', markersize=6)
    axes[1].semilogy(df['size'], df['Singular Value Error'], 'd-', label='Singular Values', markersize=6)
    axes[1].set_xlabel('Matrix Size (n x n)')
    axes[1].set_ylabel('Relative Error')
    axes[1].set_title('Accuracy vs. Baseline')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, which='both')
    
    # Orthogonality of computed vectors
    axes[2].semilogy(df['size'], df['Custom U Orthogonality Error'], 'o-', label='$||U_{gk}^T U_{gk} - I||_F$', markersize=6)
    axes[2].semilogy(df['size'], df['Custom V Orthogonality Error'], 's-', label='$||V_{gk}^T V_{gk} - I||_F$', markersize=6)
    axes[2].semilogy(df['size'], df['Scipy U Orthogonality Error'], 'd-', label='$||U_{svd}^T U_{svd} - I||_F$', markersize=6)
    axes[2].semilogy(df['size'], df['Scipy V Orthogonality Error'], 'p-', label='$||V_{svd}^T V_{svd} - I||_F$', markersize=6)
    axes[2].semilogy(df['size'], df['Numba U Orthogonality Error'], 'd-', label='$||U_{njit}^T U_{njit} - I||_F$', markersize=6)
    axes[2].semilogy(df['size'], df['Numba V Orthogonality Error'], 'p-', label='$||V_{njit}^T V_{njit} - I||_F$', markersize=6)
    axes[2].axhline(y=1e-14, color='r', linestyle='--', label='Machine $\\epsilon$', alpha=0.6)
    axes[2].set_xlabel('Matrix Size (n x n)')
    axes[2].set_ylabel('Orthogonality Error')
    axes[2].set_title('Numerical Stability')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3, which='both')
    
    plt.tight_layout()
    plt.savefig('imgs/svd/exp1_validation.pdf', dpi=300, bbox_inches='tight')
    print("\n[Saved] imgs/svd/exp1_validation.pdf")
    plt.close()

# ---------------------------
# EXPERIMENT 2: ACCURACY OF SINGULAR VALUE COMPUTATION
# ---------------------------

def experiment_2_sv_computation_accuracy():
    """
    GOAL: Test the paper's claim on singular value computation accuracy.
    METHOD: Compare the accuracy of the J^T J method vs. the more stable 2n x 2n method
            for matrices with high condition numbers.
    HYPOTHESIS: The J^T J method will lose accuracy for the smallest singular values
                because squaring the matrix squares its condition number.
    """
    print("\n" + "="*60)
    print("EXPERIMENT 2: SINGULAR VALUE COMPUTATION METHODS")
    print("="*60)

    conditions = np.logspace(2, 12, 6)
    n = 100
    results = []
    
    for cond in conditions:
        A = generate_controlled_sv_matrix(n, n, condition=cond)
        J_bidiag, _, _ = householder_bidiagonalization(A)
                
        # Method 1: J^T J (Potentially unstable)
        sv_jtj = compute_sv_from_bidiagonal(J_bidiag, method='JTJ')

        # Method 2: 2n x 2n (Numerically preferred)
        sv_2n = compute_sv_from_bidiagonal(J_bidiag, method='2n')

        # Baseline: True singular values
        sv_true = np.linalg.svd(A, compute_uv=False)
        err_jtj = np.linalg.norm(sv_jtj - sv_true) / np.linalg.norm(sv_true)
        err_2n = np.linalg.norm(sv_2n - sv_true) / np.linalg.norm(sv_true)
        smallest_sv_error_jtj = abs(sv_jtj[-1] - sv_true[-1]) / sv_true[-1]
        smallest_sv_error_2n = abs(sv_2n[-1] - sv_true[-1]) / sv_true[-1]
        results.append({
            'condition': cond,
            'Error (J^T J)': err_jtj,
            'Error (2n x 2n)': err_2n,
            'Smallest SV Error (J^T J)': smallest_sv_error_jtj,
            'Smallest SV Error (2n x 2n)': smallest_sv_error_2n
        })
        print(f"Condition {cond:.1e}: error (J^T J) = {err_jtj:.2e}, error (2n x 2n) = {err_2n:.2e}, "
              f"Smallest SV Error (J^T J) = {smallest_sv_error_jtj:.2e}, "
              f"Smallest SV Error (2n x 2n) = {smallest_sv_error_2n:.2e}")

    df = pd.DataFrame(results)

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))

    # Overall error
    axes[0].loglog(df['condition'], df['Error (J^T J)'], 'o-', label='$J^T J$ Method', markersize=6)
    axes[0].loglog(df['condition'], df['Error (2n x 2n)'], 's-', label='$2n \\times 2n$ Method', markersize=6)
    axes[0].set_xlabel('Condition Number $\\kappa(A)$')
    axes[0].set_ylabel('Relative Error in $\\sigma$')
    axes[0].set_title('Overall Singular Value Error')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, which='both')
    
    # Error in smallest singular value (the key test)
    axes[1].loglog(df['condition'], df['Smallest SV Error (J^T J)'], 'o-', label='$J^T J$ Method', markersize=6)
    axes[1].loglog(df['condition'], df['Smallest SV Error (2n x 2n)'], 's-', label='$2n \\times 2n$ Method', markersize=6)
    axes[1].set_xlabel('Condition Number $\\kappa(A)$')
    axes[1].set_ylabel('Relative Error in $\\sigma_{min}$')
    axes[1].set_title('Smallest Singular Value Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, which='both')

    plt.tight_layout()
    plt.savefig('imgs/svd/exp2_sv_accuracy.pdf', dpi=300, bbox_inches='tight')
    print("\n[Saved] imgs/svd/exp2_sv_accuracy.pdf")
    plt.close()

# ---------------------------
# EXPERIMENT 3: PSEUDO-INVERSE ROBUSTNESS
# ---------------------------

def experiment_3_pseudoinverse_robustness():
    """
    GOAL: Demonstrate the superior numerical stability of the SVD-based pseudo-inverse.
    METHOD: Compare the SVD method against the normal equations method for solving
            least-squares problems with ill-conditioned and rank-deficient matrices.
    HYPOTHESIS: The normal equations method will fail or give poor results for
                problematic matrices, while the SVD method will remain robust.
    """
    print("\n" + "="*60)
    print("EXPERIMENT 3: PSEUDO-INVERSE ROBUSTNESS")
    print("="*60)
    
    # Test on ill-conditioned and rank-deficient matrices
    test_cases = {
        'Well-Conditioned': generate_controlled_sv_matrix(50, 30, condition=100),
        'Ill-Conditioned': generate_controlled_sv_matrix(50, 30, condition=1e8),
        'Rank-Deficient': generate_low_rank_matrix(50, 30, rank=15) # Effectively rank-deficient
    }
    
    results = []
    
    for name, A in test_cases.items():
        m, n = A.shape
        b = np.random.randn(m) # A random system to solve
        
        # Method 1: SVD-based pseudo-inverse (Stable)
        A_pinv_svd = pseudoinverse_svd(A)
        x_svd = A_pinv_svd @ b
        
        # Method 2: Normal Equations (Unstable)
        try:
            A_pinv_normal = pseudoinverse_normal_eq(A)
            x_normal = A_pinv_normal @ b
            normal_failed = False
        except np.linalg.LinAlgError:
            x_normal = np.full(n, np.nan)
            normal_failed = True
            
        res_svd = np.linalg.norm(A @ x_svd - b)
        sol_norm_svd = np.linalg.norm(x_svd)
        
        results.append({
            'case': name, 'method': 'SVD',
            'residual': res_svd,
            'solution_norm': sol_norm_svd,
            'failed': False
        })
        res_normal = np.linalg.norm(A @ x_normal - b) if not normal_failed else np.inf
        sol_norm_normal = np.linalg.norm(x_normal) if not normal_failed else np.inf
        results.append({
            'case': name, 'method': 'Normal Eq.',
            'residual': res_normal,
            'solution_norm': sol_norm_normal,
            'failed': normal_failed
        })
        print(f"Case '{name}': Normal equations {'FAILED' if normal_failed else 'succeeded'}."
              f" Residuals - SVD: {res_svd:.2e}, Normal Eq.: {res_normal:.2e}."
              f" Solution Norms - SVD: {sol_norm_svd:.2e}, Normal Eq.: {sol_norm_normal:.2e}")
    
    df = pd.DataFrame(results)
    
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    df_pivot = df.pivot(index='case', columns='method', values=['residual', 'solution_norm'])
    cases = df['case'].unique()
    x = np.arange(len(cases))
    width = 0.35
    
    # Residual ||Ax - b||
    df_pivot['residual'].plot(kind='bar', ax=axes[0], width=0.7, position=0.5, rot=0)
    axes[0].set_ylabel('Residual $||Ax - b||_2$')
    axes[0].set_title('Least Squares Residual')
    axes[0].set_yscale('log')
    axes[0].grid(True, alpha=0.3)
    
    # Solution norm ||x|| (SVD finds the minimum norm solution)
    df_pivot['solution_norm'].plot(kind='bar', ax=axes[1], width=0.7, position=0.5, rot=0)
    axes[1].set_ylabel('Solution Norm $||x||_2$')
    axes[1].set_title('Minimum Norm Property')
    # Use a log scale so a single large value doesn't dominate the plot
    axes[1].set_yscale('log')
    axes[1].grid(True, alpha=0.3, which='both')

    plt.tight_layout()
    plt.savefig('imgs/svd/exp3_pseudoinverse.pdf', dpi=300, bbox_inches='tight')
    print("\n[Saved] imgs/svd/exp3_pseudoinverse.pdf")
    plt.close()

# ---------------------------
# EXPERIMENT 4: RANDOMIZED SVD (MODERN BENCHMARK)
# ---------------------------

def experiment_4_randomized_svd_analysis():
    """
    GOAL: Analyze a modern alternative, Randomized SVD, which is not in the original paper
          but is highly relevant for large-scale problems.
    METHOD: Investigate how its accuracy is affected by key hyperparameters.
    - Oversampling (p): How many extra random vectors are needed?
    - Power Iterations (q): How to improve accuracy for ill-conditioned matrices?
    """
    print("\n" + "="*60)
    print("EXPERIMENT 4: RANDOMIZED SVD PARAMETER ANALYSIS")
    print("="*60)
    
    n, rank = 500, 20
    A_fast_decay = generate_controlled_sv_matrix(n, n, condition=1e8) # Fast SV decay
    A_slow_decay = generate_controlled_sv_matrix(n, n, sv_type='uniform', condition=1e2) # Slow SV decay

    oversample_params = [0, 5, 10, 20, 40]
    power_iter_params = [0, 1, 2, 3]
    results = []

    for A, decay_type in [(A_fast_decay, 'Fast Decay'), (A_slow_decay, 'Slow Decay')]:
        U_true, S_true, VT_true = svd(A, full_matrices=False)
        A_best_k = U_true[:, :rank] @ np.diag(S_true[:rank]) @ VT_true[:rank, :]
        
        for p in oversample_params:
            for q in power_iter_params:
                # Average over a few trials to smooth out randomness
                errors = []
                for _ in range(5):
                    U_r, S_r, VT_r = randomized_svd(A, rank, oversample=p, n_power_iter=q)
                    A_approx = U_r @ np.diag(S_r) @ VT_r
                    errors.append(np.linalg.norm(A_best_k - A_approx, 'fro') / np.linalg.norm(A_best_k, 'fro'))
                err = np.mean(errors)
                results.append({
                    'decay': decay_type,
                    'oversample': p,
                    'power_iter': q,
                    'error': err
                })
                print(f"Completed analysis for matrix with {decay_type.lower()}."
                      f" Results: {err:.2e} error for p={p}, q={q}")

    df = pd.DataFrame(results)

    # Plotting
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5), sharey=True)
    
    for ax, decay_type in zip(axes, ['Fast Decay', 'Slow Decay']):
        df_sub = df[df['decay'] == decay_type]
        for q in power_iter_params:
            df_plot = df_sub[df_sub['power_iter'] == q]
            ax.semilogy(df_plot['oversample'], df_plot['error'], 'o-', label=f'$q={q}$', markersize=5)
        
        ax.set_xlabel('Oversampling Parameter ($p$)')
        ax.set_title(f'Matrix with {decay_type}')
        ax.grid(True, which='both', alpha=0.3)
        ax.legend(title='Power Iterations')
    
    axes[0].set_ylabel('Error vs. Best Rank-k Approx.')
    
    plt.tight_layout()
    plt.savefig('imgs/svd/exp4_randomized_svd.pdf', dpi=300, bbox_inches='tight')
    print("\n[Saved] imgs/svd/exp4_randomized_svd.pdf")
    plt.close()


if __name__ == '__main__':
    experiment_1_golub_kahan_validation()
    experiment_2_sv_computation_accuracy()
    experiment_3_pseudoinverse_robustness()
    experiment_4_randomized_svd_analysis()