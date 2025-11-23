import numpy as np
from scipy.linalg import svd
from numba import njit
import time

# ============================================================================
# NUMBA-ACCELERATED VERSION
# ============================================================================

@njit(fastmath=True, cache=True)
def householder_bidiag_numba(A):
    """
    Pure Numba implementation of bidiagonalization.
    """
    A_work = A.copy()
    m, n = A_work.shape
    
    d = np.zeros(n, dtype=np.float64)
    e = np.zeros(n-1, dtype=np.float64)
    
    for k in range(n):
        # Left Householder (Column)
        norm_x = 0.0
        for i in range(k, m):
            norm_x += A_work[i, k] ** 2
        norm_x = np.sqrt(norm_x)
        
        if norm_x > 1e-14:
            # Determine sign and calculate alpha (the diagonal element)
            s = 1.0 if A_work[k, k] >= 0.0 else -1.0
            alpha = -s * norm_x
            
            d[k] = alpha  
            
            # Create reflector v in-place in A_work
            A_work[k, k] -= alpha
            
            v_norm = 0.0
            for i in range(k, m):
                v_norm += A_work[i, k] ** 2
            v_norm = np.sqrt(v_norm)
            
            if v_norm > 1e-14:
                # Normalize the reflector vector
                for i in range(k, m):
                    A_work[i, k] /= v_norm
                
                # Apply H to the remaining submatrix A_work[k:m, k+1:n]
                # This is slow - iterating over rows for each column update
                for j in range(k+1, n):
                    dot = 0.0
                    for i in range(k, m):
                        dot += A_work[i, k] * A_work[i, j]
                    dot *= 2.0
                    
                    for i in range(k, m):
                        A_work[i, j] -= A_work[i, k] * dot
        else:
            d[k] = 0.0
        
        # Right Householder (Row)
        if k < n - 1:
            norm_x = 0.0
            for j in range(k+1, n):
                norm_x += A_work[k, j] ** 2
            norm_x = np.sqrt(norm_x)
            
            if norm_x > 1e-14:
                # Determine sign and calculate beta (the super-diagonal element)
                s = 1.0 if A_work[k, k+1] >= 0.0 else -1.0
                beta = -s * norm_x
                
                e[k] = beta  
                
                # Create reflector v in-place in A_work
                A_work[k, k+1] -= beta
                
                v_norm = 0.0
                for j in range(k+1, n):
                    v_norm += A_work[k, j] ** 2
                v_norm = np.sqrt(v_norm)
                
                if v_norm > 1e-14:
                    # Normalize the reflector vector
                    for j in range(k+1, n):
                        A_work[k, j] /= v_norm
                    
                    # Apply H to the remaining submatrix A_work[k+1:m, k+1:n]
                    for i in range(k+1, m): # Note: was k, should be k+1? No, k is correct
                        dot = 0.0
                        for j in range(k+1, n):
                            dot += A_work[i, j] * A_work[k, j]
                        dot *= 2.0
                        
                        for j in range(k+1, n):
                            A_work[i, j] -= dot * A_work[k, j]
            else:
                e[k] = 0.0
    
    return d, e, A_work


@njit(fastmath=True, cache=True)
def reconstruct_from_bidiag(A_work, U_B, Vt_B):
    """
    Numba-accelerated reconstruction.d
    """
    m, n = A_work.shape
    
    # --- Reconstruct U = P * U_B ---
    # U_B is (n, n). We seed U as an (m, n) matrix with U_B in the
    # top-left corner, and zeros elsewhere.
    U = np.zeros((m, n), dtype=np.float64)
    for i in range(n):
        for j in range(n):
            U[i, j] = U_B[i, j]

    # Apply P = H_0 * H_1 * ... * H_{n-1}
    # We apply them in reverse order (H_{n-1} ... H_0)
    for k in range(n-1, -1, -1):
        # Get Householder vector v = A_work[k:m, k]
        v_norm_sq = 0.0
        for i in range(k, m):
            v_norm_sq += A_work[i, k] ** 2
        
        if v_norm_sq > 1e-14:
            # Apply H to U[k:m, :]
            # H = I - 2*v*v^T
            for j in range(U.shape[1]): # Loop over columns of U
                dot = 0.0
                for i in range(k, m):
                    dot += A_work[i, k] * U[i, j]
                dot *= 2.0
                
                for i in range(k, m):
                    U[i, j] -= A_work[i, k] * dot
    
    # --- Reconstruct V^T = Vt_B * Q^T ---
    # We form V = Q * Vt_B^T, then transpose at the end.
    V = Vt_B.T.copy() # V is (n, n)
    
    # Apply Q = H_0 * ... * H_{n-2}
    # We apply them in reverse order (H_{n-2} ... H_0)
    for k in range(n-2, -1, -1):
        # Get Householder vector v = A_work[k, k+1:n]
        v_norm_sq = 0.0
        for j in range(k+1, n):
            v_norm_sq += A_work[k, j] ** 2
        
        if v_norm_sq > 1e-14:
            # Apply H to V[k+1:n, :]
            # (Note: V is (n,n), so V[k+1:n, :] is correct)
            for j in range(V.shape[1]): # Loop over columns of V
                dot = 0.0
                for i in range(k+1, n):
                    dot += A_work[k, i] * V[i, j]
                dot *= 2.0
                
                for i in range(k+1, n):
                    V[i, j] -= A_work[k, i] * dot
    
    return U, V.T


def golub_kahan_svd_numba(A, full_matrices=False):
    """
    Numba-accelerated version.
    """
    m, n = A.shape
    
    # Handle m < n case by transposing
    if m < n:
        # A = U S Vt  =>  A.T = V S Ut
        # We compute SVD of A.T, then swap U and V and transpose them
        V, S, Ut = golub_kahan_svd_numba(A.T, full_matrices=full_matrices)
        return Ut.T, S, V.T
    
    # Ensure contiguous float64 array for Numba
    A_c = np.ascontiguousarray(A, dtype=np.float64)
    
    # 1. Numba Bidiagonalization
    d, e, A_work = householder_bidiag_numba(A_c)
    
    # 2. Reconstruct Bidiagonal Matrix B
    B = np.diag(d)
    if len(e) > 0:
        B = B + np.diag(e, k=1)
    
    # 3. SVD of B (this is fast)
    U_B, S, Vt_B = svd(B, full_matrices=False)
    
    # 4. Numba Reconstruction
    # (The function was renamed reconstruct_from_bidiag in the prompt)
    U, Vt = reconstruct_from_bidiag(A_work, U_B, Vt_B)
        
    return U, S, Vt


# ============================================================================
# BENCHMARKING
# ============================================================================

def comprehensive_test():
    """
    Test correctness and performance.
    """
    print("=" * 90)
    print("COMPREHENSIVE GOLUB-KAHAN SVD TEST")
    print("=" * 90)
    
    # Add a "warm-up" call to compile the Numba functions
    print("Warming up Numba JIT compiler (this may take a second)...")
    A_warmup = np.random.randn(10, 5)
    golub_kahan_svd_numba(A_warmup)
    print("Warm-up complete. Starting benchmark.")
    print("=" * 90)

    
    sizes = [(50, 50), (100, 100), (200, 200), (500, 500), (1000, 1000)]
    
    for m, n in sizes:
        A = np.random.randn(m, n)
        
        # Test all implementations        
        t0 = time.perf_counter()
        U_sp, S_sp, Vt_sp = svd(A, full_matrices=False)
        t_sp = time.perf_counter() - t0
        
        t0 = time.perf_counter()
        U_nb, S_nb, Vt_nb = golub_kahan_svd_numba(A, full_matrices=False)
        t_nb = time.perf_counter() - t0
        
        # Compute errors
        
        A_sp = U_sp @ np.diag(S_sp) @ Vt_sp
        err_sp = np.linalg.norm(A - A_sp, 'fro') / np.linalg.norm(A, 'fro')
        
        A_nb = U_nb @ np.diag(S_nb) @ Vt_nb
        err_nb = np.linalg.norm(A - A_nb, 'fro') / np.linalg.norm(A, 'fro')
                
        orth_U_sp = np.linalg.norm(U_sp.T @ U_sp - np.eye(U_sp.shape[1]), 'fro')
        orth_V_sp = np.linalg.norm(Vt_sp @ Vt_sp.T - np.eye(Vt_sp.shape[0]), 'fro')
        
        orth_U_nb = np.linalg.norm(U_nb.T @ U_nb - np.eye(U_nb.shape[1]), 'fro')
        orth_V_nb = np.linalg.norm(Vt_nb @ Vt_nb.T - np.eye(Vt_nb.shape[0]), 'fro')
                
        print(f"Size {m}x{n}: "
              f"SciPy={t_sp:.4f}s, GK+Numba={t_nb:.4f}s")
        print(f"  Reconstruction: SciPy={err_sp:.2e}, Numba={err_nb:.2e}")
        print(f"  Orthogonality U: SciPy={orth_U_sp:.2e}, Numba={orth_U_nb:.2e}")
        print(f"  Orthogonality V: SciPy={orth_V_sp:.2e}, Numba={orth_V_nb:.2e}")
        print()
    
    print("=" * 90)


if __name__ == '__main__':
    comprehensive_test()