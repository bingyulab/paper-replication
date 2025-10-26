# Python code to run conjugate gradient experiments and produce plots.
# This code will:
# - Implement Steepest Descent (SD), Conjugate Gradient (CG), Preconditioned CG (PCG) with Jacobi and Richardson preconditioner.

# The outputs (plots) will be visible.
# Do not specify explicit colors or seaborn.

# Implementing:
# We'll run:
# 1) Linear test: SPD matrix, compare CG, PCG (Jacobi), PCG (Richardson)
# Plots: residual norms for linear; gradient norms for nonlinear.
import numpy as np
from numpy.linalg import norm
from contextlib import contextmanager
from time import perf_counter
import os
import pandas as pd
from scipy.sparse.linalg import cg as scipy_cg, LinearOperator
from scipy.linalg import solve, eigvals

import json
import warnings
warnings.filterwarnings('ignore')


# ---------- Timing utility ----------
@contextmanager
def timer(name):
    """Context manager for timing code blocks"""
    start = perf_counter()
    result = {'time': 0, 'iters': 0}
    yield result
    result['time'] = perf_counter() - start
    # print(f"{name}: {result['iters']} iters, {result['time']:.4f}s")

# ---------- Helpers ----------

def report(A):
    # assuming A is your matrix and M = np.diag(np.diag(A))
    diagA = np.diag(A)
    print("diag(A) stats: min, max, mean, std:", diagA.min(), diagA.max(), diagA.mean(), diagA.std())

    # condition numbers
    print("cond(A) (numpy):", np.linalg.cond(A))

    # Jacobi preconditioned matrix (left preconditioning)
    M_inv = np.diag(1.0 / diagA)
    B = M_inv @ A
    print("cond(M^{-1} A):", np.linalg.cond(B))

    # eigenvalues (a quick look)
    ev = np.sort(np.real(eigvals(A)))
    print("eigenvalues (smallest 5):", ev[:5])
    print("eigenvalues (largest 5):", ev[-5:])  
    
    # 1) symmetry check
    sym_diff = np.max(np.abs(A - A.T))
    print("max |A - A.T| =", sym_diff)

    # 2) use symmetric eigen-solver on symmetrized A
    A_sym = (A + A.T) * 0.5
    ev_sym = np.linalg.eigvalsh(A_sym)      # robust for symmetric
    print("eigsh smallest 5 (symmetrized):", ev_sym[:5])
    print("eigsh largest 5 (symmetrized):", ev_sym[-5:])

    # 3) try Cholesky (will raise if not SPD)
    try:
        np.linalg.cholesky(A_sym)
        print("Cholesky OK on symmetrized A (numerically SPD).")
    except np.linalg.LinAlgError as e:
        print("Cholesky failed:", e)  
    
def make_spd(n, kappa, scale_factor=2, seed=None):
    """Generate SPD matrix using Hilbert-like structure with prescribed condition number"""
    rng = np.random.default_rng(seed)
    # Generate eigenvalues with prescribed condition number
    eigs = np.logspace(0, np.log10(kappa), n)
    # Random orthogonal matrix
    X = rng.standard_normal((n, n))
    Q, _ = np.linalg.qr(X)
    # Construct A = Q @ diag(eigs) @ Q.T: 
    # random orthogonal no difference, hence we add nonuniform diagonal dominance
    D = np.diag(np.linspace(1, scale_factor, n))
    # D = np.diag(np.sqrt(rng.uniform(0.5, 2.0, n)))
    A = D @ (Q @ np.diag(eigs) @ Q.T) @ D
    
    # Get actual eigendecomposition
    actual_eigs, V = np.linalg.eigh(A)
    
    # Replace eigenvalues with target ones, keep eigenvectors
    target_eigs = np.logspace(0, np.log10(kappa), n)
    A = V @ np.diag(target_eigs) @ V.T
    report(A)
    return A

def make_band(n, kappa, band_width=3, seed=None):
    """Generate banded SPD matrix with prescribed condition number"""
    rng = np.random.default_rng(seed)
    # Diagonal with prescribed condition number
    diag_vals = np.logspace(np.log10(kappa), 1, n)[::-1]
    A = np.diag(diag_vals)
    # Add off-diagonal bands
    for k in range(1, band_width + 1):
        off_diag = rng.uniform(0, 0.1 * min(diag_vals), n - k)
        A += np.diag(off_diag, k) + np.diag(off_diag, -k)
    return A

def steepest_descent(A, b, x0, tol=1e-8, maxit=10000):
    x = x0.copy()
    r = b - A @ x
    res = [norm(r)]
    iters = 0
    while res[-1] > tol * res[0]:
        q = A @ r
        alpha = (r @ r) / (r @ q)
        x = x + alpha * r
        if iters > 0 and iters % 50 == 0:
            r = b - A @ x
        else:
            r = r - alpha * q
        res.append(norm(r))
        iters += 1
        if iters > maxit:  # Safety limit
            break
    return x, np.array(res), iters

# Linear methods (reused from previous)
def conjugate_gradient(A, b, x0, tol=1e-8, maxit=10000):
    x = x0.copy()
    r = b - A @ x
    d = r.copy()
    delta_new = r @ r
    r0_norm = np.sqrt(delta_new)
    res = [r0_norm]
    iters = 0
    while res[-1] > tol * res[0]:
        q = A @ d
        alpha = delta_new / (d @ q)
        x = x + alpha * d
        if iters > 0 and iters % 50 == 0:
            r = b - A @ x
        else:
            r = r - alpha * q
        delta_old = delta_new
        delta_new = r @ r
        res.append(np.sqrt(delta_new))
        iters += 1
        if iters > maxit:
            break
        beta = delta_new / delta_old
        d = r + beta * d
    return x, np.array(res), iters

def pcg(A, b, x0, Msolver, tol=1e-8, maxit=10000):
    x = x0.copy()
    r = b - A @ x
    r0_norm = norm(r)
    z = Msolver(r)
    d = z.copy()
    delta_new = r @ z
    res = [r0_norm]
    iters = 0
    while res[-1] > tol * res[0]:
        q = A @ d
        alpha = delta_new / (d @ q)
        x = x + alpha * d
        if iters > 0 and iters % 50 == 0:
            r = b - A @ x
        else:
            r = r - alpha * q
        res.append(norm(r))
        iters += 1
        if iters > maxit:
            break
        z = Msolver(r)
        delta_old = delta_new
        delta_new = r @ z
        beta = delta_new / delta_old
        d = z + beta * d
    return x, np.array(res), iters


# ---------- Preconditioners ----------
def jacobi_preconditioner(A):
    Dinv = np.diag(1.0 / np.diag(A))
    return lambda r: Dinv @ r

def richardson_preconditioner(A, tau=None):
    if tau is None:
        tau = 1.0 / np.mean(np.diag(A))
    return lambda r: tau * r

def inverse_preconditioner(A):
    """Full matrix inverse preconditioner (only for small problems!)"""
    Ainv = np.linalg.inv(A)
    return lambda r: Ainv @ r


# ---------- SciPy and Direct Solvers ----------
def scipy_cg_wrapper(A, b, x0, tol=1e-8, M=None, maxit=10000):
    """Wrapper for scipy.sparse.linalg.cg"""
    res_history = []
    def callback(xk):
        res_history.append(norm(b - A @ xk))
    
    res_history.append(norm(b - A @ x0))
    start = perf_counter()
    x, info = scipy_cg(A, b, x0=x0, rtol=tol, M=M, callback=callback, atol=0, maxiter=maxit)
    time_taken = perf_counter() - start
    
    if len(res_history) > 0:
        res_history.append(norm(b - A @ x))
    return x, np.array(res_history), len(res_history) - 1, time_taken

def gaussian_elimination(A, b):
    """Direct solver using Gaussian elimination (LU decomposition)"""
    start = perf_counter()
    x = solve(A, b)
    time_taken = perf_counter() - start
    return x, time_taken

# ---------- Test problems and runs ----------
# 1) Linear: compare CG, PCG(Jacobi), PCG(Richardson), PCG(IC) for different problem sizes and condition numbers
problem_sizes = [200, 1000, 2000, 5000]
condition_numbers = [1e2, 1e4, 1e6]
matrix_types = ['SPD Dense', 'Banded']
tol = 1e-8
maxit = 10000

all_results = []
convergence_data = {}  # Store residual histories

for n in problem_sizes:
    for kappa in condition_numbers:
        print(f"\n{'='*80}")
        print(f"Problem size: n={n}, Condition number: κ={kappa:.0e}")
        print('='*80)
        
        # Generate test problems
        rng = np.random.default_rng(123)
        x_true = rng.standard_normal(n)
        
        for matrix_type in matrix_types:
            print(f"\nMatrix Type: {matrix_type}")
            print('-'*80)
            
            key = (n, kappa, matrix_type)
            convergence_data[key] = {}
            
            if matrix_type == 'SPD Dense':
                A = make_spd(n, kappa, seed=123)
            else:
                band_width = max(3, int(0.05 * n))
                A = make_band(n, kappa, band_width=band_width, seed=123)            
            
            b = A @ x_true
            x0 = np.zeros(n)
            
            # Gaussian Elimination (direct)
            x_ge, time_ge = gaussian_elimination(A, b)
            error_ge = norm(x_ge - x_true) / norm(x_true)
            print(f"{'Gaussian Elimination':<35} {0:<10} {time_ge:<12.4f} {error_ge:.2e}")
            all_results.append({
                'problem_size': n, 'condition_number': kappa, 'matrix_type': matrix_type,
                'method': 'Gaussian Elimination', 'iterations': 0, 'time': time_ge,
                'final_residual': norm(b - A @ x_ge), 'relative_error': error_ge
            })
            
            # Steepest Descent
            with timer("Steepest Descent") as t:
                x_sd, res_sd, iters_sd = steepest_descent(A, b, x0, tol=tol, maxit=maxit)
                t['iters'] = iters_sd
            error_sd = norm(x_sd - x_true) / norm(x_true)
            print(f"{'Steepest Descent':<35} {iters_sd:<10} {t['time']:<12.4f} {error_sd:.2e}")
            convergence_data[key]['Steepest Descent'] = res_sd
            all_results.append({
                'problem_size': n, 'condition_number': kappa, 'matrix_type': matrix_type,
                'method': 'Steepest Descent', 'iterations': iters_sd, 'time': t['time'],
                'final_residual': res_sd[-1], 'relative_error': error_sd
            })
            
            # Conjugate Gradient (Custom)
            with timer("CG (Custom)") as t:
                x_cg, res_cg, iters_cg = conjugate_gradient(A, b, x0, tol=tol, maxit=maxit)
                t['iters'] = iters_cg
            error_cg = norm(x_cg - x_true) / norm(x_true)            
            print(f"{'CG (Custom)':<35} {iters_cg:<10} {t['time']:<12.4f} {error_cg:.2e}")
            convergence_data[key]['CG (Custom)'] = res_cg
            all_results.append({
                'problem_size': n, 'condition_number': kappa, 'matrix_type': matrix_type,
                'method': 'CG (Custom)', 'iterations': iters_cg, 'time': t['time'],
                'final_residual': res_cg[-1], 'relative_error': error_cg
            })
            
            # SciPy CG
            x_scipy, res_scipy, iters_scipy, time_scipy = scipy_cg_wrapper(A, b, x0, tol=tol, maxit=maxit)
            error_scipy = norm(x_scipy - x_true) / norm(x_true)
            convergence_data[key]['CG (SciPy)'] = res_scipy
            print(f"{'CG (SciPy)':<35} {iters_scipy:<10} {time_scipy:<12.4f} {error_scipy:.2e}")
            all_results.append({
                'problem_size': n, 'condition_number': kappa, 'matrix_type': matrix_type,
                'method': 'CG (SciPy)', 'iterations': iters_scipy, 'time': time_scipy,
                'final_residual': res_scipy[-1] if len(res_scipy) > 0 else 0,
                'relative_error': error_scipy
            })
            
            # PCG (Jacobi)
            with timer("PCG (Jacobi)") as t:
                Mj = jacobi_preconditioner(A)
                x_pj, res_pj, iters_pj = pcg(A, b, x0, Mj, tol=tol, maxit=maxit)
                t['iters'] = iters_pj
            error_pj = norm(x_pj - x_true) / norm(x_true)            
            print(f"{'PCG (Jacobi)':<35} {iters_pj:<10} {t['time']:<12.4f} {error_pj:.2e}")
            convergence_data[key]['PCG (Jacobi)'] = res_pj
            all_results.append({
                'problem_size': n, 'condition_number': kappa, 'matrix_type': matrix_type,
                'method': 'PCG (Jacobi)', 'iterations': iters_pj, 'time': t['time'],
                'final_residual': res_pj[-1], 'relative_error': error_pj
            })
            
            # SciPy PCG (Jacobi)
            M_jacobi = LinearOperator((n, n), matvec=jacobi_preconditioner(A))
            x_scipy_pj, res_scipy_pj, iters_scipy_pj, time_scipy_pj = scipy_cg_wrapper(
                A, b, x0, tol=tol, M=M_jacobi, maxit=maxit)
            error_scipy_pj = norm(x_scipy_pj - x_true) / norm(x_true)
            convergence_data[key]['PCG (Jacobi, SciPy)'] = res_scipy_pj
            print(f"{'PCG (Jacobi, SciPy)':<35} {iters_scipy_pj:<10} {time_scipy_pj:<12.4f} {error_scipy_pj:.2e}")
            all_results.append({
                'problem_size': n, 'condition_number': kappa, 'matrix_type': matrix_type,
                'method': 'PCG (Jacobi, SciPy)', 'iterations': iters_scipy_pj,
                'time': time_scipy_pj, 'final_residual': res_scipy_pj[-1] if len(res_scipy_pj) > 0 else 0,
                'relative_error': error_scipy_pj
            })
            
            # PCG (Inverse)
            with timer("PCG (Inverse)") as t:
                Mj = inverse_preconditioner(A)
                x_pj, res_pj, iters_pj = pcg(A, b, x0, Mj, tol=tol, maxit=maxit)
                t['iters'] = iters_pj
            error_pj = norm(x_pj - x_true) / norm(x_true)            
            print(f"{'PCG (Inverse)':<35} {iters_pj:<10} {t['time']:<12.4f} {error_pj:.2e}")
            convergence_data[key]['PCG (Inverse)'] = res_pj
            all_results.append({
                'problem_size': n, 'condition_number': kappa, 'matrix_type': matrix_type,
                'method': 'PCG (Inverse)', 'iterations': iters_pj, 'time': t['time'],
                'final_residual': res_pj[-1], 'relative_error': error_pj
            })
            
            # SciPy PCG (Inverse)
            M_inverse = LinearOperator((n, n), matvec=inverse_preconditioner(A))
            x_scipy_pj, res_scipy_pj, iters_scipy_pj, time_scipy_pj = scipy_cg_wrapper(
                A, b, x0, tol=tol, M=M_inverse, maxit=maxit)
            error_scipy_pj = norm(x_scipy_pj - x_true) / norm(x_true)
            convergence_data[key]['PCG (Inverse, SciPy)'] = res_scipy_pj
            print(f"{'PCG (Inverse, SciPy)':<35} {iters_scipy_pj:<10} {time_scipy_pj:<12.4f} {error_scipy_pj:.2e}")
            all_results.append({
                'problem_size': n, 'condition_number': kappa, 'matrix_type': matrix_type,
                'method': 'PCG (Inverse, SciPy)', 'iterations': iters_scipy_pj,
                'time': time_scipy_pj, 'final_residual': res_scipy_pj[-1] if len(res_scipy_pj) > 0 else 0,
                'relative_error': error_scipy_pj
            })
                        
# Save results
df_results = pd.DataFrame(all_results)
os.makedirs('imgs/conjugate_gradient', exist_ok=True)
df_results.to_csv('imgs/conjugate_gradient/cg_results.csv', index=False)

# Convert tuple keys to strings and make NumPy arrays (and nested structures) JSON-serializable
def make_json_serializable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [make_json_serializable(v) for v in obj]
    return obj

convergence_data_str_keys = {str(k): make_json_serializable(v) for k, v in convergence_data.items()}
with open('imgs/conjugate_gradient/cg_convergence.json', 'w') as f:
    json.dump(convergence_data_str_keys, f, indent=2)

print("\n" + "="*80)
print("Results saved to: imgs/conjugate_gradient/cg_results.csv")
print("="*80)
print("="*80)
