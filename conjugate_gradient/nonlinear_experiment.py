# Python code to run conjugate gradient experiments and produce plots.
# This code will:
# - Nonlinear CG (Fletcher-Reeves + Newton-Raphson inner step) as B4
# - Preconditioned Nonlinear CG with Secant line search and Polak-Ribiere beta as B5

# The outputs (plots) will be visible.
# Do not specify explicit colors or seaborn.

# Implementing:
# We'll run:
# 1) Nonlinear test: Rosenbrock function in 2D (and a 10D generalised Rosenbrock) with B4 and B5
# Plots: residual norms for linear; gradient norms for nonlinear.
import numpy as np
from contextlib import contextmanager
from time import perf_counter
import pandas as pd


# ---------- Timing utility ----------
@contextmanager
def timer(name):
    """Context manager for timing code blocks"""
    start = perf_counter()
    result = {'time': 0, 'iters': 0}
    yield result
    result['time'] = perf_counter() - start
    print(f"{name}: {result['iters']} iters, {result['time']:.4f}s")


# ---------- Nonlinear methods B4 and B5 ----------
# We'll implement for general f: need grad and a Hessian-vector product (or full Hessian).
# For simplicity use full Hessian for moderate dimensions.

def rosenbrock(x):
    x = np.asarray(x)
    n = x.size
    val = 0.0
    for i in range(0, n-1):
        val += 100.0*(x[i+1]-x[i]**2)**2 + (1.0 - x[i])**2
    return val

def grad_rosen(x):
    x = np.asarray(x)
    n = x.size
    g = np.zeros_like(x)
    for i in range(n):
        if i < n-1:
            g[i] += -400*x[i]*(x[i+1]-x[i]**2) - 2*(1-x[i])
        if i > 0:
            g[i] += 200*(x[i]-x[i-1]**2)
    return g

def hess_rosen(x):
    x = np.asarray(x)
    n = x.size
    H = np.zeros((n,n))
    for i in range(n):
        if i < n-1:
            H[i,i] += -400*(x[i+1]-x[i]**2) + 800*x[i]**2 + 2
            H[i,i+1] += -400*x[i]
            H[i+1,i] += -400*x[i]
        if i > 0:
            H[i,i] += 200
    return H

def nonlinear_cg_fletcher_reeves_newton(f, grad, hess, x0, i_max=50, eps_cg=1e-6,
                                       j_max=20, eps_nr=1e-6):
    x = x0.copy()
    i = 0
    k = 0
    r = -grad(x)
    d = r.copy()
    delta_new = r @ r
    delta0 = delta_new
    res = [np.sqrt(delta_new)]
    n = x.size
    while i < i_max and delta_new > (eps_cg**2) * delta0:
        j = 0
        delta_d = d @ d
        # inner Newton-Raphson-like step(s) along direction d using exact 1D Newton step
        while True:
            Hd = hess(x) @ d
            gradx = grad(x)
            denom = d @ Hd
            if abs(denom) < 1e-14:
                alpha = 0.0
            else:
                alpha = - (gradx @ d) / denom
            x = x + alpha * d
            j += 1
            if not (j < j_max and (alpha*alpha*delta_d > eps_nr*eps_nr)):
                break
        r = -grad(x)
        delta_old = delta_new
        delta_new = r @ r
        beta = delta_new / delta_old if delta_old != 0 else 0.0
        d = r + beta * d
        k += 1
        if k == n or (r @ d) <= 0:
            d = r.copy()
            k = 0
        i += 1
        res.append(np.sqrt(delta_new))
    return x, np.array(res)

def nonlinear_pcg_secant_polak_ribiere(f, grad, hess, x0, Msolver,
                                       i_max=50, eps_cg=1e-6,
                                       sigma0=1e-3, j_max=20, eps_sec=1e-6):
    x = x0.copy()
    i = 0
    k = 0
    r = -grad(x)
    # preconditioner applied
    s = Msolver(r)
    d = s.copy()
    delta_new = r @ s
    delta0 = delta_new
    res = [np.sqrt(r @ r)]
    n = x.size
    while i < i_max and delta_new > (eps_cg**2) * delta0:
        j = 0
        delta_d = d @ d
        alpha = -sigma0  # initial secant step parameter (negative sign as in description)
        # compute eta_prev = grad(x + sigma0 d)^T d
        eta_prev = grad(x + sigma0 * d) @ d
        while True:
            eta = grad(x) @ d
            denom = (eta_prev - eta)
            if abs(denom) < 1e-14:
                # fallback to simple backtracking-like small step
                alpha = -sigma0 * 0.5
            else:
                alpha = alpha * (eta / denom)
            x = x + alpha * d
            eta_prev = eta
            j += 1
            if not (j < j_max and (alpha*alpha*delta_d > eps_sec*eps_sec)):
                break
        
        r_old = r.copy()  # FIX: Save old residual
        s_old = s.copy()  # FIX: Save old preconditioned residual
        r = -grad(x)
        
        s = Msolver(r)
        
        delta_old = delta_new        
        delta_new = r @ s
        
        # Polak-Ribiere-like with preconditioning: beta = (r_new^T s_new - r_old^T s_old)/ (r_old^T s_old)
        beta = eta = max(0.0, (r @ s - r_old @ s) / delta_old) if delta_old > 1e-16 else 0.0
        k += 1
        if k == n or beta <= 0:
            d = s.copy()
            k = 0
        else:
            d = s + beta * d
        i += 1
        res.append(np.sqrt(r @ r))
    return x, np.array(res)

# 2) Nonlinear: Rosenbrock function (2D and 10D)
# Test on multiple dimensions
problem_sizes = [200, 1000, 2000, 5000]
condition_numbers = [1e2, 1e4, 1e6]
nonlinear_dims = [2, 10, 20, 50]
all_nonlinear_results = {}

for dim in nonlinear_dims:
    print(f"\n{dim}D Rosenbrock problem")
    print("-" * 60)
    
    x0 = np.full(dim, -1.2)
    x0[1::2] = 1.0
    
    results = {}
    
    with timer("Nonlin CG (F-R + Newton)") as t:
        x_b4, res_b4 = nonlinear_cg_fletcher_reeves_newton(rosenbrock, grad_rosen, hess_rosen,
                                                           x0, i_max=200, eps_cg=1e-6,
                                                           j_max=50, eps_nr=1e-6)
        t['iters'] = len(res_b4) - 1
    results['Nonlin CG (F-R + Newton)'] = {'res': res_b4, 'time': t['time'], 'iters': t['iters']}
    
    with timer("Precond Nonlin CG (diag)") as t:
        H0 = hess_rosen(x0)
        D0 = np.diag(np.diag(H0))
        D0[D0 == 0] = 1.0
        def Msolver_diag(r):
            return np.linalg.solve(D0, r)
        x_b5, res_b5 = nonlinear_pcg_secant_polak_ribiere(rosenbrock, grad_rosen, hess_rosen,
                                                          x0, Msolver_diag, i_max=200,
                                                          eps_cg=1e-6, sigma0=1e-3, j_max=50, eps_sec=1e-6)
        t['iters'] = len(res_b5) - 1
    results['Precond Nonlin CG (diag)'] = {'res': res_b5, 'time': t['time'], 'iters': t['iters']}
    
    all_nonlinear_results[dim] = results
    print("-" * 60)

# Summary table for nonlinear
print("\n" + "="*90)
print("NONLINEAR SUMMARY: Time and Iterations")
print("="*90)
for dim in nonlinear_dims:
    print(f"\nDimension = {dim}")
    print("-" * 90)
    print(f"{'Method':<35} {'Iterations':<15} {'Time (s)':<15}")
    print("-" * 90)
    for name, data in all_nonlinear_results[dim].items():
        print(f"{name:<35} {data['iters']:<15} {data['time']:<15.4f}")

# Replace the pickle save for nonlinear results with:
# Save nonlinear results to DataFrame
nonlinear_data = []
for dim in nonlinear_dims:
    for method, data in all_nonlinear_results[dim].items():
        nonlinear_data.append({
            'dimension': dim,
            'method': method,
            'iterations': data['iters'],
            'time': data['time'],
            'final_gradient_norm': data['res'][-1]
        })

df_nonlinear = pd.DataFrame(nonlinear_data)
df_nonlinear.to_csv('imgs/conjugate_gradient/nonlinear_results.csv', index=False)
print("Saved: imgs/conjugate_gradient/nonlinear_results.csv")

print("\n" + "="*90)
print("All plots and results saved successfully!")
print("="*90)
print("\nSaved files:")
print("- imgs/conjugate_gradient/linear_results.csv")
print("- imgs/conjugate_gradient/nonlinear_results.csv")
print("- Multiple PNG plots in imgs/conjugate_gradient/")
