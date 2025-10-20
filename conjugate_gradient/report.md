# Comparative Analysis of Conjugate Gradient Methods with Preconditioning

**Experimental Study on SPD and Banded Matrices**  
*Authors: Experimental Study*  
*Date:*

---

## Abstract

We present a comprehensive experimental comparison of conjugate gradient (CG) methods for solving linear systems $Ax = b$ where $A$ is symmetric positive definite (SPD). We evaluate steepest descent, standard CG, preconditioned CG with Jacobi preconditioners, and compare against SciPy's implementation and Gaussian elimination. Experiments span problem sizes $n \in \{200,1000,2000,5000\}$ and condition numbers $\kappa \in \{10^2,10^4,10^6\}$ on both dense SPD and banded matrices. Results show PCG methods significantly reduce iterations but not always wall-clock time for dense matrices; Gaussian elimination remains competitive for moderate-sized dense systems.

---

## Introduction

The conjugate gradient method is a fundamental iterative algorithm for solving large sparse linear systems. For an SPD matrix $A \in \mathbb{R}^{n\times n}$, CG minimizes the quadratic form $\phi(x)=\tfrac12 x^T A x - b^T x$ and converges in at most $n$ iterations in exact arithmetic. The convergence rate depends on the condition number $\kappa(A)=\lambda_{\max}/\lambda_{\min}$, with the error bound:

$$
\|x_k - x^*\|_A \le 2\left(\frac{\sqrt{\kappa}-1}{\sqrt{\kappa}+1}\right)^k \|x_0 - x^*\|_A.
$$

Preconditioning transforms the system to $M^{-1}Ax = M^{-1}b$ where $M\approx A$ is easy to invert, potentially reducing the effective condition number.

---

## Experimental Setup

### Matrix Generation

- **Dense SPD matrices:** Constructed via an eigen-decomposition $A = Q \Lambda Q^{T}$ with reproducible randomness. Specifically:

    1. Draw $X\in\mathbb{R}^{n\times n}$ with i.i.d. $X_{ij}\sim\mathcal{N}(0,1)$ using a fixed RNG seed for reproducibility. Obtain an orthogonal matrix $Q$ from the QR factorization $X = \tilde Q\tilde R$ and enforce a deterministic sign convention, e.g.
         $$
         Q = \tilde Q \,\operatorname{diag}(\operatorname{sign}(\operatorname{diag}(\tilde R))).
         $$
    2. Set the eigenvalues uniformly spaced from $1$ to $\kappa$:
         $$
         \lambda_i = 1 + (i-1)\frac{\kappa-1}{n-1},\qquad i=1,\dots,n,
         $$
         and form $\Lambda=\operatorname{diag}(\lambda_1,\dots,\lambda_n)$ (equivalently, use linspace(1,κ,n)).
    3. Introduce per‑row scaling to break rotational symmetry and produce nonuniform diagonal dominance by
         $$
         D=\operatorname{diag}\big(\operatorname{logspace}(0,6,n)\big),
         $$
         where $\operatorname{logspace}(0,6,n)=10^{\operatorname{linspace}(0,6,n)}$.
    4. Form the matrix
         $$
         A \;=\; D\,(Q\Lambda Q^{T})\,D \;=\; (D Q)\Lambda (D Q)^{T}.
         $$
         Since $\Lambda\succ 0$ and $D$ is invertible, $A$ is symmetric positive definite. The diagonal scaling preserves SPDness while adding realistic per‑row scaling and variability in conditioning.

- **Banded matrices:** Constructed as tridiagonal-like matrices with diagonal entries
    $$d_i = \kappa - (i-1)\frac{\kappa-1}{n-1}$$
    and small random off-diagonal bands to ensure positive definiteness.

### Methods Compared

- **Steepest Descent:** Line search along gradient direction.  
- **CG (Custom):** Our implementation.  
- **CG (SciPy):** `scipy.sparse.linalg.cg`.  
- **PCG (Jacobi):** $M=\operatorname{diag}(A)$.  
- **PCG (Inverse):** $M=A$ (ideal preconditioner).  
- **Gaussian Elimination:** Direct solver via LU decomposition.

### Parameters

- Problem sizes: $n \in \{200,1000,2000,5000\}$.  
- Condition numbers: $\kappa \in \{10^2,10^4,10^6\}$.  
- Convergence tolerance: $\|r_k\| \le 10^{-8}\|r_0\|$.  
- Test system: MacBook Pro, 2.2 GHz Intel Core i7.

---

## Results and Analysis

### Convergence Behavior

Figure: residual convergence for $n=200$ across $\kappa$ values. Key observations:

1. **Preconditioning accelerates convergence:** PCG (Jacobi, Inverse) consistently reach tolerance faster than unpreconditioned CG.  
2. **Steepest descent is slowest:** Linear convergence vs. superlinear for CG.  
3. **Inverse preconditioner achieves rapid convergence:** Converges in 1–2 iterations (effectively solving the system).  
4. **SciPy vs Custom CG:** Nearly identical convergence profiles, validating implementation.

![Residual convergence for dense SPD matrices with n=200](imgs/conjugate_gradient/1.convergence_SPD_Dense_n200.png)  
*Figure: Residual convergence for dense SPD matrices with n=200.*

For larger systems ($n=5000$):

- All methods require significantly more iterations.  
- PCG methods maintain advantage but gap narrows.  
- For $\kappa=10^6$, unpreconditioned CG struggles to converge within 10,000 iterations.

![Residual convergence for n=5000](imgs/conjugate_gradient/1.convergence_SPD_Dense_n5000.png)  
*Figure: Residual convergence for n=5000.*

### Effect of Condition Number

Critical finding: iterations for our dense SPD matrices are surprisingly independent of $\kappa$, contradicting the classical $O(\sqrt{\kappa})$ expectation.

Explanation: the eigenvalue distribution
$$\lambda_i = 1 + (i-1)\frac{\kappa-1}{n-1}$$
creates many clustered eigenvalues. CG convergence depends on eigenvalue distribution, not just $\kappa$. With most eigenvalues clustered, CG identifies dominant eigendirections quickly regardless of extreme eigenvalues.

Implication: Jacobi preconditioners show minimal improvement because they don't change the clustering pattern for these matrices.

![Iterations vs. condition number](imgs/conjugate_gradient/3.iterations_vs_kappa_all_n_SPD_Dense.png)  
*Figure: Iterations vs. condition number (flat trends).*

### Computational Time Analysis

Figure: wall-clock time vs. problem size. Observations:

1. **Gaussian elimination dominates for dense matrices:** Direct solve (O(n^3)) is faster than iterative methods for $n \le 5000$.  
2. **Iterative methods scale better asymptotically:** CG scales roughly O(k n^2) per iteration; becomes competitive for very large $n$.  
3. **PCG overhead:** Preconditioning reduces iterations but adds per-iteration cost to apply $M^{-1}$.  
4. **SciPy CG is slower:** Higher overhead than custom implementation due to Python callbacks.

![Computation time vs. problem size](imgs/conjugate_gradient/4.time_vs_size_all_kappa_SPD_Dense.png)  
*Figure: Computation time vs. problem size.*

### Speedup Analysis

- For $n=200$: PCG (Inverse) achieves ~3.5× speedup at $\kappa=10^2$ but drops to ~1× at $\kappa=10^6$.  
- For $n \ge 1000$: All iterative methods are slower than Gaussian elimination for dense matrices.  

Conclusion: For dense systems, direct methods remain superior unless $n>10^4$ or memory constraints exist.

![Speedup over Gaussian elimination](imgs/conjugate_gradient/5.speedup_analysis_SPD_Dense.png)  
*Figure: Speedup over Gaussian elimination.*

### Method Comparison Summary

At $n=5000$:

- Iterations: PCG (Jacobi) requires ≈200 iterations vs. ≈10,000 for unpreconditioned CG at $\kappa=10^6$ (≈50× reduction).  
- Time: Despite fewer iterations, PCG (Jacobi) takes ≈5s vs. ≈100s for CG (Custom) — only ~20× faster in iterations due to per-iteration overhead.  
- Gaussian elimination: completes in ≈2s, outperforming all iterative methods for this dense problem.

![Method comparison at n=5000](imgs/conjugate_gradient/6.method_comparison_n5000.png)  
*Figure: Method comparison at n=5000.*

### Nonlinear Conjugate Gradient

Setup: Minimize Rosenbrock function using Fletcher–Reeves (F–R) CG and preconditioned nonlinear CG with diagonal Hessian approximation.

Findings:

- Preconditioned CG (diagonal) converges in fewer iterations.  
- At higher dimensions, preconditioning provides 2–3× speedup.  
- Gradient norm decreases consistently.

![Nonlinear CG methods comparison](imgs/conjugate_gradient/7.nonlinear_comparison.png)  
*Figure: Nonlinear CG methods comparison.*

---

## Key Insights

1. **PCG reduces iterations significantly** (10–50×) but wall-clock time improvement is modest (2–5×) for dense matrices due to preconditioning overhead.  
2. **Condition number effect is matrix-dependent:** Spectral distribution matters more than just $\kappa(A)$.  
3. **Jacobi preconditioning is ineffective** for uniformly-scaled matrices with linearly spread eigenvalues and random eigenvectors.  
4. **Gaussian elimination wins for dense systems**: Direct methods are faster for $n < 10^4$ in dense settings. Iterative methods excel for sparse/structured matrices.  
5. **SciPy CG matches custom implementation** in convergence but has higher overhead.  
6. **Inverse preconditioner ($M=A$)** achieves ideal convergence but is impractical ($O(n^3)$ setup).  
7. **Banded matrices favor iterative methods:** Structure allows efficient mat-vec products, making CG competitive (not fully shown here due to space).

---

## Conclusion

This experimental study demonstrates that while preconditioning theoretically improves CG convergence, practical benefits depend critically on matrix structure, implementation efficiency, and problem size. For dense SPD systems, direct methods remain the gold standard for moderate $n$. Future work should explore incomplete Cholesky and multigrid preconditioners for structured problems, and investigate GPU-accelerated implementations for large-scale systems.

**Code:** https://github.com/bingyulab/paper-replication
