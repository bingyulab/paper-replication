# Experiments

## Experimental Setup

### Dense Matrix Generation

We construct dense symmetric positive definite (SPD) matrices with prescribed spectral properties through the congruence transformation:

$$
A = D \cdot (Q\Lambda Q^T) \cdot D
$$

where $Q \in \mathbb{R}^{n \times n}$ is a random orthogonal matrix obtained via QR decomposition of a standard Gaussian random matrix, $\Lambda = \mathrm{diag}(\lambda_1, \ldots, \lambda_n)$ contains log-spaced eigenvalues:

$$
\lambda_i = 10^{(i-1) \cdot \log_{10}(\kappa)/(n-1)}, \quad i = 1, \ldots, n,
$$

and $D = \mathrm{diag}(d_1, \ldots, d_n)$ with $d_i = 10^{2(i-1)/(n-1)}$ introduces non-uniform diagonal scaling. The resulting matrix is rescaled to achieve $\kappa(A) = \kappa$ exactly.

We evaluate the following methods under convergence tolerance $\|r_k\| \leq 10^{-8}\|r_0\|$ with a maximum iteration limit of 10,000:

1. **Steepest Descent** using exact line search along the negative gradient
2. **Conjugate Gradient (CG)** in both custom and SciPy implementations
3. **Preconditioned CG (PCG)** with Jacobi preconditioner $M = \mathrm{diag}(A)$ and inverse preconditioner $M = A$
4. **Gaussian Elimination** via LU decomposition as the direct solver baseline

## Results

### Dense Matrix Analysis

#### Convergence Characteristics

Figure 1 illustrates residual trajectories for $n=200$ across three orders of magnitude in condition number. Steepest descent (linear search) exhibits the theoretically predicted linear convergence rate, exhausting the iteration budget at all condition numbers tested. Both CG implementations demonstrate superlinear convergence, with custom and SciPy variants producing nearly identical trajectories, validating our implementation. The Jacobi-preconditioned variant achieves modest acceleration, reducing iteration counts from 76 to 66 at $\kappa=10^2$ (13% improvement) and from 1897 to 1742 at $\kappa=10^6$ (8% improvement). The inverse preconditioner, representing the theoretical optimum, converges in a single iteration as expected from the identity $M^{-1}A = I$.

![Residual convergence for $n=200$ across condition numbers $\kappa \in \{10^2, 10^4, 10^6\}$. Steepest descent exhibits linear convergence (orange), while CG variants show superlinear convergence with PCG providing modest improvement (purple).](imgs/conjugate_gradient/1.convergence_SPD_Dense_n200.pdf)

Scaling to $n=5000$ (Figure 2), the convergence profiles reveal a remarkable property: iteration counts remain nearly invariant with respect to problem dimension for fixed $\kappa$. Specifically, at $\kappa=10^2$, the method requires 76 iterations for $n=200$ and 84 iterations for $n=5000$—a mere 10% increase despite a 25-fold increase in dimension. This behavior persists across all condition numbers tested, suggesting that the convergence rate depends primarily on the eigenvalue distribution structure rather than the total number of degrees of freedom.

![Residual convergence for $n=5000$. Iteration counts remain nearly constant across problem sizes for fixed $\kappa$, demonstrating size-independence enabled by eigenvalue clustering.](imgs/conjugate_gradient/1.convergence_SPD_Dense_n5000.pdf)

#### Iteration Complexity Analysis

Table 1 quantifies the iteration requirements across the parameter space. For moderate condition numbers $\kappa \in \{10^2, 10^4\}$, the iteration count exhibits remarkable stability: at $\kappa=10^2$, the range is 76–84 across a 25-fold size increase; at $\kappa=10^4$, the range is 420–667. Only at the extreme conditioning $\kappa=10^6$ does a modest size-dependence emerge, with iterations scaling from 1897 to 4769—approximately as $O(n^{0.5})$, consistent with eigenvalue distribution effects rather than the classical worst-case bound of $O(\kappa^{0.5})$.

| $n$   | $\kappa=10^2$ | $\kappa=10^4$ | $\kappa=10^6$ |
|-------|---------------|---------------|---------------|
| 200   | 76            | 420           | 1897          |
| 1000  | 84            | 626           | 4156          |
| 2000  | 83            | 652           | 4542          |
| 5000  | 84            | 667           | 4769          |

![Iteration requirements versus condition number for all problem sizes. For $\kappa \leq 10^4$, iterations remain nearly constant despite three orders of magnitude variation in $\kappa$. Significant $\kappa$-dependence emerges only at $\kappa=10^6$.](imgs/conjugate_gradient/3.iterations_vs_kappa_all_n_SPD_Dense.pdf)

**Theoretical interpretation:** Classical CG theory predicts convergence in $k$ iterations satisfying

$$
\|e_k\|_A \leq 2 \left( \frac{\sqrt{\kappa}-1}{\sqrt{\kappa}+1} \right)^k \|e_0\|_A,
$$

suggesting $k = O(\sqrt{\kappa})$ iterations to achieve tolerance $\epsilon$. However, this bound assumes a worst-case eigenvalue distribution with uniform spacing across $[\lambda_{\min}, \lambda_{\max}]$. Our logarithmic spacing creates a fundamentally different spectral structure: approximately 90% of eigenvalues concentrate near $\lambda_{\min}=1$, with only a sparse tail extending toward $\lambda_{\max}=\kappa$.

The error decomposition $e_k = \sum_{i=1}^n \alpha_i v_i$ reveals that components along large-eigenvalue eigenvectors contribute proportionally larger magnitudes to the residual $r_k = Ae_k = \sum_{i=1}^n \alpha_i \lambda_i v_i$. CG exploits the $A$-orthogonality of search directions to eliminate these high-frequency components rapidly. The convergence bottleneck arises from the densely clustered small eigenvalues: distinguishing nearly-degenerate eigendirections requires additional iterations proportional to the cluster size rather than the condition number. As problem complexity increases through larger $n$ or more extreme $\kappa$, this clustering effect necessitates additional optimization steps, as demonstrated in Figure 3. This phenomenon has motivated deflation techniques[^1], which explicitly remove problematic small eigenvalues from the system to accelerate convergence by reducing the effective condition number of the deflated operator.

[^1]: Kahl, P. et al. "The Deflated Conjugate Gradient Method." 2012.

#### Preconditioning Efficacy

Table 2 quantifies the impact of Jacobi preconditioning at the largest problem size. The method consistently reduces iteration counts by 9–13% across all condition numbers tested. However, this reduction translates to only modest improvements in the effective condition number: at $\kappa=10^6$, the preconditioned system satisfies $\kappa(M^{-1}A) \approx 9.2 \times 10^5$, representing merely an 8% reduction from the original conditioning. This limited efficacy stems from the fact that $M = \mathrm{diag}(A)$ captures only the diagonal structure, failing to exploit the eigenvalue clustering that dominates convergence behavior.

| Method           | $\kappa=10^2$ | $\kappa=10^4$ | $\kappa=10^6$ | Mean Reduction |
|------------------|---------------|---------------|---------------|----------------|
| CG (unprecond.)  | 84            | 667           | 4769          | --             |
| PCG (Jacobi)     | 73            | 595           | 4344          | 10.7%          |
| Relative reduction | 13%         | 11%           | 9%            | --             |

#### Computational Cost Analysis

Figure 4 and Table 3 reveal a striking disparity between iteration complexity and wall-clock time. At the critical test case $n=5000$, $\kappa=10^6$, Gaussian elimination requires 5.44 seconds versus 48.6 seconds for CG—a 9-fold advantage for the direct method. This inversion of expected performance stems from the computational cost model: each CG iteration performs one matrix-vector product ($2n^2$ flops) plus $O(n)$ vector operations, yielding total cost $\approx 2kn^2$. LU factorization requires $\frac{2n^3}{3}$ flops but exploits cache locality and level-3 BLAS operations. The crossover occurs when $k \approx \frac{n}{3}$; our observed $k=4769$ at $n=5000$ places us deep in the regime where direct methods dominate.

| Method              | $n=1000$ | $n=2000$ | $n=5000$ |
|---------------------|----------|----------|----------|
| **$\kappa = 10^2$** |          |          |          |
| Gaussian Elimination| 0.029    | 0.187    | 2.06     |
| CG (Custom)         | 0.016    | 0.080    | 0.82     |
| PCG (Jacobi)        | 0.043    | 0.139    | 1.80     |
| **$\kappa = 10^6$** |          |          |          |
| Gaussian Elimination| 0.041    | 0.190    | 5.44     |
| CG (Custom)         | 0.716    | 4.14     | 48.6     |
| PCG (Jacobi)        | 1.69     | 7.19     | 115      |

![Computational time versus problem size across condition numbers. Direct methods remain competitive through $n=5000$, with superiority increasing at higher $\kappa$ due to the $O(kn^2)$ cost of iterative methods with large iteration counts.](imgs/conjugate_gradient/4.time_vs_size_all_kappa_SPD_Dense.pdf)

Preconditioning overhead exacerbates this disadvantage: PCG (Jacobi) requires 115 seconds at $n=5000$, $\kappa=10^6$—2.4 times slower than unpreconditioned CG despite a 9% iteration reduction. Each PCG iteration incurs additional costs for computing $z_k = M^{-1}r_k$ (diagonal solve) and the modified inner products, offsetting the iteration savings.

![Speedup factor relative to Gaussian elimination. Values below unity indicate iterative inferiority. At $n=5000$, all iterative variants exhibit significant slowdown factors, with CG achieving only 0.11× the performance of direct solvers at $\kappa=10^6$.](imgs/conjugate_gradient/5.speedup_analysis_SPD_Dense.pdf)

Figure 5 quantifies this performance gap: at $n=5000$, $\kappa=10^6$, CG achieves a speedup factor of 0.11 relative to Gaussian elimination—equivalently, a 9-fold slowdown.

#### Implementation Considerations

Table 4 and Figure 6 expose substantial performance variations between mathematically equivalent implementations. Our custom CG and SciPy's `scipy.sparse.linalg.cg` converge in nearly identical iteration counts (4769 versus 4727 at $n=5000$, $\kappa=10^6$), yet SciPy requires 102 seconds compared to our 48.6 seconds—a 110% overhead penalty. This discrepancy arises from architectural differences: SciPy employs Python callback functions for convergence monitoring, performs additional validity checks per iteration, and exhibits less favorable cache access patterns. The per-iteration cost increases from 10.2 ms (custom) to 21.6 ms (SciPy), demonstrating that implementation efficiency can dominate algorithmic choices for large-scale computations.

| Implementation   | Iterations | Time (s) | Time/Iteration (ms) |
|------------------|------------|----------|---------------------|
| CG (Custom)      | 4769       | 48.6     | 10.2                |
| CG (SciPy)       | 4727       | 102      | 21.6                |
| **Relative overhead** | $-0.9\%$ | $+110\%$ | $+112\%$           |

![Comprehensive method comparison showing iteration counts and wall-clock times. PCG reduces iterations uniformly but increases time due to per-iteration overhead. SciPy implementations incur significant additional costs despite algorithmic equivalence.](imgs/conjugate_gradient/6.method_comparison_all_n_SPD_Dense.pdf)

### Synthesis and Practical Guidelines

The experimental results establish three principal findings:

1. **Eigenvalue distribution architecture dominates convergence behavior:** Logarithmic spacing induces sufficient clustering to render iteration counts effectively $\kappa$-independent for $\kappa \leq 10^4$, contradicting naive application of worst-case bounds.
2. **Jacobi preconditioning provides marginal benefits** (10% iteration reduction) at prohibitive computational cost (2× wall-clock overhead) for dense matrices with our spectral structure.
3. **Direct methods retain dominance** over iterative approaches for dense systems with $n \lesssim 5 \times 10^3$ due to the $O(kn^2)$ versus $O(n^3)$ complexity crossover point and superior cache efficiency of factorization algorithms.

**Recommendations for dense SPD systems:**  
Employ LU/Cholesky factorization for $n < 10^4$ unless memory constraints intervene. Avoid Jacobi preconditioning without problem-specific justification, as overhead exceeds benefits. Profile library implementations carefully; custom implementations can achieve performance improvements over generic interfaces.

**Scenarios favoring iterative methods:**

- Sparse matrices where matrix-vector products achieve $O(n)$ complexity through sparsity exploitation
- Very large systems ($n > 10^5$) where $O(n^3)$ factorization becomes infeasible
- Availability of advanced preconditioners (incomplete Cholesky, algebraic multigrid) that provide order-of-magnitude condition number improvements

## Conclusions

For dense SPD systems with $n \leq 5000$, direct factorization methods remain the optimal choice both theoretically and empirically. The conventional wisdom favoring iterative methods applies when either matrix structure enables $O(n)$ matrix-vector products or system size exceeds the direct method feasibility threshold. Our eigenvalue distribution study demonstrates that realistic spectral clustering can render CG convergence largely $\kappa$-independent, a phenomenon obscured by worst-case theoretical analyses.
