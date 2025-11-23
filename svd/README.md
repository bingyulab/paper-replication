# Experiment 

File `old_experiment`, which has been finished last semester, compares various matrix approximation methods including SVD, randomized SVD, CUR decomposition, interpolative decomposition, Nystrom method, and manifold-based approximations. It also includes implementations using libraries such as Scipy and Sklearn for comparison.

Based on the Golub-Kahan (1965) paper:
1. **Bidiagonalization** using Householder transformations
2. **Computing singular values** from the bidiagonal matrix
3. **Computing orthogonal vectors** U and V
4. **Pseudo-inverse applications** for least squares

## Designed Experiments

### **Experiment 1: Bidiagonalization Algorithm Validation**

* **Objective:** To evaluate the performance, accuracy, and numerical stability of the Householder bidiagonalization algorithm (Golub-Kahan) as described in Section 2 of the paper.

* **Methodology:**
    1.  **Golub-Kahan Bidiagonalization (Householder Reflections):** This is the main algorithm presented in the paper. It will be implemented in Python to decompose a matrix $A$ into $U_1 B V^T$, where $B$ is bidiagonal and $U_1, V$ are orthogonal.
    2.  **LAPACK Baseline (via `scipy.linalg.svd`):** A highly optimized, production-grade implementation will be used as the ground truth for accuracy and a benchmark for performance.

* **Test Matrices:**
    * **Sizes:** Square matrices of varying dimensions (e.g., 100x100, 500x500, 1000x1000) to evaluate scalability.
    * **Conditioning:** Matrices with controlled condition numbers (e.g., $10^2, 10^8, 10^{16}$) to test robustness.

* **Metrics:**
    1.  **Wall-clock Time:** Measure the execution time to assess practical performance against the baseline.
    2.  **Reconstruction Accuracy:** Calculate the relative Frobenius norm of the error: $||A - U_1 B V^T||_F / ||A||_F$. This verifies the correctness of the decomposition.
    3.  **Orthogonality:** Measure the stability by calculating $||U_1^T U_1 - I||_F$ and $||V^T V - I||_F$. Errors should be close to machine epsilon.

---

### **Experiment 2: Accuracy of Singular Value Computation**

* **Objective:** To experimentally verify the paper's claim that computing eigenvalues from $J^T J$ is less stable for finding small singular values compared to methods that operate on a related, larger matrix.

* **Hypothesis:** The central hypothesis is that the `J^T J` method will suffer from a significant loss of precision for small singular values due to the squaring of the condition number ($\kappa(J^T J) = \kappa(J)^2$).

* **Methodology:**
    1.  First, bidiagonalize a set of test matrices using the validated Householder method from Experiment 1 to obtain the matrix $J$.
    2.  **Method A (Paper's Recommendation - `J^T J`):** Form the symmetric tridiagonal matrix $K = J^T J$ and compute its eigenvalues ($\lambda_i$). The singular values are $\sigma_i = \sqrt{\lambda_i}$.
    3.  **Method B (Alternative - `2n x 2n` matrix):** Construct the $2n \times 2n$ symmetric matrix described in the paper (Equation 3.2) and compute its positive eigenvalues, which are the singular values of $J$.
    4.  **Baseline:** The singular values computed by `scipy.linalg.svd` on the original matrix $A$ will serve as the ground truth.

* **Test Matrices:**
    * A series of square matrices ($n=100$) with progressively increasing condition numbers, from well-conditioned ($\kappa=10^2$) to extremely ill-conditioned ($\kappa=10^{12}$).

* **Metrics:**
    1.  **Overall Relative Error:** The relative error in the vector of all computed singular values compared to the baseline.
    2.  **Smallest Singular Value Error:** The relative error in the computed $\sigma_{min}$. This is the primary metric for testing the hypothesis.

---

### **Experiment 3: Robustness of the Pseudo-Inverse**

* **Objective:** To demonstrate the superior numerical stability and correctness of the SVD-based pseudo-inverse for solving ill-posed linear least-squares problems, as discussed in Section 5.

* **Methodology:**
    1.  **SVD-Based Pseudo-Inverse (Paper's Method):** Compute $A^+ = V \Sigma^+ U^T$, where $\Sigma^+$ is formed by inverting the non-zero singular values.
    2.  **Normal Equations Method:** Compute the pseudo-inverse via $(A^T A)^{-1} A^T$. This method is expected to fail or produce poor results for ill-conditioned or rank-deficient matrices.
    3.  Solve the least-squares problem $min ||b - Ax||_2$ for a random vector $b$ using the solution $x = A^+ b$.

* **Test Matrices:**
    1.  **Well-Conditioned:** A full-rank matrix with a low condition number.
    2.  **Ill-Conditioned:** A full-rank matrix with a high condition number ($>10^8$).
    3.  **Exactly Rank-Deficient:** A matrix where the rank is explicitly less than the number of columns.

* **Metrics:**
    1.  **Residual Norm ($||b - Ax||_2$):** Measures how well the solution fits the data. A smaller residual is better.
    2.  **Solution Norm ($||x||_2$):** The SVD method is guaranteed to find the solution with the minimum possible norm. This metric will verify that property.
    3.  **Success/Failure:** Note whether the Normal Equations method fails due to a singular $A^T A$ matrix.

---

### **Experiment 4: Benchmarking Against Randomized SVD**

* **Objective:** To compare the classical Golub-Kahan SVD against a modern, probabilistic alternative (Randomized SVD) to understand the trade-offs between accuracy, speed, and matrix structure.

* **Methodology:**
    * Implement Randomized SVD (rSVD) and compare its output against the full SVD from `scipy.linalg.svd`. The key parameters of rSVD—**oversampling (`p`)** and **power iterations (`q`)**—will be varied systematically.

* **Sub-Experiments:**
    1.  **Parameter Sensitivity:** For matrices with both fast and slow singular value decay, sweep through a range of `p` (e.g., 0, 5, 10, 20) and `q` (e.g., 0, 1, 2, 3) values. This will show how these parameters tune accuracy.
    2.  **Performance vs. Accuracy:** For a fixed, high accuracy target, measure the wall-clock time for both full SVD and rSVD on large matrices to determine the speedup factor.
    3.  **Conditioning and Power Iterations:** For a set of matrices with increasing condition numbers, quantify how many power iterations (`q`) are necessary for rSVD to achieve an acceptable error.

* **Metrics:**
    1.  **Approximation Error:** Relative Frobenius norm error $||A - A_k||_F / ||A||_F$, where $A_k$ is the rank-k approximation.
    2.  **Computation Time & Speedup:** Measure wall-clock time and calculate the ratio `Time(SVD) / Time(rSVD)`.
    3.  **Singular Value Accuracy:** Compare the leading singular values found by rSVD to the true values.
