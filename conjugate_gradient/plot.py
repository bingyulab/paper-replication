import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import os, json
import numpy as np

# Add at the beginning of your plotting section:
def set_science_style():
    import scienceplots
    plt.style.use(['science', 'nature'])  # Or 'nature', 'science', 'high-vis'

set_science_style()

# Set global matplotlib parameters for better PDF output
matplotlib.rcParams.update({
    'font.size': 11,           # Base font size
    'axes.labelsize': 12,      # Axis label font size
    'axes.titlesize': 13,      # Title font size
    'xtick.labelsize': 10,     # X-tick label size
    'ytick.labelsize': 10,     # Y-tick label size
    'legend.fontsize': 9,      # Legend font size
    'figure.titlesize': 14,    # Figure title size
    'lines.linewidth': 1.5,    # Line width
    'lines.markersize': 4,     # Marker size
    'figure.dpi': 150,         # Figure DPI for display
    'savefig.dpi': 300,        # Save DPI (high quality)
    'savefig.format': 'pdf',   # Default save format
    'pdf.fonttype': 42,        # TrueType fonts in PDF (better compatibility)
    'ps.fonttype': 42,
    'text.usetex': False,      # Set to True if you have LaTeX installed
})

# Define consistent color scheme and line styles
linestyles = {
    'Steepest Descent': {'linestyle': '-', 'color': 'C1', 'marker': 'o'},
    'CG (Custom)': {'linestyle': '-', 'color': 'C2', 'marker': 's'},
    'CG (SciPy)': {'linestyle': '--', 'color': 'C3', 'marker': '^'},
    'PCG (Jacobi)': {'linestyle': '-.', 'color': 'C4', 'marker': 'd'},
    'PCG (Jacobi, SciPy)': {'linestyle': ':', 'color': 'C5', 'marker': 'v'},
    'PCG (Inverse)': {'linestyle': '-', 'color': 'C6', 'marker': 'p'},
    'PCG (Inverse, SciPy)': {'linestyle': '--', 'color': 'C7', 'marker': 'h'},
    'Gaussian Elimination': {'linestyle': '-', 'color': 'C0', 'marker': 'o'},
}

methods_all = ['Steepest Descent', 'CG (Custom)', 'CG (SciPy)', 
               'PCG (Jacobi)', 'PCG (Jacobi, SciPy)', 'PCG (Inverse)',
               'Gaussian Elimination']

# Create output directory
os.makedirs('imgs/conjugate_gradient', exist_ok=True)

# Load Data
df_results = pd.read_csv('imgs/conjugate_gradient/cg_results.csv')
with open('imgs/conjugate_gradient/cg_convergence.json', 'r') as f:
    raw_data = json.load(f)

# Convert keys and values to correct types
convergence_data = {}
for key_str, methods in raw_data.items():
    # Key format: "[n, kappa, matrix_type]"
    n, kappa, matrix_type = eval(key_str)
    n = int(n)
    kappa = float(kappa)
    key = (n, kappa, matrix_type)
    convergence_data[key] = {}
    for method, res_history in methods.items():
        convergence_data[key][method] = np.array(res_history, dtype=float)

# Focus on Dense only
problem_sizes = [200, 1000, 2000, 5000]
condition_numbers = [1e2, 1e4, 1e6]
matrix_types = ['SPD Dense']  # removed 'Banded'
tol = 1e-8

# Plot 1: Convergence histories - Fixed problem size, vary condition number (Dense only)
for n in problem_sizes:
    fig, axes = plt.subplots(1, len(condition_numbers), figsize=(18, 5))
    fig.suptitle(f'Residual Convergence: SPD Dense, $n={n}$', fontweight='bold')

    for idx, kappa in enumerate(condition_numbers):
        ax = axes[idx]
        key = (n, kappa, 'SPD Dense')

        if key in convergence_data:
            for method, res_history in convergence_data[key].items():
                if len(res_history) > 1:
                    res_normalized = res_history / res_history[0]
                    style = linestyles.get(method, {'linestyle': '-', 'color': None, 'marker': 'o'})
                    ax.semilogy(range(len(res_normalized)), res_normalized,
                               marker=style['marker'], markersize=3, label=method,
                               linewidth=1.5, alpha=0.9,
                               linestyle=style['linestyle'],
                               color=style['color'])
            ax.axhline(y=tol, color='red', linestyle='--', linewidth=1.2, label=f'Tolerance={tol}')
            ax.set_xlabel('Iteration')
            ax.set_ylabel(r'Relative Residual $\|r\|/\|r_0\|$')
            ax.set_title(f'$\\kappa = {kappa:.0e}$')
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8, loc='best', framealpha=0.9)

    plt.tight_layout()
    filename = f'imgs/conjugate_gradient/1.convergence_SPD_Dense_n{n}.pdf'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Saved: {filename}")
    plt.close()

# Plot 2: Convergence histories - Fixed condition number, vary problem size (Dense only)
for kappa in condition_numbers:
    fig, axes = plt.subplots(1, len(problem_sizes), figsize=(18, 5))
    fig.suptitle(f'Residual Convergence: SPD Dense, $\\kappa={kappa:.0e}$', fontweight='bold')

    for idx, n in enumerate(problem_sizes):
        ax = axes[idx]
        key = (n, kappa, 'SPD Dense')

        if key in convergence_data:
            for method, res_history in convergence_data[key].items():
                if len(res_history) > 1:
                    res_normalized = res_history / res_history[0]
                    style = linestyles.get(method, {'linestyle': '-', 'color': None, 'marker': 'o'})
                    ax.semilogy(range(len(res_normalized)), res_normalized,
                               marker=style['marker'], markersize=3, label=method,
                               linewidth=1.5, alpha=0.9,
                               linestyle=style['linestyle'],
                               color=style['color'])
            ax.axhline(y=tol, color='red', linestyle='--', linewidth=1.2, label=f'Tolerance={tol}')
            ax.set_xlabel('Iteration')
            ax.set_ylabel(r'Relative Residual $\|r\|/\|r_0\|$')
            ax.set_title(f'$n = {n}$')
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8, loc='best', framealpha=0.9)

    plt.tight_layout()
    filename = f'imgs/conjugate_gradient/2.convergence_SPD_Dense_kappa{kappa:.0e}.pdf'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

# Plot 3: Iterations comparison - merge all problem_sizes into one figure (Dense only)
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Iterations vs Condition Number - SPD Dense', fontweight='bold')
axes = axes.flatten()

for idx, n in enumerate(problem_sizes):
    ax = axes[idx]
    df_subset = df_results[(df_results['problem_size'] == n) &
                           (df_results['matrix_type'] == 'SPD Dense') &
                           (df_results['iterations'] > 0)]

    for method in methods_all:
        df_method = df_subset[df_subset['method'] == method]
        if not df_method.empty:
            style = linestyles.get(method, {'linestyle': '-', 'color': None, 'marker': 'o'})
            ax.plot(df_method['condition_number'], df_method['iterations'],
                    marker=style['marker'], markersize=6, label=method, linewidth=1.8,
                    linestyle=style['linestyle'], color=style['color'])
    ax.set_xlabel(r'Condition Number ($\kappa$)')
    ax.set_ylabel('Iterations')
    ax.set_title(f'$n = {n}$')
    ax.set_xscale('log')
    ax.legend(fontsize=8, framealpha=0.9)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
filename = f'imgs/conjugate_gradient/3.iterations_vs_kappa_all_n_SPD_Dense.pdf'
plt.savefig(filename, dpi=300, bbox_inches='tight')
print(f"Saved: {filename}")
plt.close()

# Plot 4: Time comparison - merge all condition_numbers into one (Dense only)
fig, axes = plt.subplots(1, len(condition_numbers), figsize=(18, 5))
fig.suptitle('Computation Time vs Problem Size - SPD Dense', fontweight='bold')

for idx, kappa in enumerate(condition_numbers):
    ax = axes[idx]
    df_subset = df_results[(df_results['condition_number'] == kappa) &
                           (df_results['matrix_type'] == 'SPD Dense')]

    for method in methods_all:
        df_method = df_subset[df_subset['method'] == method]
        if not df_method.empty:
            style = linestyles.get(method, {'linestyle': '-', 'color': None, 'marker': 'o'})
            ax.plot(df_method['problem_size'], df_method['time'],
                    marker=style['marker'], markersize=6, label=method, linewidth=1.8,
                    linestyle=style['linestyle'], color=style['color'])
    ax.set_xlabel('Problem Size ($n$)')
    ax.set_ylabel('Time (seconds)')
    ax.set_title(f'$\\kappa={kappa:.0e}$')
    ax.set_yscale('log')
    ax.legend(fontsize=8, framealpha=0.9)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
filename = f'imgs/conjugate_gradient/4.time_vs_size_all_kappa_SPD_Dense.pdf'
plt.savefig(filename, dpi=300, bbox_inches='tight')
print(f"Saved: {filename}")
plt.close()

# Plot 5: Speedup over Gaussian Elimination (Dense only)
# Note: this plot can be hard to interpret; keep it but limited to SPD Dense.
fig, axes = plt.subplots(2, 1, figsize=(10, 8))
fig.suptitle('Speedup Factor over Gaussian Elimination - SPD Dense', fontweight='bold')

for i, n in enumerate([problem_sizes[0], problem_sizes[-1]]):
    ax = axes[i]
    df_ge = df_results[(df_results['problem_size'] == n) &
                       (df_results['matrix_type'] == 'SPD Dense') &
                       (df_results['method'] == 'Gaussian Elimination')]

    for method in ['CG (Custom)', 'PCG (Jacobi)', 'CG (SciPy)', 'PCG (Jacobi, SciPy)', 'PCG (Inverse)']:
        df_method = df_results[(df_results['problem_size'] == n) &
                               (df_results['matrix_type'] == 'SPD Dense') &
                               (df_results['method'] == method)]
        if not df_ge.empty and not df_method.empty:
            # Align by condition_number
            common_kappa = np.intersect1d(df_ge['condition_number'].values, df_method['condition_number'].values)
            if common_kappa.size == 0:
                continue
            ge_times, meth_times, kappas = [], [], []
            for k in common_kappa:
                ge_t = df_ge[df_ge['condition_number'] == k]['time'].values
                m_t = df_method[df_method['condition_number'] == k]['time'].values
                if ge_t.size and m_t.size:
                    ge_times.append(ge_t[0])
                    meth_times.append(m_t[0])
                    kappas.append(k)
            if len(kappas) > 0:
                style = linestyles.get(method, {'linestyle': '-', 'color': None, 'marker': 'o'})
                ax.plot(kappas, np.array(ge_times) / np.array(meth_times),
                        marker=style['marker'], markersize=6, label=method, linewidth=1.8,
                        linestyle=style['linestyle'], color=style['color'])
    ax.axhline(y=1, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='No speedup')
    ax.set_xlabel(r'Condition Number ($\kappa$)')
    ax.set_ylabel('Speedup Factor')
    ax.set_title(f'$n={n}$')
    ax.set_xscale('log')
    ax.legend(fontsize=8, framealpha=0.9)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('imgs/conjugate_gradient/5.speedup_analysis_SPD_Dense.pdf', 
            dpi=300, bbox_inches='tight')
print(f"Saved: imgs/conjugate_gradient/5.speedup_analysis_SPD_Dense.pdf")
plt.close()

# Plot 6: Method comparison summary (merge all problem_sizes into one figure, Dense only)
methods_to_compare = ['CG (Custom)', 'CG (SciPy)', 'PCG (Jacobi)', 'PCG (Jacobi, SciPy)']
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Method Comparison Summary - SPD Dense', fontweight='bold')

for i, n in enumerate([problem_sizes[0], problem_sizes[-1]]):
    # Iterations - Dense
    ax_iter = axes[i, 0] if len(problem_sizes) > 1 else axes[0]
    df_iter = df_results[(df_results['problem_size'] == n) &
                         (df_results['matrix_type'] == 'SPD Dense') &
                         (df_results['method'].isin(methods_to_compare))]
    # grouped bar positions
    kappas = sorted(df_iter['condition_number'].unique())
    x = np.arange(len(kappas))
    width = 0.2
    for j, method in enumerate(methods_to_compare):
        df_m = df_iter[df_iter['method'] == method].set_index('condition_number').reindex(kappas)
        vals = df_m['iterations'].fillna(0).values
        ax_iter.bar(x + (j - 1.5) * width, vals, width=width, alpha=0.8, label=method)
    ax_iter.set_xticklabels([f"$\\kappa={k:.0e}$" for k in kappas])
    ax_iter.set_ylabel('Iterations')
    ax_iter.set_title(f'Iterations ($n={n}$)')
    ax_iter.legend(fontsize=8, framealpha=0.9)
    ax_iter.grid(True, alpha=0.3, axis='y')

    # Time - Dense
    ax_time = axes[i, 1]
    df_time = df_results[(df_results['problem_size'] == n) &
                         (df_results['matrix_type'] == 'SPD Dense') &
                         (df_results['method'].isin(methods_to_compare))]
    kappas_t = sorted(df_time['condition_number'].unique())
    x_t = np.arange(len(kappas_t))
    for j, method in enumerate(methods_to_compare):
        df_m = df_time[df_time['method'] == method].set_index('condition_number').reindex(kappas_t)
        vals = df_m['time'].fillna(0).values
        ax_time.bar(x_t + (j - 1.5) * width, vals, width=width, alpha=0.8, label=method)
    ax_time.set_xticks(x_t)
    ax_time.set_xticklabels([f"$\\kappa={k:.0e}$" for k in kappas_t])
    ax_time.set_ylabel('Time (seconds)')
    ax_time.set_title(f'Time ($n={n}$)')
    ax_time.legend(fontsize=8, framealpha=0.9)
    ax_time.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
filename = f'imgs/conjugate_gradient/6.method_comparison_all_n_SPD_Dense.pdf'
plt.savefig(filename, dpi=300, bbox_inches='tight')
print(f"Saved: {filename}")
plt.close()

# Nonlinear results (Plot 7) - keep behavior: subplot 3 has solid + dashed lines per method
non_linear = pd.read_csv('imgs/conjugate_gradient/nonlinear_results.csv')

methods_nl = non_linear['method'].unique().tolist()
colors = ['C0', 'C1']

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Non-linear CG Methods Comparison', fontweight='bold')

# Subplot 1: Iterations vs Dimension
ax = axes[0]
for i, method in enumerate(methods_nl):
    dfm = non_linear[non_linear['method'] == method].sort_values('dimension')
    ax.plot(dfm['dimension'], dfm['iterations'], marker='o', linewidth=2, markersize=6,
            label=method, color=colors[i % len(colors)])
ax.set_xlabel('Dimension')
ax.set_ylabel('Iterations')
ax.set_title('Iterations vs Dimension')
ax.grid(True, alpha=0.3)
ax.legend(fontsize=9, framealpha=0.9)

# Subplot 2: Time vs Dimension
ax = axes[1]
for i, method in enumerate(methods_nl):
    dfm = non_linear[non_linear['method'] == method].sort_values('dimension')
    ax.plot(dfm['dimension'], dfm['time'], marker='o', linewidth=2, markersize=6,
            label=method, color=colors[i % len(colors)])
ax.set_xlabel('Dimension')
ax.set_ylabel('Time (seconds)')
ax.set_title('Computation Time vs Dimension')
ax.grid(True, alpha=0.3)
ax.legend(fontsize=9, framealpha=0.9)

# Subplot 3: Final gradient norm (and avg per iteration) vs Dimension
ax = axes[2]
for i, method in enumerate(methods_nl):
    dfm = non_linear[non_linear['method'] == method].sort_values('dimension')
    ax.plot(dfm['dimension'], dfm['final_gradient_norm'], marker='o', linewidth=2, markersize=6,
            label=method, color=colors[i % len(colors)])
ax.set_xlabel('Dimension')
ax.set_ylabel(r'Final Gradient Norm $\|\nabla f\|$')
ax.set_yscale('log')
ax.set_title('Final Gradient Norm (log scale)')
ax.grid(True, alpha=0.3)

# twin axis: average final gradient norm per iteration (approximate convergence rate) - dashed lines
ax2 = ax.twinx()
for i, method in enumerate(methods_nl):
    dfm = non_linear[non_linear['method'] == method].sort_values('dimension')
    avg_per_iter = dfm['final_gradient_norm'] / dfm['iterations'].replace(0, 1)
    ax2.plot(dfm['dimension'], avg_per_iter, marker='x', linestyle='--',
             linewidth=1.5, color=colors[i % len(colors)], alpha=0.7)
ax2.set_yscale('log')
ax2.set_ylabel(r'Gradient Norm per Iteration', fontsize=10)
ax.legend(fontsize=9, loc='upper left', framealpha=0.9)

plt.tight_layout()
filename = 'imgs/conjugate_gradient/7.nonlinear_comparison.pdf'
plt.savefig(filename, dpi=300, bbox_inches='tight')
print(f"Saved: {filename}")
plt.close()

print("\n" + "="*80)
print("All plots saved to: imgs/conjugate_gradient/")
print("="*80)