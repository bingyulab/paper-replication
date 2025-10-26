# Conjugate Gradient Methods

There are 5 methods:

## B1. Steepest Descent

Given the inputs $A$, $b$, a starting value $x$, a maximum number of iterations $i_{\max}$, and an error tolerance $\varepsilon < 1$:

$$
\begin{aligned}
& i \Leftarrow 0 \\
& r \Leftarrow b - Ax \\
& \delta \Leftarrow r^T r \\
& \delta_0 \Leftarrow \delta \\
& \text{While } i < i_{\max} \text{ and } \delta > \varepsilon^2 \delta_0 \text{ do} \\
& \quad q \Leftarrow Ar \\
& \quad \alpha \Leftarrow \frac{\delta}{r^T q} \\
& \quad x \Leftarrow x + \alpha r \\
& \quad \text{If } i \text{ is divisible by } 50 \\
& \quad\quad r \Leftarrow b - Ax \\
& \quad \text{else} \\
& \quad\quad r \Leftarrow r - \alpha q \\
& \quad \delta \Leftarrow r^T r \\
& \quad i \Leftarrow i + 1
\end{aligned}
$$

This algorithm terminates when the maximum number of iterations $i_{\max}$ has been exceeded, or when $\|r_{(i)}\| \leq \varepsilon\|r_{(0)}\|$.

## B2. Conjugate Gradients

Given the inputs $A$, $b$, a starting value $x$, a maximum number of iterations $i_{\max}$, and an error tolerance $\varepsilon < 1$:

$$
\begin{aligned}
& i \Leftarrow 0 \\
& r \Leftarrow b - Ax \\
& d \Leftarrow r \\
& \delta_{\text{new}} \Leftarrow r^T r \\
& \delta_0 \Leftarrow \delta_{\text{new}} \\
& \text{While } i < i_{\max} \text{ and } \delta_{\text{new}} > \varepsilon^2 \delta_0 \text{ do} \\
& \quad q \Leftarrow Ad \\
& \quad \alpha \Leftarrow \frac{\delta_{\text{new}}}{d^T q} \\
& \quad x \Leftarrow x + \alpha d \\
& \quad \text{If } i \text{ is divisible by } 50 \\
& \quad\quad r \Leftarrow b - Ax \\
& \quad \text{else} \\
& \quad\quad r \Leftarrow r - \alpha q \\
& \quad \delta_{\text{old}} \Leftarrow \delta_{\text{new}} \\
& \quad \delta_{\text{new}} \Leftarrow r^T r \\
& \quad \beta \Leftarrow \frac{\delta_{\text{new}}}{\delta_{\text{old}}} \\
& \quad d \Leftarrow r + \beta d \\
& \quad i \Leftarrow i + 1
\end{aligned}
$$

## B3. Preconditioned Conjugate Gradients

Given the inputs $A$, $b$, a starting value $x$, a (perhaps implicitly defined) preconditioner $M$, a maximum number of iterations $i_{\max}$, and an error tolerance $\varepsilon < 1$:

$$
\begin{aligned}
& i \Leftarrow 0 \\
& r \Leftarrow b - Ax \\
& d \Leftarrow M^{-1} r \\
& \delta_{\text{new}} \Leftarrow r^T d \\
& \delta_0 \Leftarrow \delta_{\text{new}} \\
& \text{While } i < i_{\max} \text{ and } \delta_{\text{new}} > \varepsilon^2 \delta_0 \text{ do} \\
& \quad q \Leftarrow Ad \\
& \quad \alpha \Leftarrow \frac{\delta_{\text{new}}}{d^T q} \\
& \quad x \Leftarrow x + \alpha d \\
& \quad \text{If } i \text{ is divisible by } 50 \\
& \quad\quad r \Leftarrow b - Ax \\
& \quad \text{else} \\
& \quad\quad r \Leftarrow r - \alpha q \\
& \quad s \Leftarrow M^{-1} r \\
& \quad \delta_{\text{old}} \Leftarrow \delta_{\text{new}} \\
& \quad \delta_{\text{new}} \Leftarrow r^T s \\
& \quad \beta \Leftarrow \frac{\delta_{\text{new}}}{\delta_{\text{old}}} \\
& \quad d \Leftarrow s + \beta d \\
& \quad i \Leftarrow i + 1
\end{aligned}
$$

The statement "$s \Leftarrow M^{-1} r$" implies that one should apply the preconditioner, which may not actually be in the form of a matrix.

## B4. Nonlinear Conjugate Gradients with Newton-Raphson and Fletcher-Reeves

Given a function $f$, a starting value $x$, a maximum number of CG iterations $i_{\max}$, a CG error tolerance $\varepsilon < 1$, a maximum number of Newton-Raphson iterations $j_{\max}$, and a Newton-Raphson error tolerance $\epsilon < 1$:

$$
\begin{aligned}
& i \Leftarrow 0 \\
& k \Leftarrow 0 \\
& r \Leftarrow -f'(x) \\
& d \Leftarrow r \\
& \delta_{\text{new}} \Leftarrow r^T r \\
& \delta_0 \Leftarrow \delta_{\text{new}} \\
& \text{While } i < i_{\max} \text{ and } \delta_{\text{new}} > \varepsilon^2 \delta_0 \text{ do} \\
& \quad j \Leftarrow 0 \\
& \quad \delta_d \Leftarrow d^T d \\
& \quad \text{Do} \\
& \quad\quad \alpha \Leftarrow -\frac{[f'(x)]^T d}{d^T f''(x) d} \\
& \quad\quad x \Leftarrow x + \alpha d \\
& \quad\quad j \Leftarrow j + 1 \\
& \quad \text{while } j < j_{\max} \text{ and } \alpha^2 \delta_d > \epsilon^2 \\
& \quad r \Leftarrow -f'(x) \\
& \quad \delta_{\text{old}} \Leftarrow \delta_{\text{new}} \\
& \quad \delta_{\text{new}} \Leftarrow r^T r \\
& \quad \beta \Leftarrow \frac{\delta_{\text{new}}}{\delta_{\text{old}}} \\
& \quad d \Leftarrow r + \beta d \\
& \quad k \Leftarrow k + 1 \\
& \quad \text{If } k = n \text{ or } r^T d \leq 0 \\
& \quad\quad d \Leftarrow r \\
& \quad\quad k \Leftarrow 0 \\
& \quad i \Leftarrow i + 1
\end{aligned}
$$

This algorithm terminates when the maximum number of iterations $i_{\max}$ has been exceeded, or when $\|r_{(i)}\| \leq \varepsilon\|r_{(0)}\|$.

## B5. Preconditioned Nonlinear Conjugate Gradients with Secant and Polak-Ribière

Given a function $f$, a starting value $x$, a maximum number of CG iterations $i_{\max}$, a CG error tolerance $\varepsilon < 1$, a Secant method step parameter $\sigma_0$, a maximum number of Secant method iterations $j_{\max}$, and a Secant method error tolerance $\epsilon < 1$:

$$
\begin{aligned}
& i \Leftarrow 0 \\
& k \Leftarrow 0 \\
& r \Leftarrow -f'(x) \\
& \text{Calculate a preconditioner } M \approx f''(x) \\
& s \Leftarrow M^{-1} r \\
& d \Leftarrow s \\
& \delta_{\text{new}} \Leftarrow r^T d \\
& \delta_0 \Leftarrow \delta_{\text{new}} \\
& \text{While } i < i_{\max} \text{ and } \delta_{\text{new}} > \varepsilon^2 \delta_0 \text{ do} \\
& \quad j \Leftarrow 0 \\
& \quad \delta_d \Leftarrow d^T d \\
& \quad \alpha \Leftarrow -\sigma_0 \\
& \quad \eta_{\text{prev}} \Leftarrow [f'(x + \sigma_0 d)]^T d \\
& \quad \text{Do} \\
& \quad\quad \eta \Leftarrow [f'(x)]^T d \\
& \quad\quad \alpha \Leftarrow \alpha \frac{\eta}{\eta_{\text{prev}} - \eta} \\
& \quad\quad x \Leftarrow x + \alpha d \\
& \quad\quad \eta_{\text{prev}} \Leftarrow \eta \\
& \quad\quad j \Leftarrow j + 1 \\
& \quad \text{while } j < j_{\max} \text{ and } \alpha^2 \delta_d > \epsilon^2 \\
& \quad r \Leftarrow -f'(x) \\
& \quad \delta_{\text{mid}} \Leftarrow \delta_{\text{new}} \\
& \quad \delta_{\text{old}} \Leftarrow \delta_{\text{new}} \\
& \quad \text{Calculate a preconditioner } M \approx f''(x) \\
& \quad s \Leftarrow M^{-1} r \\
& \quad \delta_{\text{new}} \Leftarrow r^T s \\
& \quad \beta \Leftarrow \frac{\delta_{\text{new}} - \delta_{\text{mid}}}{\delta_{\text{old}}} \\
& \quad k \Leftarrow k + 1 \\
& \quad \text{If } k = n \text{ or } \beta \leq 0 \\
& \quad\quad d \Leftarrow s \\
& \quad\quad k \Leftarrow 0 \\
& \quad \text{else} \\
& \quad\quad d \Leftarrow s + \beta d \\
& \quad i \Leftarrow i + 1
\end{aligned}
$$

This algorithm terminates when the maximum number of iterations $i_{\max}$ has been exceeded, or when $\|r_{(i)}\| \leq \varepsilon\|r_{(0)}\|$.

Again, more details about algorithms can be found in the document [conjugate_gradient](doc/conjugate_gradient.md).
