"""Internal orthogonal regression for pointwise, scalar-valued models."""

from dataclasses import dataclass
import inspect

import numpy as np
from scipy.optimize import least_squares, minimize
from scipy.sparse import lil_matrix, issparse


@dataclass
class EIVResult:
    params: np.ndarray
    cov: np.ndarray
    x_adjusted: np.ndarray
    pulls: np.ndarray
    effective_y_err: np.ndarray


def _initial_parameters(func, p0, bounds):
    if p0 is None:
        try:
            positional = [
                p for p in inspect.signature(func).parameters.values()
                if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
            ]
            count = len(positional) - 1
        except (TypeError, ValueError) as exc:
            raise ValueError("Bitte p0 für den EIV-Fit explizit angeben.") from exc
        if count < 1:
            raise ValueError("Bitte p0 für den EIV-Fit explizit angeben.")
    else:
        p0 = np.atleast_1d(np.asarray(p0, dtype=float))
        if p0.ndim != 1 or p0.size == 0:
            raise ValueError("p0 muss ein eindimensionales Parameterarray sein.")
        count = p0.size
    try:
        lower, upper = (bounds.lb, bounds.ub) if hasattr(bounds, "lb") else bounds
        lo = np.broadcast_to(np.asarray(lower, dtype=float), (count,)).copy()
        hi = np.broadcast_to(np.asarray(upper, dtype=float), (count,)).copy()
    except (TypeError, ValueError) as exc:
        raise ValueError("bounds muss zur Anzahl der Fitparameter passen.") from exc
    if np.any(np.isnan(lo) | np.isnan(hi)) or np.any(lo >= hi):
        raise ValueError("Untere Parametergrenzen müssen kleiner als obere sein.")
    if p0 is None:
        p0 = np.ones(count)
        finite_lo, finite_hi = np.isfinite(lo), np.isfinite(hi)
        both = finite_lo & finite_hi
        p0[both] = lo[both]/2 + hi[both]/2
        p0[finite_lo & ~finite_hi] = lo[finite_lo & ~finite_hi] + 1
        p0[~finite_lo & finite_hi] = hi[~finite_lo & finite_hi] - 1
    if not np.all(np.isfinite(p0)) or np.any(p0 < lo) or np.any(p0 > hi):
        raise ValueError("p0 muss endlich und innerhalb der Parametergrenzen sein.")
    return p0, lo, hi


def _values(func, x, p):
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        return np.broadcast_to(np.asarray(func(x, *p), dtype=float), x.shape)


def _x_derivative(func, x, p):
    """Pointwise central differences, with one-sided fallback at domain edges."""
    scale = np.max(np.abs(x)) or 1.0
    h = np.cbrt(np.finfo(float).eps) * np.maximum(np.abs(x), scale*1e-6)
    center = _values(func, x, p)
    plus, minus = _values(func, x+h, p), _values(func, x-h, p)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(np.isfinite(plus) & np.isfinite(minus),
                        (plus-minus)/(2*h),
                        np.where(np.isfinite(plus), (plus-center)/h, (center-minus)/h))


def _parameter_derivatives(func, x, p, lo, hi):
    center = _values(func, x, p)
    columns = []
    for i in range(len(p)):
        h = np.cbrt(np.finfo(float).eps) * max(abs(p[i]), 1e-6)
        left, right = max(lo[i], p[i]-h), min(hi[i], p[i]+h)
        a, b = p.copy(), p.copy()
        a[i], b[i] = left, right
        if left == right:
            raise ValueError("Numerisch nicht unterscheidbare Parametergrenzen.")
        fa, fb = _values(func, x, a), _values(func, x, b)
        with np.errstate(invalid="ignore", divide="ignore"):
            col = (fb-fa)/(right-left)
            if right > p[i]:
                col = np.where(np.isfinite(col), col, (fb-center)/(right-p[i]))
            if left < p[i]:
                col = np.where(np.isfinite(col), col, (center-fa)/(p[i]-left))
        columns.append(col)
    return np.column_stack(columns)


def _covariance(design):
    """Unscaled local covariance from a nuisance-projected Jacobian."""
    if not np.isfinite(design).all():
        raise ValueError("Die EIV-Kovarianz ist nicht endlich.")
    _, s, vt = np.linalg.svd(design, full_matrices=False)
    if len(s) != design.shape[1] or s[-1] <= s[0]*max(design.shape)*np.finfo(float).eps:
        raise ValueError("Die Fitparameter sind lokal nicht eindeutig bestimmbar.")
    cov = (vt.T / s**2) @ vt
    if not np.isfinite(cov).all():
        raise ValueError("Die EIV-Kovarianz ist nicht endlich.")
    return cov


def fit_eiv(func, x, y, dx, dy, p0, bounds):
    """Optimize normalized horizontal shifts along with model parameters.

    For independent Gaussian coordinate uncertainties, this is orthogonal
    distance regression, not the effective-variance approximation. For
    numerical efficiency the Jacobian has one independent nuisance column
    per movable measurement. Only the parameter block of its covariance
    is retained, via the Schur complement.
    """
    p0, lo, hi = _initial_parameters(func, p0, bounds)
    n, m = len(x), len(p0)
    if n < m:
        raise ValueError("Zu wenige Messpunkte für die Anzahl der Fitparameter.")
    free = np.flatnonzero(dx > 0)
    k = len(free)
    if dy is None and k != n:
        raise ValueError("Ohne y-Fehler müssen alle x-Fehler strikt positiv sein.")
    z0 = np.r_[p0, np.zeros(k)]
    lower, upper = np.r_[lo, np.full(k, -np.inf)], np.r_[hi, np.full(k, np.inf)]

    def adjusted(z):
        coords = x.copy()
        coords[free] += dx[free] * z[m:]
        return coords

    if dy is not None:
        def residuals(z):
            return np.r_[z[m:], (y-_values(func, adjusted(z), z[:m]))/dy]

        # For large datasets group the independent horizontal derivatives
        # using sparsity; small datasets use the exact dense trust-region step.
        options = {}
        if n > 80:
            sparsity = lil_matrix((k+n, m+k), dtype=bool)
            sparsity[k:, :m] = True
            idx = np.arange(k)
            sparsity[idx, m+idx] = True
            sparsity[k+free, m+idx] = True
            options["jac_sparsity"] = sparsity.tocsr()
        opt = least_squares(
            residuals, z0, bounds=(lower, upper), x_scale="jac",
            ftol=1e-11, xtol=1e-11, gtol=1e-11, max_nfev=2000, **options,
        )
        if not opt.success or not np.all(np.isfinite(opt.x)):
            raise ValueError("Errors-in-Variables-Fit ist nicht konvergiert: "+opt.message)
        p = opt.x[:m]
        X = adjusted(opt.x)
        # J = [0 I; A B]. B has a single nonzero in each movable row.
        # Eliminating horizontal shifts yields A^T (I+B B^T)^-1 A.
        J = opt.jac
        A = J[k:, :m]
        if issparse(A):
            A = A.toarray()
        B = np.asarray(J[k:, m:][free, :].diagonal()).ravel()
        weights = np.ones(n)
        weights[free] = 1.0 / np.hypot(1.0, B)
        cov = _covariance(A * weights[:, None])
        distances = np.abs(opt.fun[k:]).copy()
        distances[free] = np.hypot(distances[free], opt.x[m:])
    else:
        # Exact y: solve the equality-constrained orthogonal problem.
        # No arbitrary fictitious y-error or implicit unit-dependent weight.
        y_scale = np.max(np.abs(y)) or 1.0

        def objective(z):
            return 0.5 * np.dot(z[m:], z[m:])

        def gradient(z):
            return np.r_[np.zeros(m), z[m:]]

        def constraint(z):
            return (_values(func, adjusted(z), z[:m])-y)/y_scale

        opt = minimize(objective, z0, jac=gradient, method="SLSQP",
                       bounds=list(zip(lower, upper)),
                       constraints={"type": "eq", "fun": constraint},
                       options={"ftol": 1e-11, "maxiter": 1000})
        if not opt.success or not np.all(np.isfinite(opt.x)) or np.max(np.abs(constraint(opt.x))) > 1e-8:
            raise ValueError("EIV-Fit mit exaktem y nicht konvergiert: "+opt.message)
        p, X = opt.x[:m], adjusted(opt.x)
        derivative = _x_derivative(func, X, p)
        if not np.isfinite(derivative).all() or np.any(derivative == 0):
            raise ValueError("Bei exaktem y muss das Modell lokal regulär invertierbar sein.")
        A = _parameter_derivatives(func, X, p, lo, hi)
        cov = _covariance(A / (derivative*dx)[:, None])
        distances = np.abs(opt.x[m:])

    vertical = y-_values(func, x, p)
    # The sign is a display convention relative to the curve at measured x.
    # A zero vertical residual uses +1 so the objective is never lost.
    pulls = np.where(vertical < 0, -distances, distances)
    derivative = _x_derivative(func, x, p)
    propagated_x = np.zeros_like(x)
    propagated_x[free] = derivative[free]*dx[free]
    effective = np.hypot(0.0 if dy is None else dy, propagated_x)
    # The exact orthogonal objective and pulls remain available even when
    # a local vertical derivative cannot be evaluated for plotting.
    return EIVResult(p, cov, X, pulls, effective)
