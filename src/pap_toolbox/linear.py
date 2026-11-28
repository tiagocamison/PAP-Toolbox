"""Convenience helpers for linear PAP fits and their intersections.

The fitted model is y = slope*x + intercept. Fits without x errors delegate
to pap_toolbox.fit_curve. With x errors, independent Gaussian uncertainties
in both coordinates are handled by an errors-in-variables least-squares fit.
"""

from dataclasses import dataclass
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit, least_squares

from .fitting import FitResult, fit_curve, _fit_mask
from .latex import format_latex


def _linear_model(x, slope, intercept):
    return slope * x + intercept


def _origin_model(x, slope):
    return slope * x


def _linear_bounds(bounds, fit_intercept):
    n = 2 if fit_intercept else 1
    try:
        lo, hi = (bounds.lb, bounds.ub) if hasattr(bounds, "lb") else bounds
        lo = np.broadcast_to(np.asarray(lo, dtype=float), (n,)).copy()
        hi = np.broadcast_to(np.asarray(hi, dtype=float), (n,)).copy()
    except (TypeError, ValueError) as exc:
        raise ValueError(f"bounds benötigt Grenzen für {n} freie Parameter.") from exc
    if np.any(np.isnan(lo) | np.isnan(hi)) or np.any(lo >= hi):
        raise ValueError("Untere Parametergrenzen müssen kleiner als obere sein.")
    return lo, hi


def _column_or_array(data, key, name):
    """Read a named column or a standalone sequence as a 1-D float array."""
    if isinstance(key, str):
        if data is None:
            raise ValueError(f"{name}: Spaltenname benötigt ein DataFrame oder Dictionary.")
        try:
            key = data[key]
        except (KeyError, TypeError) as exc:
            raise ValueError(f"{name}: Spalte nicht gefunden.") from exc

    try:
        result = np.asarray(key, dtype=float)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"{name} muss numerisch sein.") from exc
    if result.ndim != 1:
        raise ValueError(f"{name} muss eindimensional sein.")
    return result


def _errors(data, spec, n, name):
    """Resolve a column name, a scalar uncertainty, or a full-length array."""
    if spec is None:
        return None
    if isinstance(spec, str):
        result = _column_or_array(data, spec, name)
    elif np.isscalar(spec):
        try:
            result = np.full(n, float(spec))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} muss numerisch sein.") from exc
    else:
        result = _column_or_array(None, spec, name)
    if result.size != n:
        raise ValueError(f"{name} muss genau {n} Einträge haben.")
    return result


def _errors_in_variables_fit(x, y, dx, dy, bounds, fit_intercept):
    """Minimize the profiled orthogonal chi-square, with optional bounds.

    Independent Gaussian coordinate errors give residuals
    (y - m*x - b) / hypot(dy, m*dx). Covariance is the unscaled local
    Gauss-Newton approximation; active bounds need non-Gaussian intervals.
    """
    model = _linear_model if fit_intercept else _origin_model
    lo, hi = bounds
    exact_y = dy is None
    if exact_y:
        if np.any(dx <= 0):
            raise ValueError("Bei ausschließlich xerr müssen alle x-Fehler strikt positiv sein.")
        if fit_intercept and np.ptp(y) == 0:
            raise ValueError("Bei ausschließlich xerr müssen die y-Werte verschieden sein.")
        if not fit_intercept and not np.any(y):
            raise ValueError("Bei ausschließlich xerr muss mindestens ein y-Wert ungleich null sein.")

        # In inverse coordinates x = a*y+c the objective is a convex quadratic.
        # For each sign of a, the bounds on m=1/a and b=-c/a become linear
        # inequalities. Enumerate the unconstrained minimum, each boundary
        # minimum, and all boundary intersections. This also handles finite
        # parameter bounds without using a singular residual at m=0.
        design = np.column_stack((y, np.ones_like(y))) if fit_intercept else y[:, None]
        weighted = design / dx[:, None]
        target = x / dx
        info = weighted.T @ weighted
        if np.linalg.matrix_rank(info) < len(lo):
            raise ValueError("Die Fitparameter sind nicht eindeutig bestimmbar.")
        inv_cov = np.linalg.inv(info)
        center = inv_cov @ weighted.T @ target
        candidates = []
        for sign in (1.0, -1.0):
            if (sign > 0 and hi[0] <= 0) or (sign < 0 and lo[0] >= 0):
                continue
            # inequalities G @ (a,c) <= h, including a=0 as a limit
            rows, rhs = [], []
            def add(row, value):
                rows.append(row[:len(lo)])
                rhs.append(value)
            add([-sign, 0.0], 0.0)
            if np.isfinite(lo[0]) and lo[0] * sign > 0:
                add([1.0, 0.0], 1.0 / lo[0])
            if np.isfinite(hi[0]) and hi[0] * sign > 0:
                add([-1.0, 0.0], -1.0 / hi[0])
            if fit_intercept:
                if np.isfinite(lo[1]):
                    add([sign * lo[1], sign], 0.0)
                if np.isfinite(hi[1]):
                    add([-sign * hi[1], -sign], 0.0)
            G, h = np.array(rows), np.array(rhs)
            points = [center]
            for i, row in enumerate(G):
                direction = inv_cov @ row
                points.append(center - direction * ((row @ center - h[i]) / (row @ direction)))
                if len(lo) == 2:
                    for j in range(i):
                        pair = G[[i, j]]
                        if np.linalg.matrix_rank(pair) == 2:
                            points.append(np.linalg.solve(pair, h[[i, j]]))
            for point in points:
                if np.all(G @ point <= h + 1e-10 * (1 + np.abs(h))):
                    cost = float(np.sum((weighted @ point - target)**2))
                    candidates.append((cost, point))
        if not candidates:
            raise ValueError("Keine zulässige Lösung für die Parametergrenzen.")
        _, inverse = min(candidates, key=lambda item: item[0])
        a = inverse[0]
        if not np.isfinite(a) or abs(a) <= 1e-14:
            raise ValueError("Die inverse Steigung ist null; kein endlicher Fit möglich.")
        if fit_intercept:
            c = inverse[1]
            params = np.array([1.0/a, -c/a])
            transform = np.array([[-1.0/a**2, 0.0], [c/a**2, -1.0/a]])
        else:
            params = np.array([1.0/a])
            transform = np.array([[-1.0/a**2]])
        # Small rounding deviations at an active constraint are harmless.
        if np.any(params < lo - 1e-8*(1+np.abs(lo))) or np.any(params > hi + 1e-8*(1+np.abs(hi))):
            raise ValueError("Inverse Fitlösung liegt außerhalb der Parametergrenzen.")
        params = np.clip(params, lo, hi)
        covariance = transform @ inv_cov @ transform.T
    else:
        # A weighted linear fit supplies a starting point; project it into bounds.
        start = (np.polyfit(x, y, 1, w=1.0/dy) if fit_intercept else
                 np.array([np.sum(x*y/dy**2)/np.sum((x/dy)**2)]))
        start = np.clip(start, lo, hi)

        def residuals(params):
            m = params[0]
            return (y - model(x, *params)) / np.hypot(dy, m*dx)

        def jacobian(params):
            m = params[0]
            scale = np.hypot(dy, m*dx)
            r = y - model(x, *params)
            derivative = -x/scale - r*m*dx**2/scale**3
            return (np.column_stack((derivative, -1.0/scale))
                    if fit_intercept else derivative[:, None])

        optimization = least_squares(
            residuals, start, jac=jacobian, bounds=(lo, hi), max_nfev=10000,
        )
        if not optimization.success or not np.all(np.isfinite(optimization.x)):
            raise ValueError("Errors-in-Variables-Fit ist nicht konvergiert.")
        params = optimization.x
        information = optimization.jac.T @ optimization.jac
        if np.linalg.matrix_rank(information) < len(lo):
            raise ValueError("Die Fitparameter sind nicht eindeutig bestimmbar.")
        covariance = np.linalg.inv(information)

    if not np.all(np.isfinite(covariance)) or np.any(np.diag(covariance) < 0):
        raise ValueError("Die Parameter-Kovarianz ist nicht endlich oder ungültig.")
    return FitResult(model, x, y, dx, dy, params, covariance)


def _label(name, unit):
    label = name if isinstance(name, str) else "x"
    return f"{label} [{unit}]" if unit else label


class LinearFitResult(FitResult):
    """FitResult for y = m*x+b, with named parameters and residual helpers."""

    def __init__(self, result, x_unit=None, y_unit=None, errors_in_variables=False,
                 fit_intercept=True):
        super().__init__(
            result.func,
            result.x_data,
            result.y_data,
            result.x_err,
            result.y_err,
            result.params,
            result.cov_matrix,
        )
        self.fit_intercept = fit_intercept
        self.x_unit = x_unit
        self.y_unit = y_unit
        self.fit_method = "errors_in_variables" if errors_in_variables else "vertical"
        # These are the *vertical equivalents* of orthogonal measurement errors.
        if errors_in_variables:
            self.effective_y_err = np.hypot(
                self.y_err if self.y_err is not None else 0.0,
                self.slope * self.x_err,
            )
            # Analytically profiled true x coordinates for the linear model.
            self.x_adjusted = self.x_data + (
                self.slope * (self.x_err / self.effective_y_err)**2 * self.residuals
            )
            self.chi2 = float(np.sum((self.residuals / self.effective_y_err) ** 2))
            self.ndof = len(self.x_data) - len(self.params)
            self.chi2_red = self.chi2 / self.ndof if self.ndof > 0 else np.nan
        else:
            self.effective_y_err = self.y_err
        # Populated when linear_fit(plot=True) creates a figure.
        self.axes = None

    @property
    def slope(self):
        return float(self.params[0])

    @property
    def intercept(self):
        return float(self.params[1]) if self.fit_intercept else 0.0

    @property
    def slope_err(self):
        return float(self.param_errs[0])

    @property
    def intercept_err(self):
        return float(self.param_errs[1]) if self.fit_intercept else 0.0

    @property
    def line_cov_matrix(self):
        """Full (m,b) covariance, including a zero row/column for a fixed b."""
        if self.fit_intercept:
            return self.cov_matrix
        return np.array([[self.cov_matrix[0, 0], 0.0], [0.0, 0.0]])

    @property
    def slope_unit(self):
        if self.y_unit and self.x_unit:
            return f"{self.y_unit}/{self.x_unit}"
        if self.y_unit:
            return self.y_unit
        if self.x_unit:
            return f"1/{self.x_unit}"
        return None

    def print_results(self, param_names=None):
        """Print the coefficients with PAP uncertainty formatting and units."""
        names = param_names if param_names is not None else ("m", "b")
        if len(names) < len(self.params):
            raise ValueError("param_names benötigt eine Bezeichnung pro freiem Parameter.")
        print("=== LINEARER FIT: y = m*x + b ===" if self.fit_intercept
              else "=== LINEARER FIT: y = m*x (b = 0 fixiert) ===")
        entries = [(names[0], self.slope, self.slope_err, self.slope_unit)]
        if self.fit_intercept:
            entries.append((names[1], self.intercept, self.intercept_err, self.y_unit))
        for name, value, error, unit in entries:
            unit_text = f" [{unit}]" if unit else ""
            print(f"{name} = {format_latex(value, error)}{unit_text}")
        if np.isfinite(self.chi2_red):
            print(f"chi2 = {self.chi2:.3g}, ndof = {self.ndof}, chi2/ndof = {self.chi2_red:.3g}")
        else:
            print("chi2 nicht definiert (keine Messunsicherheiten oder ndof=0).")


@dataclass(frozen=True)
class LineIntersection:
    """Intersection of two independently fitted lines; uncertainties are 1 sigma."""

    x: float
    y: float
    x_err: float
    y_err: float
    cov_matrix: np.ndarray

    def as_frequency(self, base=10.0):
        """Convert x = log_base(f / unit) to (f / unit, sigma_f / unit).

        The uncertainty is first-order propagated: sigma_f = ln(base)*f*sigma_x.
        No logarithmic conversion is performed unless this method is called.
        """
        if not np.isfinite(base) or base <= 0 or base == 1:
            raise ValueError("base muss positiv und ungleich 1 sein.")
        frequency = float(base ** self.x)
        sigma = float(abs(np.log(base) * frequency * self.x_err))
        if not np.isfinite(frequency) or not np.isfinite(sigma):
            raise ValueError("Frequenzumrechnung überläuft den Zahlenbereich.")
        return frequency, sigma

    def print_results(self, x_unit=None, y_unit=None):
        x_suffix = f" [{x_unit}]" if x_unit else ""
        y_suffix = f" [{y_unit}]" if y_unit else ""
        print(f"x = {format_latex(self.x, self.x_err)}{x_suffix}")
        print(f"y = {format_latex(self.y, self.y_err)}{y_suffix}")


def intersect_lines(first, second):
    """Intersect two independent LinearFitResult objects.

    Propagates each complete (slope, intercept) covariance matrix, including
    their off-diagonal correlations. The two *different fits* are assumed
    independent: shared calibration/systematic errors need separate treatment.
    """
    if not isinstance(first, LinearFitResult) or not isinstance(second, LinearFitResult):
        raise TypeError("intersect_lines benötigt zwei Ergebnisse aus linear_fit().")

    m1, b1 = first.slope, first.intercept
    m2, b2 = second.slope, second.intercept
    dm = m1 - m2
    if abs(dm) <= 1e-12 * max(1.0, abs(m1), abs(m2)):
        raise ValueError("Die Geraden sind parallel oder numerisch nicht unterscheidbar.")

    x = (b2 - b1) / dm
    y = m1 * x + b1

    dx1 = np.array([-x / dm, -1.0 / dm])
    dx2 = np.array([x / dm, 1.0 / dm])
    dy1 = np.array([x, 1.0]) + m1 * dx1
    dy2 = m1 * dx2
    jac1 = np.vstack([dx1, dy1])
    jac2 = np.vstack([dx2, dy2])

    cov1 = np.asarray(first.line_cov_matrix, dtype=float)
    cov2 = np.asarray(second.line_cov_matrix, dtype=float)
    if (
        cov1.shape != (2, 2) or cov2.shape != (2, 2)
        or not np.all(np.isfinite(cov1)) or not np.all(np.isfinite(cov2))
    ):
        raise ValueError("Beide Fits benötigen endliche 2x2-Kovarianzmatrizen.")

    covariance = jac1 @ cov1 @ jac1.T + jac2 @ cov2 @ jac2.T
    variances = np.diag(covariance)
    if np.any(variances < 0):
        raise ValueError("Die berechnete Schnittpunkt-Kovarianz ist ungültig.")
    return LineIntersection(float(x), float(y), float(np.sqrt(variances[0])),
                            float(np.sqrt(variances[1])), covariance)


def linear_fit(
    data, x, y=None, *, xerr=None, yerr=None, x_range=None, fitranges=None,
    bounds=(-np.inf, np.inf), fit_intercept=True,
    show=False, plot=None, report=None, residuals=False, standardized=False,
    plot_band=False, x_unit=None, y_unit=None,
    xlabel=None, ylabel=None, title="", ax=None, language="de", **plot_kwargs
):
    """Fit y=m*x+b and optionally display the figure and PAP-formatted result.

    Two calling conventions:
        linear_fit(df, "time", "voltage", yerr="dvoltage", show=True)
        linear_fit(x_array, y_array, yerr=0.1, show=True)

    x_range=(minimum, maximum) or fitranges=[(lo, hi), ...] selects an inclusive union.
    bounds=(lower, upper) follows SciPy; free parameter order is (m,b) or (m,).
    fit_intercept=False fixes b=0 exactly.
    With xerr, fits use an errors-in-variables objective accounting for
    both horizontal and vertical independent measurement uncertainties.
    With only yerr, the existing curve_fit weighted fit is unchanged.
    With only xerr, y is assumed exact and inverse linear regression is used;
    in that case every selected xerr must be strictly positive.
    Covariances are first-order Gauss-Newton estimates, not scaled by chi2/ndof.
    All 1-D error arrays must match the *unfiltered* data length.
    show=True enables plot and report unless overridden separately.
    """
    column_mode = y is not None
    if column_mode:
        data_source = data
        x_values = _column_or_array(data, x, "x")
        y_values = _column_or_array(data, y, "y")
        x_name, y_name = x, y
    else:
        data_source = None
        x_values = _column_or_array(None, data, "x")
        y_values = _column_or_array(None, x, "y")
        x_name, y_name = "x", "y"

    if len(x_values) != len(y_values):
        raise ValueError("x und y müssen gleich viele Werte enthalten.")
    n = len(x_values)
    dx = _errors(data_source, xerr, n, "xerr")
    dy = _errors(data_source, yerr, n, "yerr")

    mask = _fit_mask(x_values, x_range, fitranges)

    x_values, y_values = x_values[mask], y_values[mask]
    dx = dx[mask] if dx is not None else None
    dy = dy[mask] if dy is not None else None

    if x_values.size < (3 if fit_intercept else 2):
        raise ValueError("Ein linearer Fit benötigt mindestens drei Messpunkte."
                         if fit_intercept else "Ein Ursprungsfit benötigt mindestens zwei Messpunkte.")
    if not np.all(np.isfinite(x_values)) or not np.all(np.isfinite(y_values)):
        raise ValueError("x und y müssen im gewählten Bereich endlich sein.")
    if fit_intercept and np.ptp(x_values) == 0:
        raise ValueError("Die x-Werte müssen verschieden sein.")
    if not fit_intercept and not np.any(x_values):
        raise ValueError("Für einen Ursprungsfit muss mindestens ein x-Wert ungleich null sein.")
    if dx is not None and (not np.all(np.isfinite(dx)) or np.any(dx < 0)):
        raise ValueError("xerr muss endlich und nichtnegativ sein.")
    if dy is not None and (not np.all(np.isfinite(dy)) or np.any(dy <= 0)):
        raise ValueError("yerr muss endlich und strikt positiv sein.")

    limits = _linear_bounds(bounds, fit_intercept)
    model = _linear_model if fit_intercept else _origin_model
    use_eiv = dx is not None and np.any(dx > 0)
    if use_eiv:
        base = _errors_in_variables_fit(x_values, y_values, dx, dy, limits, fit_intercept)
    else:
        data_fit = {"x": x_values, "y": y_values}
        if dx is not None:
            data_fit["xerr"] = dx
        if dy is not None:
            data_fit["yerr"] = dy
        base = fit_curve(
            model, data_fit, "x", "y", bounds=limits,
            x_err_col="xerr" if dx is not None else None,
            y_err_col="yerr" if dy is not None else None,
        )
    result = LinearFitResult(
        base, x_unit=x_unit, y_unit=y_unit,
        errors_in_variables=use_eiv, fit_intercept=fit_intercept,
    )

    plot_requested = bool(show) if plot is None else bool(plot)
    report_requested = bool(show) if report is None else bool(report)
    if (residuals or standardized) and not plot_requested:
        raise ValueError("Residuen benötigen plot=True oder show=True.")
    if standardized:
        residuals = True
    if ax is not None and residuals:
        raise ValueError("Bei residuals=True wird ein zweiachsiger Plot neu angelegt.")

    x_label = xlabel if xlabel is not None else _label(x_name, x_unit)
    y_label = ylabel if ylabel is not None else _label(y_name, y_unit)

    if plot_requested:
        if residuals:
            result.axes = result.plot_with_residuals(
                title=title, xlabel=x_label, ylabel=y_label,
                standardized=standardized, plot_band=plot_band, language=language,
                **plot_kwargs,
            )
        else:
            plot_ax = result.plot(
                ax=ax, title=title, xlabel=x_label, ylabel=y_label,
                plot_band=plot_band, language=language, **plot_kwargs,
            )
            result.axes = (plot_ax,)
        plt.show()
    if report_requested:
        result.print_results()
    return result
