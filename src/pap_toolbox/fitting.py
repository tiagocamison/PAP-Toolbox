import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

from .latex import format_latex


class FitResult:
    """
    Speichert die Ergebnisse eines Curve-Fits und stellt Hilfsfunktionen
    zum Plotten und Auswerten zur Verfügung.
    """

    def __init__(self, func, x_data, y_data, x_err, y_err, popt, pcov):
        self.func = func
        self.x_data = np.asarray(x_data, dtype=float)
        self.y_data = np.asarray(y_data, dtype=float)
        self.x_err = np.asarray(x_err, dtype=float) if x_err is not None else None
        self.y_err = np.asarray(y_err, dtype=float) if y_err is not None else None

        self.params = popt
        self.param_errs = np.sqrt(np.diag(pcov))
        self.cov_matrix = pcov

        if self.y_err is not None:
            residuals = self.y_data - self.func(self.x_data, *self.params)
            self.chi2 = np.sum((residuals / self.y_err) ** 2)
            self.ndof = len(self.x_data) - len(self.params)
            self.chi2_red = self.chi2 / self.ndof if self.ndof > 0 else np.nan
        else:
            self.chi2, self.ndof, self.chi2_red = np.nan, np.nan, np.nan

    def evaluate(self, x):
        """Wertet die angefittete Funktion an den Stellen x aus."""
        return self.func(x, *self.params)

    def print_results(self, param_names=None):
        """Gibt die gefitteten Parameter inkl. Fehler im PAP-Format aus."""
        print("=== FIT ERGEBNISSE ===")
        for i, (p_val, p_err) in enumerate(zip(self.params, self.param_errs)):
            name = param_names[i] if param_names and i < len(param_names) else f"p{i}"
            print(f"{name} = {format_latex(p_val, p_err)}")
        if not np.isnan(self.chi2_red):
            print(
                f"Fit-Güte: chi2 = {self.chi2:.2f}, ndof = {self.ndof} "
                f"-> chi2/ndof = {self.chi2_red:.2f}"
            )
        print("======================")

    def plot(
        self,
        ax=None,
        title="",
        xlabel="",
        ylabel="",
        data_label="Messdaten",
        fit_label="Fit",
        color_data="black",
        color_fit="red",
        plot_fit=True,
        plot_band=True,
        extra_points=100,
        x_lim=None,
        y_lim=None,
        language="de",
    ):
        """Plottet Daten, Fit und Fehlerband. Optimiert für große Datensätze (>10k Punkte)."""
        if ax is None:
            _, ax = plt.subplots(figsize=(8, 5))

        N = len(self.x_data)
        if N > 1000:
            fmt, alpha, markersize = ",", 0.5, 1
            err_every = max(1, N // 50)
            raster = True
        else:
            fmt, alpha, markersize = "x", 1.0, 5
            err_every = 1
            raster = False

        ax.errorbar(
            self.x_data,
            self.y_data,
            xerr=self.x_err,
            yerr=self.y_err,
            fmt=fmt,
            color=color_data,
            label=data_label,
            capsize=3 if N <= 1000 else 0,
            alpha=alpha,
            markersize=markersize,
            errorevery=err_every,
            rasterized=raster,
            zorder=5,
        )

        if x_lim is not None:
            ax.set_xlim(x_lim)
        if y_lim is not None:
            ax.set_ylim(y_lim)

        if plot_fit:
            current_xlim = ax.get_xlim()
            x_smooth = np.linspace(current_xlim[0], current_xlim[1], extra_points)
            y_smooth = self.evaluate(x_smooth)
            ax.plot(x_smooth, y_smooth, "-", color=color_fit, label=fit_label, zorder=4)

            if plot_band and self.cov_matrix is not None and not np.isinf(self.cov_matrix).all():
                try:
                    samples = np.random.multivariate_normal(self.params, self.cov_matrix, 1000)
                    y_samples = np.array([self.func(x_smooth, *p) for p in samples])
                    y_lower = np.percentile(y_samples, 15.87, axis=0)
                    y_upper = np.percentile(y_samples, 84.13, axis=0)
                    if language.casefold() == "de":
                        label = r"1$\sigma$ Fehlerband"
                    elif language.casefold() == "en":
                        label = r"1$\sigma$ Error Band"
                    else:
                        raise NotImplementedError("Sprache nicht supported: use 'de' or 'en'.")
                    ax.fill_between(
                        x_smooth,
                        y_lower,
                        y_upper,
                        color=color_fit,
                        alpha=0.2,
                        label=label,
                        zorder=3,
                    )
                except ValueError:
                    pass
            ax.set_xlim(current_xlim)

        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(True, linestyle="--", alpha=0.7)
        ax.legend()
        return ax


def fit_curve(func, df_or_dict, x_col, y_col, x_err_col=None, y_err_col=None, p0=None):
    """
    Führt einen Curve-Fit durch. Falls y_err gegeben ist, wird ein gewichteter Fit gemacht.
    Gibt ein FitResult-Objekt zurück.
    """
    x_data = df_or_dict[x_col]
    y_data = df_or_dict[y_col]

    x_err = None
    if x_err_col is not None:
        if isinstance(x_err_col, str):
            x_err = df_or_dict[x_err_col]
        else:
            x_err = np.full(len(x_data), x_err_col)

    y_err = None
    if y_err_col is not None:
        if isinstance(y_err_col, str):
            y_err = df_or_dict[y_err_col]
        else:
            y_err = np.full(len(y_data), y_err_col)

    if y_err is not None:
        popt, pcov = curve_fit(func, x_data, y_data, sigma=y_err, absolute_sigma=True, p0=p0)
    else:
        popt, pcov = curve_fit(func, x_data, y_data, p0=p0)

    return FitResult(func, x_data, y_data, x_err, y_err, popt, pcov)


def min_max_slope(x_data, y_data, y_err):
    """
    Berechnet die minimale und maximale Steigung für Handzeichnungen (PAP 1).
    Nimmt den ersten und letzten Messpunkt inklusive ihrer Fehlerbalken.
    """
    x = np.asarray(x_data, dtype=float)
    y = np.asarray(y_data, dtype=float)
    dy = np.asarray(y_err, dtype=float)

    idx = np.argsort(x)
    x, y, dy = x[idx], y[idx], dy[idx]

    m_max = ((y[-1] + dy[-1]) - (y[0] - dy[0])) / (x[-1] - x[0])
    m_min = ((y[-1] - dy[-1]) - (y[0] + dy[0])) / (x[-1] - x[0])

    m_opt = (y[-1] - y[0]) / (x[-1] - x[0])
    dm = abs(m_max - m_min) / 2.0

    print("=== MIN/MAX STEIGUNGEN FÜR MILLIMETERPAPIER ===")
    print(f"Optimale Steigung: m_opt = {m_opt:.4g}")
    print(f"Steilste Gerade:  m_max = {m_max:.4g}")
    print(f"Flachste Gerade:  m_min = {m_min:.4g}")
    print(f"Unsicherheit:     Delta m = {dm:.4g}")
    print("===============================================")

    return m_opt, dm
