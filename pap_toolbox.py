import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import sympy as sp

from IPython.display import display, Math
from decimal import Decimal
from scipy.optimize import curve_fit


# =====================================================================
# 1. GERÄTEFEHLER (SYSTEMATISCHE FEHLER)
# =====================================================================

def err_digital(reading, percent=0.0, digits=0, resolution=1.0):
    """
    Berechnet den absoluten Fehler eines Digitalmessgeräts.
    Formel: Fehler = |Wert| * (percent/100) + digits * resolution
    """
    reading = np.asarray(reading)
    return np.abs(reading) * (percent / 100.0) + (digits * resolution)

def err_analog(resolution, fraction=0.5):
    """
    Gibt den Ablesefehler einer Analogskala (Lineal, Barometer etc.) zurück.
    Standardmäßig ein halber Skalenteil (fraction=0.5).
    """
    return resolution * fraction


# =====================================================================
# 2. STATISTIK FÜR MESSREIHEN (Z.B. MEHRFACHE ZEITMESSUNG)
# =====================================================================

def series_stats(data, print_results=False):
    """
    Nimmt ein Array von Messwerten (z.B. 10 Schwingungsdauern) und berechnet:
    - N (Anzahl)
    - mean (Mittelwert)
    - std (Empirische Standardabweichung der Einzelmessung)
    - sem (Standardfehler des Mittelwerts = std / sqrt(N)) -> Fehler fürs Protokoll
    """
    arr = np.asarray(data)
    # NaN-Werte ignorieren
    arr = arr[~np.isnan(arr)]
    
    N = len(arr)
    if N == 0:
        print("Warning: no valid values passed. Return nan.")
        return {"N": 0, "mean": np.nan, "std": np.nan, "sem": np.nan}
    
    mean = np.mean(arr)
    std = np.std(arr, ddof=1) if N > 1 else 0.0 # ddof=1 für empirische (N-1) Standardabweichung
    sem = std / np.sqrt(N) if N > 1 else 0.0
    
    if print_results:
        print(f"Messreihe (N={N}):")
        print(f"  Mittelwert:        {mean}")
        print(f"  Std-Abweichung:    {std} (Fehler einer Einzelmessung)")
        print(f"  Std-Fehler (Mittel): {sem} (<= DAS ist dein Messfehler!)")
        
    return {"N": N, "mean": mean, "std": std, "sem": sem}

def weighted_mean(values, errors):
    """
    Berechnet das fehlergewichtete Mittel und dessen Unsicherheit.
    Formel: w_i = 1 / sigma_i^2
    """
    vals = np.asarray(values, dtype=float)
    errs = np.asarray(errors, dtype=float)
    
    weights = 1.0 / (errs**2)
    w_mean = np.sum(weights * vals) / np.sum(weights)
    w_err = 1.0 / np.sqrt(np.sum(weights))
    
    return w_mean, w_err

# =====================================================================
# 3. PAP-RUNDUNGSREGELN & FORMATIERUNG
# =====================================================================

def first_digit_func(x):
    x = Decimal(str(abs(x)))

    if x == 0:
        raise ValueError("First significant digit is undefined for zero.")

    exponent = x.adjusted()
    return int(x.scaleb(-exponent)), x.adjusted()

def _get_decimals_pap(err): # Dies ist falsch, da es floating point errors gibt, aber in einer neuen Version ist es gefixt.
    """Interne Hilfsfunktion: Ermittelt auf welche Nachkommastelle gerundet werden muss."""
    if pd.isna(err) or err == 0:
        return None
    
    first_digit, oom = first_digit_func(err)
    
    # PAP-Regel: Beginnt mit 1 oder 2 -> 2 sig. Stellen, sonst 1
    sig_figs = 2 if first_digit in [1, 2] else 1
    
    # Nachkommastellen berechnen
    decimals = sig_figs - 1 - oom
    return decimals

def round_pap(val, err):
    """
    Numerische PAP-Rundung. Gibt (val_rounded, err_rounded) als Floats zurück.
    Gut, falls man mit den gerundeten Werten weiterrechnen/plotten will.
    """
    if pd.isna(val) or pd.isna(err) or err == 0:
        return float(val), float(err)
    
    decimals = _get_decimals_pap(err)
    
    # round(x, -1) rundet auf Zehner, round(x, 2) auf Hundertstel
    return round(float(val), decimals), round(float(err), decimals)

def format_latex(val, err=None):
    """
    Rundet numerische Werte nach PAP-Regeln.
    Reine Textfelder oder Zahlen ohne Fehler werden sauber ohne \pm formatiert.
    """
    if pd.isna(val):
        return ""
    
    # 1. Fall: Reiner Text (z.B. "Blei (Pb)") -> Unverändert ausgeben
    if isinstance(val, str) and not _is_numeric(val):
        return str(val)
    
    # 2. Fall: Zahl OHNE Fehler (oder Fehler ist None / 0)
    if err is None or pd.isna(err) or err == 0 or str(err).strip().lower() in ["none", "nan", ""]:
        try:
            f_val = float(val)
            return f"{int(f_val)}" if f_val.is_integer() else f"{f_val}"
        except (ValueError, TypeError):
            return str(val)
            
    # 3. Fall: Zahl MIT Fehler -> PAP-Rundungsregel anwenden
    try:
        val = float(val)
        err = float(err)
    except (ValueError, TypeError):
        return str(val)

    
    first_digit, oom = first_digit_func(err)
    sig_figs = 2 if first_digit in [1, 2] else 1
    decimals = sig_figs - 1 - oom
    
    if decimals <= 0:
        v_rnd = int(round(val, decimals))
        e_rnd = int(round(err, decimals))
        return f"{v_rnd} $\\pm$ {e_rnd}"
    else:
        fmt = f".{decimals}f"
        v_rnd = round(val, decimals)
        e_rnd = round(err, decimals)
        return f"{v_rnd:{fmt}} $\\pm$ {e_rnd:{fmt}}"


# Vektorisieren, damit sie sofort auf Pandas Series / Numpy Arrays angewendet werden können
v_round_pap = np.vectorize(round_pap)
v_format_latex = np.vectorize(format_latex)


# =====================================================================
# 4. DISKUSSIONS-HELPER (SIGMA-ABSTAND)
# =====================================================================

def compare_to_literature(val, err, lit_val, lit_err=0.0, verbose=False):
    """
    Berechnet die Abweichung vom Literaturwert in Vielfachen von Sigma.
    Gibt direkt den Textbaustein für die Diskussion aus.
    """
    diff = abs(val - lit_val)
    sigma_total = np.sqrt(err**2 + lit_err**2)
    
    if sigma_total == 0:
        return "Fehler ist 0, Berechnung der Sigma-Abweichung nicht möglich."
    
    sigma_dist = diff / sigma_total

    if verbose:
        print(f"--- Vergleich mit Literaturwert ---")
        print(f"Dein Wert:   {format_latex(val, err)}")
        print(f"Literatur:   {format_latex(lit_val, lit_err)}")
        print(f"Abweichung:  {diff:.4g} (absolut)")
        print(f"Sigma-Abst.: {sigma_dist:.2f} \sigma")
        print(f"\nProtokoll-Text:")
        if sigma_dist <= 1.0:
            print(f"Der Messwert weicht um {sigma_dist:.2f} \\sigma vom Literaturwert ab und stimmt im Rahmen der Messgenauigkeit hervorragend überein.")
        elif sigma_dist <= 3.0:
            print(f"Der Messwert weicht um {sigma_dist:.2f} \\sigma vom Literaturwert ab. Die Übereinstimmung ist zufriedenstellend, deutet aber auf leichte systematische Fehler hin.")
        else:
            print(f"Achtung! Der Messwert weicht um {sigma_dist:.2f} \\sigma vom Literaturwert ab (> 3 Sigma). Hier muss im Protokoll ein systematischer Fehler diskutiert werden!")
        
    return sigma_dist

v_compare_to_literature = np.vectorize(compare_to_literature)

# =====================================================================
# 5. FEHLERFORTPFLANZUNG
# =====================================================================

class ErrorPropagator:
    def __init__(self, formula_str):
        """
        Initialisiert den Propagator mit einer als String übergebenen Formel.
        Beispiel: ErrorPropagator("m_w * c_w * (T_1 - T_bar) / (T_bar - T_2)")
        """
        # 1. Formel parsen
        self.f_expr = sp.parsing.sympy_parser.parse_expr(formula_str)
        
        # 2. Variablen automatisch extrahieren und alphabetisch sortieren
        self.vars = sorted(list(self.f_expr.free_symbols), key=lambda x: x.name)
        
        # 3. Exakte Fehlersymbole erstellen
        self.err_vars = [sp.Symbol(rf"d {v.name}", positive=True, real=True) for v in self.vars]
        
        # 4. Gaußsche Fehlerfortpflanzung (Absolut und Relativ)
        variance = 0
        for v, ev in zip(self.vars, self.err_vars):
            derivative = sp.diff(self.f_expr, v)
            variance += (derivative * ev)**2

        self.variance_terms = {}
        variance = 0
        for v, ev in zip(self.vars, self.err_vars):
            term = (sp.diff(self.f_expr, v) * ev)**2
            self.variance_terms[v.name] = term
            variance += term
            
        self.abs_err_expr = sp.simplify(sp.sqrt(sp.factor(variance)))
        self.rel_err_expr = sp.simplify(sp.sqrt(sp.factor(variance / self.f_expr**2)))
        
        # 5. Numerische Funktionen für schnelle Auswertung (numpy)
        self.val_func = sp.lambdify(self.vars, self.f_expr, "numpy")
        self.err_func = sp.lambdify(self.vars + self.err_vars, self.abs_err_expr, "numpy")

    def _parse_inputs(self, vals_dict, errs_dict):
        """Hilfsfunktion zum Verarbeiten der Arrays und %-Strings."""
        val_args, err_args = [], []
        for v in self.vars:
            v_name = v.name
            if v_name not in vals_dict or v_name not in errs_dict:
                raise ValueError(f"Fehlender Wert oder Fehler für Variable: {v_name}")
            
            c_val = np.asarray(vals_dict[v_name], dtype=float)
            c_err = errs_dict[v_name]
            
            if isinstance(c_err, str) and c_err.endswith('%'):
                rel_frac = float(c_err.strip('%')) / 100.0
                c_err = np.abs(c_val) * rel_frac
            else:
                c_err = np.asarray(c_err, dtype=float)
                
            val_args.append(c_val)
            err_args.append(c_err)
        return val_args, err_args

    def _get_latex(self, expr):
        """Hilfsfunktion: Ersetzt intern die Fehler-Symbole durch sauberes \Delta für LaTeX."""
        subs_dict = {ev: sp.Symbol(rf"\Delta {v.name}") for v, ev in zip(self.vars, self.err_vars)}
        return sp.latex(expr.subs(subs_dict))

    def show_math(self):
        """Rendert die theoretischen Formeln wunderschön im Jupyter Notebook."""
        print("=== THEORETISCHE FORMELN FÜR DAS PROTOKOLL ===")
        display(Math(r"f = " + sp.latex(self.f_expr)))
        display(Math(r"\Delta f = " + self._get_latex(self.abs_err_expr)))
        display(Math(r"\frac{\Delta f}{f} = " + self._get_latex(self.rel_err_expr)))
        
        print("\n--- Roher LaTeX-Code zum Kopieren ---")
        print("Wert:    $$ " + sp.latex(self.f_expr) + " $$")
        print("Absolut: $$ " + self._get_latex(self.abs_err_expr) + " $$")
        print("Relativ: $$ " + self._get_latex(self.rel_err_expr) + " $$")
        print("-------------------------------------\n")

    def analyze_error(self, vals_dict, errs_dict, threshold=0.20):
        """
        Analysiert die Fehlerbeiträge basierend auf den Messdaten (PAP 20%-Regel).
        Wirft Terme raus, die kleiner als 'threshold' vom Maximalbeitrag sind,
        und generiert eine gekürzte LaTeX-Formel.
        """
        
        val_args, err_args = self._parse_inputs(vals_dict, errs_dict)
        
        contribs = {}
        for v in self.vars:
            term_expr = self.variance_terms[v.name]
            term_func = sp.lambdify(self.vars + self.err_vars, term_expr, "numpy")
            
            # Berechne das Array der Varianzen für diesen Term
            var_array = term_func(*val_args, *err_args)
            
            # Durchschnittliche absolute Fehler-Auswirkung dieses Terms
            contribs[v.name] = np.sqrt(np.mean(var_array))
            
        max_contrib = max(contribs.values())
        if max_contrib == 0:
            print("Alle Fehler sind Null. Keine Analyse möglich.")
            return

        print(f"=== FEHLER-ANALYSE (Schwelle: {threshold*100:.0f}% vom Maximalbeitrag) ===")
        
        reduced_variance = 0
        dropped_vars = []
        
        for v_name, contrib in contribs.items():
            ratio = contrib / max_contrib
            print(f"Beitrag {v_name:<8}: {ratio*100:>5.1f}% des stärksten Fehlers")
            
            if ratio >= threshold:
                reduced_variance += self.variance_terms[v_name]
            else:
                dropped_vars.append(v_name)
                
        if not dropped_vars:
            print("\n=> Alle Terme sind signifikant. Keine Kürzung möglich.")
            return
            
        print(f"\n=> Gestrichene Terme (< {threshold*100:.0f}%): {', '.join(dropped_vars)}")
        
        reduced_err_expr = sp.simplify(sp.sqrt(sp.factor(reduced_variance)))
        
        print("\n--- Neue, gekürzte LaTeX-Formel fürs Protokoll ---")
        display(Math(rf"\Delta f_{{red}} = {self._get_latex(reduced_err_expr)}"))
        print("$$ " + self._get_latex(reduced_err_expr) + " $$")

    def evaluate(self, vals_dict, errs_dict):
        """
        Berechnet die Zahlenwerte für Skalare, Arrays oder Pandas Columns.
        - vals_dict: Dictionary mit Werten, z.B. {"m_w": 125, "T_1": df["T1"]}
        - errs_dict: Dictionary mit Fehlern. Unterstützt Strings für %, z.B. {"m_w": "1%"}
        """
        val_args = []
        err_args = []
        
        for v in self.vars:
            v_name = v.name
            if v_name not in vals_dict or v_name not in errs_dict:
                raise ValueError(f"Fehlender Wert oder Fehler für Variable: {v_name}")
                
            current_val = np.asarray(vals_dict[v_name], dtype=float)
            current_err = errs_dict[v_name]
            
            # Prüfen, ob der Fehler relativ als String (z.B. "5%") übergeben wurde
            if isinstance(current_err, str) and current_err.endswith('%'):
                rel_frac = float(current_err.strip('%')) / 100.0
                current_err = np.abs(current_val) * rel_frac
            else:
                current_err = np.asarray(current_err, dtype=float)
                
            val_args.append(current_val)
            err_args.append(current_err)
            
        # Berechnen mit NumPy
        calc_val = self.val_func(*val_args)
        calc_err = self.err_func(*val_args, *err_args)
        
        return calc_val, calc_err

    def print_latex(self):
        err_latex = sp.latex(self.abs_err_expr)
        for v in self.vars:
            err_latex = err_latex.replace(f"d{v.name}", rf"\Delta {sp.latex(v)}")
        display(Math(rf"f = {sp.latex(self.f_expr)} \quad \Longrightarrow \quad \Delta f = {err_latex}"))

# =====================================================================
# 6. TABELLENHELPER
# =====================================================================

def format_separate(val, err):
    """
    Gibt ein Tuple (Wert_String, Fehler_String) zurück. 
    Wichtig für Tabellen mit getrennten Fehler-Spalten.
    """
    if err is None or pd.isna(err) or err == 0:
        return f"{val}", ""
    
    oom = math.floor(math.log10(abs(err)))
    first_digit = math.floor(abs(err) / 10**oom)
    sig_figs = 2 if first_digit in [1, 2] else 1
    decimals = sig_figs - 1 - oom
    
    if decimals <= 0:
        return f"{int(round(val, decimals))}", f"{int(round(err, decimals))}"
    else:
        fmt = f".{decimals}f"
        return f"{round(val, decimals):{fmt}}", f"{round(err, decimals):{fmt}}"

def _is_numeric(val):
    """Prüft, ob ein Wert eine Zahl ist (Float/Int)."""
    try:
        float(val)
        return True
    except (ValueError, TypeError):
        return False

def export_pap_table(df, config, caption="", label=""):
    """
    Exportiert ein DataFrame als saubere LaTeX Tabelle.
    
    config akzeptiert jetzt flexibel 2 oder 3 Einträge pro Spalte:
    - 2 Einträge für reine Text/Zahlen-Spalten: (Spaltenname, Header)
    - 3 Einträge für Messwerte mit Fehler:     (Spaltenname, Fehlerspalte_oder_Wert, Header)
    """
    export_data = {}
    col_headers = []
    
    for item in config:
        if len(item) == 2:
            val_col, header = item
            err_col = None
        else:
            val_col, err_col, header = item
            
        # Fehlerdaten auflösen
        if err_col is None or str(err_col).strip().lower() in ["none", "nan", ""]:
            err_data = [None] * len(df)
        elif isinstance(err_col, str) and err_col in df.columns:
            err_data = df[err_col]
        else:
            err_data = [err_col] * len(df) # Konstanter Fehler
            
        val_data = df[val_col]
        
        # Zeilenweise formatieren (Header wird exakt so übernommen, wie du ihn schreibst!)
        export_data[header] = [format_latex(v, e) for v, e in zip(val_data, err_data)]
        col_headers.append("c")
        
    df_export = pd.DataFrame(export_data)
    latex_code = df_export.to_latex(index=False, escape=False, column_format="".join(col_headers))
    
    if caption or label:
        return f"\\begin{{table}}[H]\n\\centering\n\\caption{{{caption}}}\n\\label{{{label}}}\n{latex_code}\\end{{table}}\n"
    return latex_code

# =====================================================================
# 7. FITTING & PLOTTING
# =====================================================================

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
            self.chi2 = np.sum((residuals / self.y_err)**2)
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
            # Nutze unsere PAP-Formatierung!
            print(f"{name} = {format_latex(p_val, p_err)}")
        if not np.isnan(self.chi2_red):
            print(f"Fit-Güte: chi2 = {self.chi2:.2f}, ndof = {self.ndof} -> chi2/ndof = {self.chi2_red:.2f}")
        print("======================")

    def plot(self, ax=None, title="", xlabel="", ylabel="", 
                data_label="Messdaten", fit_label="Fit", 
                color_data="black", color_fit="red",
                plot_fit=True, plot_band=True, extra_points=100, x_lim=None, y_lim=None, language="de"):
        """Plottet Daten, Fit und Fehlerband. Optimiert für große Datensätze (>10k Punkte)."""
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 5))
            
        # --- PERFORMANCE OPTIMIERUNG FÜR GROßE DATENSÄTZE ---
        N = len(self.x_data)
        if N > 1000:
            fmt, alpha, markersize = ',', 0.5, 1  # Pixel-Marker
            err_every = max(1, N // 50)           # Zeige nur ~50 Fehlerbalken
            raster = True                         # Verhindert riesige PDF-Dateien
        else:
            fmt, alpha, markersize = 'x', 1.0, 5
            err_every = 1
            raster = False

        # 1. Daten plotten
        ax.errorbar(self.x_data, self.y_data, 
                    xerr=self.x_err, yerr=self.y_err, 
                    fmt=fmt, color=color_data, label=data_label, 
                    capsize=3 if N<=1000 else 0, alpha=alpha, markersize=markersize,
                    errorevery=err_every, rasterized=raster, zorder=5)
        
        # 2. Limits anwenden
        if x_lim is not None: ax.set_xlim(x_lim)
        if y_lim is not None: ax.set_ylim(y_lim)
            
        # 3. Fit und Fehlerband
        if plot_fit:
            current_xlim = ax.get_xlim()
            x_smooth = np.linspace(current_xlim[0], current_xlim[1], extra_points)
            y_smooth = self.evaluate(x_smooth)
            ax.plot(x_smooth, y_smooth, '-', color=color_fit, label=fit_label, zorder=4)
            
            # Fehlerband via Monte-Carlo
            if plot_band and self.cov_matrix is not None and not np.isinf(self.cov_matrix).all():
                try:
                    samples = np.random.multivariate_normal(self.params, self.cov_matrix, 1000)
                    y_samples = np.array([self.func(x_smooth, *p) for p in samples])
                    y_lower = np.percentile(y_samples, 15.87, axis=0)
                    y_upper = np.percentile(y_samples, 84.13, axis=0)
                    if language.casefold() == "de":
                        ax.fill_between(x_smooth, y_lower, y_upper, color=color_fit, alpha=0.2, 
                                        label=r"1$\sigma$ Fehlerband", zorder=3)
                    elif language.casefold() == "en":
                        ax.fill_between(x_smooth, y_lower, y_upper, color=color_fit, alpha=0.2, 
                                                                label=r"1$\sigma$ Error Band", zorder=3)
                    else:
                        raise NotImplementedError("Sprache nicht supported: use 'de' or 'en'.")
                except ValueError:
                    pass
            ax.set_xlim(current_xlim)
            
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend()
        return ax

def fit_curve(func, df_or_dict, x_col, y_col, x_err_col=None, y_err_col=None, p0=None):
    """
    Führt einen Curve-Fit durch. Falls y_err gegeben ist, wird ein gewichteter Fit gemacht!
    Gibt ein FitResult-Objekt zurück.
    
    Beispiel:
    def linear(x, a, b): return a*x + b
    res = fit_curve(linear, df, "Zeit", "Strecke", y_err_col=0.05)
    """
    # Daten extrahieren
    x_data = df_or_dict[x_col]
    y_data = df_or_dict[y_col]
    
    # Fehler extrahieren (falls als String Spaltenname, sonst als konstanter Wert annehmen)
    x_err = None
    if x_err_col is not None:
        if isinstance(x_err_col, str): x_err = df_or_dict[x_err_col]
        else: x_err = np.full(len(x_data), x_err_col)
        
    y_err = None
    if y_err_col is not None:
        if isinstance(y_err_col, str): y_err = df_or_dict[y_err_col]
        else: y_err = np.full(len(y_data), y_err_col)
        
    # Wenn y_err existiert, machen wir einen gewichteten Fit
    # absolute_sigma=True ist für korrekte Fehler in pcov
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
    
    # Sortieren nach x
    idx = np.argsort(x)
    x, y, dy = x[idx], y[idx], dy[idx]
    
    # Steilste Gerade: (y_end + dy_end) minus (y_start - dy_start)
    m_max = ((y[-1] + dy[-1]) - (y[0] - dy[0])) / (x[-1] - x[0])
    # Flachste Gerade: (y_end - dy_end) minus (y_start + dy_start)
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