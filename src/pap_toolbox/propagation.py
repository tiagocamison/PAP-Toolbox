import numpy as np
import sympy as sp
from IPython.display import Math, display


class ErrorPropagator:
    def __init__(self, formula_str, all_positive=True):
        """
        Initialisiert den Propagator mit einer als String übergebenen Formel.
        Beispiel: ErrorPropagator("m_w * c_w * (T_1 - T_bar) / (T_bar - T_2)")
        """
        parsed = sp.parsing.sympy_parser.parse_expr(formula_str)
        if all_positive:
            pos_symbols_map = {s: sp.Symbol(s.name, positive=True, real=True) for s in parsed.free_symbols}
            self.f_expr = sp.nsimplify(parsed.subs(pos_symbols_map), rational=True)
        else:
            self.f_expr = sp.nsimplify(parsed, rational=True)

        self.vars = sorted(list(self.f_expr.free_symbols), key=lambda x: x.name)
        self.err_vars = [sp.Symbol(rf"d {v.name}", positive=True, real=True) for v in self.vars]

        self.variance_terms = {}
        variance = 0
        for v, ev in zip(self.vars, self.err_vars):
            term = (sp.diff(self.f_expr, v) * ev) ** 2
            self.variance_terms[v.name] = term
            variance += term

        self.abs_err_expr = sp.nsimplify(sp.sqrt(sp.factor(variance)), rational=True)
        self.rel_err_expr = sp.nsimplify(sp.sqrt(sp.factor(variance / self.f_expr**2)), rational=True)

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

            if isinstance(c_err, str) and c_err.endswith("%"):
                rel_frac = float(c_err.strip("%")) / 100.0
                c_err = np.abs(c_val) * rel_frac
            else:
                c_err = np.asarray(c_err, dtype=float)

            val_args.append(c_val)
            err_args.append(c_err)
        return val_args, err_args

    def _get_latex(self, expr):
        """Hilfsfunktion: Ersetzt intern die Fehler-Symbole durch sauberes Delta für LaTeX."""
        subs_dict = {ev: sp.Symbol(rf"\Delta {v.name}") for v, ev in zip(self.vars, self.err_vars)}
        return sp.latex(expr.subs(subs_dict))

    def show_math(self):
        """Rendert die theoretischen Formeln im Jupyter Notebook."""
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
            var_array = term_func(*val_args, *err_args)
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

            if isinstance(current_err, str) and current_err.endswith("%"):
                rel_frac = float(current_err.strip("%")) / 100.0
                current_err = np.abs(current_val) * rel_frac
            else:
                current_err = np.asarray(current_err, dtype=float)

            val_args.append(current_val)
            err_args.append(current_err)

        calc_val = self.val_func(*val_args)
        calc_err = self.err_func(*val_args, *err_args)
        return calc_val, calc_err

    def print_latex(self):
        err_latex = sp.latex(self.abs_err_expr)
        for v in self.vars:
            err_latex = err_latex.replace(f"d{v.name}", rf"\Delta {sp.latex(v)}")
        display(Math(rf"f = {sp.latex(self.f_expr)} \quad \Longrightarrow \quad \Delta f = {err_latex}"))
