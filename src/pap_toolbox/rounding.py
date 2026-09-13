from decimal import Decimal

import numpy as np
import pandas as pd


def first_digit_func(x):
    x = Decimal(str(abs(x)))

    if x == 0:
        raise ValueError("First significant digit is undefined for zero.")

    exponent = x.adjusted()
    return int(x.scaleb(-exponent)), x.adjusted()


def _get_decimals_pap(err):
    """Interne Hilfsfunktion: Ermittelt auf welche Nachkommastelle gerundet werden muss."""
    if pd.isna(err) or err == 0:
        return None

    first_digit, oom = first_digit_func(err)
    sig_figs = 2 if first_digit in [1, 2] else 1
    return sig_figs - 1 - oom


def round_pap(val, err):
    """
    Numerische PAP-Rundung. Gibt (val_rounded, err_rounded) als Floats zurück.
    Gut, falls man mit den gerundeten Werten weiterrechnen/plotten will.
    """
    if pd.isna(val) or pd.isna(err) or err == 0:
        return float(val), float(err)

    decimals = _get_decimals_pap(err)
    return round(float(val), decimals), round(float(err), decimals)


v_round_pap = np.vectorize(round_pap)


def compare_to_literature(val, err, lit_val, lit_err=0.0, verbose=False):
    """
    Berechnet die Abweichung vom Literaturwert in Vielfachen von Sigma.
    Gibt optional direkt einen Textbaustein für die Diskussion aus.
    """
    diff = abs(val - lit_val)
    sigma_total = np.sqrt(err**2 + lit_err**2)

    if sigma_total == 0:
        return "Fehler ist 0, Berechnung der Sigma-Abweichung nicht möglich."

    sigma_dist = diff / sigma_total

    if verbose:
        # Lokaler Import vermeidet eine zyklische Modulabhängigkeit:
        # latex -> rounding für Rundungsregeln, rounding -> latex nur für Ausgabe.
        from .latex import format_latex

        print("--- Vergleich mit Literaturwert ---")
        print(f"Dein Wert:   {format_latex(val, err)}")
        print(f"Literatur:   {format_latex(lit_val, lit_err)}")
        print(f"Abweichung:  {diff:.4g} (absolut)")
        print(f"Sigma-Abst.: {sigma_dist:.2f} \\sigma")
        print("\nProtokoll-Text:")
        if sigma_dist <= 1.0:
            print(
                f"Der Messwert weicht um {sigma_dist:.2f} \\sigma vom Literaturwert ab "
                "und stimmt im Rahmen der Messgenauigkeit hervorragend überein."
            )
        elif sigma_dist <= 3.0:
            print(
                f"Der Messwert weicht um {sigma_dist:.2f} \\sigma vom Literaturwert ab. "
                "Die Übereinstimmung ist zufriedenstellend, deutet aber auf leichte "
                "systematische Fehler hin."
            )
        else:
            print(
                f"Achtung! Der Messwert weicht um {sigma_dist:.2f} \\sigma vom Literaturwert ab "
                "(> 3 Sigma). Hier muss im Protokoll ein systematischer Fehler diskutiert werden!"
            )

    return sigma_dist


v_compare_to_literature = np.vectorize(compare_to_literature)
