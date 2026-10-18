import numpy as np
import pandas as pd

from .rounding import _get_decimals_pap, round_pap


def _is_numeric(val):
    """Prüft, ob ein Wert eine Zahl ist (Float/Int)."""
    try:
        float(val)
        return True
    except (ValueError, TypeError):
        return False


def format_latex(val, err=None):
    """
    Rundet numerische Werte nach PAP-Regeln.
    Reine Textfelder oder Zahlen ohne Fehler werden sauber ohne \\pm formatiert.
    """
    if pd.isna(val):
        return ""

    if isinstance(val, str) and not _is_numeric(val):
        return str(val)

    if err is None or pd.isna(err) or err == 0 or str(err).strip().lower() in ["none", "nan", ""]:
        try:
            f_val = float(val)
            return f"{int(f_val)}" if f_val.is_integer() else f"{f_val}"
        except (ValueError, TypeError):
            return str(val)

    try:
        val = float(val)
        err = float(err)
    except (ValueError, TypeError):
        return str(val)

    decimals = _get_decimals_pap(err)
    v_rnd, e_rnd = round_pap(val, err)

    if decimals <= 0:
        return f"{int(v_rnd)} $\\pm$ {int(e_rnd)}"

    fmt = f".{decimals}f"
    return f"{v_rnd:{fmt}} $\\pm$ {e_rnd:{fmt}}"


v_format_latex = np.vectorize(format_latex)


def format_separate(val, err):
    """
    Gibt ein Tuple (Wert_String, Fehler_String) zurück.
    Wichtig für Tabellen mit getrennten Fehler-Spalten.
    """
    if err is None or pd.isna(err) or err == 0:
        return f"{val}", ""

    decimals = _get_decimals_pap(err)
    v_rnd, e_rnd = round_pap(val, err)

    if decimals <= 0:
        return f"{int(v_rnd)}", f"{int(e_rnd)}"

    fmt = f".{decimals}f"
    return f"{v_rnd:{fmt}}", f"{e_rnd:{fmt}}"


def export_pap_table(df, config, caption="", label=""):
    """
    Exportiert ein DataFrame als saubere LaTeX Tabelle.

    config akzeptiert flexibel 2 oder 3 Einträge pro Spalte:
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

        if err_col is None or str(err_col).strip().lower() in ["none", "nan", ""]:
            err_data = [None] * len(df)
        elif isinstance(err_col, str) and err_col in df.columns:
            err_data = df[err_col]
        else:
            err_data = [err_col] * len(df)

        val_data = df[val_col]
        export_data[header] = [format_latex(v, e) for v, e in zip(val_data, err_data)]
        col_headers.append("c")

    df_export = pd.DataFrame(export_data)
    latex_code = df_export.to_latex(index=False, escape=False, column_format="".join(col_headers))

    if caption or label:
        return (
            f"\\begin{{table}}[H]"
            f"\\centering"
            f"\\caption{{{caption}}}"
            f"\\label{{{label}}}"
            f"{latex_code}\\end{{table}}"
        )
    return latex_code
