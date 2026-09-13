import numpy as np


def series_stats(data, print_results=False):
    """
    Nimmt ein Array von Messwerten (z.B. 10 Schwingungsdauern) und berechnet:
    - N (Anzahl)
    - mean (Mittelwert)
    - std (Empirische Standardabweichung der Einzelmessung)
    - sem (Standardfehler des Mittelwerts = std / sqrt(N)) -> Fehler fürs Protokoll
    """
    arr = np.asarray(data)
    arr = arr[~np.isnan(arr)]

    N = len(arr)
    if N == 0:
        print("Warning: no valid values passed. Return nan.")
        return {"N": 0, "mean": np.nan, "std": np.nan, "sem": np.nan}

    mean = np.mean(arr)
    std = np.std(arr, ddof=1) if N > 1 else 0.0
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
