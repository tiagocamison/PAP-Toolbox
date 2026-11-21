# PAP Toolbox

Dies ist ein Helper-Skript für das PAP (Physikalisches Anfängerpraktikum) an der Universität Heidelberg. In `tutorial.ipynb` findet sich eine Erklärung der Funktionen.

Aufgrund von Copyright und um Plagiate zu vermeiden (Versuche wiederholen soll nervig sein) muss dieses GitHub-Repository auf allen Abgaben verlinkt werden. Aufgrund der MIT-Lizenz ist die weitere Verwendung ansonsten frei möglich.

## Installation

### Editable Installation für Entwicklung

Repository klonen und in den Projektordner wechseln:

```bash
git clone https://github.com/tiagocamison/PAP-Toolbox.git
cd PAP-Toolbox
```

Für die lokale Entwicklung:

```bash
python -m pip install -e ".[test]"
```

Danach kann die Toolbox unabhängig vom aktuellen Arbeitsordner importiert werden:

```python
import pap_toolbox as pap
```

Durch die editable Installation (`-e`) werden Änderungen an `src/pap_toolbox/` direkt von der installierten Umgebung verwendet; eine Neuinstallation nach jeder Codeänderung ist nicht nötig.

### Normale lokale Installation

Ohne Entwicklungsabhängigkeiten und ohne editable mode:

```bash
python -m pip install .
```

## Tests

Nach Installation mit dem `test`-Extra:

```bash
python -m pytest
```

Optional mit Coverage:

```bash
python -m pytest --cov=pap_toolbox
```

## Abhängigkeiten

Die installierbaren Runtime-Abhängigkeiten werden in `pyproject.toml` verwaltet. `requirements.txt` bleibt vorerst als zusätzliche Übersicht bestehen.

## Kontakt und Beiträge

Bei Verbesserungsvorschlägen oder Bugs gerne an tiago.camison_grimbach@stud.uni-heidelberg.de wenden oder direkt ein Issue bzw. einen Pull Request öffnen.


## Lineare Fits (Convenience API)

Kurzer Fit mit Grafik **und** PAP-formatierter Ausgabe:

```python
import pap_toolbox as pap

fit = pap.linear_fit(df, "t_s", "ln_A", yerr="dln_A", show=True)
delta, ddelta = -fit.slope, fit.slope_err
```

Mit Fehlerbalken, Residuen und Einheiten:

```python
fit = pap.linear_fit(
    df, "t_s", "ln_A", yerr="dln_A",
    x_unit="s", show=True, residuals=True
)
```

Man kann Ergebnisse ohne Grafik oder Bericht bekommen (`plot=False`,
`report=False`, Standard) oder die Ausgabe mit `show=True` aktivieren.
`plot=True` und `report=True` lassen sich unabhängig setzen.
Ein optionales `plot_band=True` zeigt das bestehende Monte-Carlo-Fehlerband.

```python
# Fit nur über einen Abschnitt (Grenzen inklusive)
fit = pap.linear_fit(df, "log_f", "gain_db", yerr="dgain",
                     x_range=(2, 3), plot=True)
# Alternativ: linear_fit(x_array, y_array, yerr=0.2)
```

Ein Schnittpunkt zweier linearer Fits (z.B. asymptotische Bode-Geraden):

```python
left = pap.linear_fit(df, "log_f", "gain_db", yerr="dgain", x_range=(2, 3))
right = pap.linear_fit(df, "log_f", "gain_db", yerr="dgain", x_range=(4, 5))
crossing = pap.intersect_lines(left, right)
print(crossing.x, crossing.x_err)   # Schnittpunkt in den x-Koordinaten
freq, dfreq = crossing.as_frequency(base=10)  # nur wenn x = log10(f / Einheit)
```

Die Schnittpunkt-Unsicherheiten verwenden die vollständigen Fit-Kovarianzmatrizen.
**Die beiden Fits werden dabei als unabhängig angenommen.**
Geteilte Kalibrierungs- und systematische Fehler werden nicht automatisch berücksichtigt.

**Wichtige Einschränkung:** `xerr` erzeugt Fehlerbalken im Plot, wird aber nicht
in der Regression berücksichtigt. Die Regression nutzt `scipy.optimize.curve_fit`
mit optionalen `yerr`-Gewichten (`absolute_sigma=True`). Für
signifikante x-Unsicherheiten oder korrelierte Fehler ist eine andere Methode nötig.
Die Toolbox führt keine Logarithmierung oder andere Daten-Transformation automatisch durch;
die physikalisch korrekte Fehlerfortpflanzung der transformierten Daten bleibt explizit.
