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

**Messunsicherheiten in beiden Achsen:** Mit `xerr` und `yerr` minimiert
`linear_fit` die orthogonale Errors-in-Variables-Fehlerquadratsumme
`χ² = Σ (y - m*x - b)² / (yerr² + m²*xerr²)`.
Das umfasst die Streuung in *beiden* Achsen. Die Fit-Kovarianz wird
näherungsweise aus dem Jacobian (Gauss-Newton) berechnet. Mit ausschließlich
`yerr` bleibt der bisherige `curve_fit` mit `absolute_sigma=True` erhalten.
Mit ausschließlich `xerr` wird bei strikt positiven x-Fehlern eine inverse
gewichtete Regression verwendet (y wird als fehlerfrei angenommen).
`fit.chi2`, `fit.chi2_red` und `fit.pulls` beziehen die x-Fehler mit ein.
Korrelationen, gemeinsame Eichfehler oder Fehler der gewählten
Daten-Transformation werden *nicht* automatisch berücksichtigt.
Die Toolbox führt keine Logarithmierung oder andere Daten-Transformation automatisch durch;
die physikalisch korrekte Fehlerfortpflanzung der transformierten Daten bleibt explizit.


Beispiel mit Unsicherheiten in beiden Koordinaten:

```python
fit = pap.linear_fit(df, "U", "I", xerr="dU", yerr="dI", show=True)
print(fit.fit_method, fit.chi2_red)
```


## Fitbereiche, Parametergrenzen und Residuen

Beide Fitfunktionen akzeptieren `x_range=(lo, hi)` (ein Intervall) oder
`fitranges=[(lo1, hi1), (lo2, hi2)]` (Vereinigung mehrerer Intervalle für
**einen** gemeinsamen Fit; Endpunkte inklusive, keine doppelten Messpunkte).
Die Auswahl betrifft Fit, Fehlerarrays, Chi-Quadrat und Residuen; der Plot
zeigt die ausgewählten Daten und die Fitkurve über die dargestellte x-Achse.
`x_lim` bleibt eine reine Plotgrenze.

```python
fit = pap.fit_curve(
    resonance, df, "f", "A", y_err_col="dA",
    p0=[1, 100, 5],
    fitranges=[(80, 95), (105, 120)],
    bounds=([0, 80, 0], [np.inf, 120, np.inf]),
)
fit.plot(residuals=True, standardized=True)
# Alternativ: fit.plot_residuals() oder fit.plot_with_residuals()
```

`fit_curve(..., show=True, residuals=True)` erzeugt Plot und Bericht direkt.
Beim allgemeinen Fit berücksichtigt die Regression weiterhin nur y-Fehler;
x-Fehler dienen ausschließlich der Darstellung. Bei `linear_fit` gehen
sie wie oben beschrieben in die Regression ein.

```python
fit = pap.linear_fit(df, "U", "I", xerr="dU", yerr="dI",
                     fit_intercept=False, bounds=(0, np.inf), show=True)
m, dm = fit.slope, fit.slope_err
```

`fit_intercept=False` fixiert b exakt auf 0; `params` und `cov_matrix`
haben dann nur einen freien Parameter. `fit.intercept` und
`fit.intercept_err` sind 0, `fit.line_cov_matrix` liefert die
vollständige (m,b)-Kovarianz für Schnittpunkte.

Bounds sind wie in SciPy `(lower, upper)`, in Parameterreihenfolge
(m, b) bzw. (m,) beim Ursprungsfit; skalare Grenzen gelten für alle freien
Parameter. Grenzen sind inklusive: für nur positive Steigung und freien
Achsenabschnitt etwa `bounds=([0, -np.inf], [np.inf, np.inf])`.
Die lokale Kovarianznäherung ist bei aktiven Grenzen **kein** verlässliches
symmetrisches Konfidenzintervall.
