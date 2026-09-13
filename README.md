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

Für die Arbeit am `package-prep`-Branch:

```bash
git switch package-prep
python -m pip install -e ".[test]"
```

Danach kann die Toolbox unabhängig vom aktuellen Arbeitsordner importiert werden:

```python
import pap_toolbox as pap
```

Durch die editable Installation (`-e`) werden Änderungen an `pap_toolbox.py` direkt von der installierten Umgebung verwendet; eine Neuinstallation nach jeder Codeänderung ist nicht nötig.

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
