import numpy as np


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
