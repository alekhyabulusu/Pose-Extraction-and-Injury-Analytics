import numpy as np


def angle_between(a: np.ndarray, vertex: np.ndarray, b: np.ndarray) -> float:
    """Interior angle at `vertex` formed by rays vertex→a and vertex→b.

    Args:
        a:      (2,) proximal point
        vertex: (2,) the joint being measured
        b:      (2,) distal point

    Returns:
        Angle in degrees [0, 180].
    """
    u = a - vertex
    v = b - vertex
    nu, nv = np.linalg.norm(u), np.linalg.norm(v)
    if nu < 1e-9 or nv < 1e-9:
        return 0.0
    cos_theta = np.clip(np.dot(u, v) / (nu * nv), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_theta)))


def angle_to_vertical(a: np.ndarray, b: np.ndarray) -> float:
    """Angle of segment a→b relative to the vertical (Y) axis.

    0° = perfectly upright. 90° = horizontal.

    Args:
        a: (2,) proximal point (e.g. hip)
        b: (2,) distal point  (e.g. shoulder)

    Returns:
        Angle in degrees [0, 90].
    """
    vec = b - a
    norm = np.linalg.norm(vec)
    if norm < 1e-9:
        return 0.0
    vertical = np.array([0.0, 1.0])
    cos_theta = np.clip(np.dot(vec / norm, vertical), -1.0, 1.0)
    return float(np.degrees(np.arccos(abs(cos_theta))))


def midpoint(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """2D midpoint between two points."""
    return (a + b) / 2.0


def unit_vec(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Unit vector from a to b. Returns zero vector if degenerate."""
    v = b - a
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else np.zeros(2)
