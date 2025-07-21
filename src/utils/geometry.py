import numpy as np

from src import config


def get_intersection_or_none(
    pos: np.ndarray, direction: np.ndarray, radius: float = config.LUNAR_RADIUS_KM
) -> np.ndarray | None:
    """
    Find the intersection of a ray with a sphere (e.g., lunar surface).

    Args:
        pos: Starting position of the ray (3D vector)
        direction: Direction vector of the ray (3D vector)
        radius: Radius of the sphere to intersect with (default: lunar radius)

    Returns:
        Intersection point as 3D vector, or None if no intersection
    """
    # Normalize direction just in case
    v = direction / np.linalg.norm(direction)
    p = pos

    # Solve quadratic equation: |p + t*v|² = radius²
    a = np.dot(v, v)
    b = 2 * np.dot(p, v)
    c = np.dot(p, p) - radius**2

    discriminant = b**2 - 4 * a * c

    if discriminant < 0:
        return None  # No intersection

    sqrt_disc = np.sqrt(discriminant)
    t1 = (-b - sqrt_disc) / (2 * a)
    t2 = (-b + sqrt_disc) / (2 * a)

    # We want the closest *positive* t (forward along ray)
    t_candidates = [t for t in (t1, t2) if t > 0]
    if not t_candidates:
        return None

    t = min(t_candidates)
    return p + t * v  # Intersection point
