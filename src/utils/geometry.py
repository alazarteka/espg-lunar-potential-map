import numpy as np

from src.utils.units import LengthType, ureg


def get_intersection_or_none(
    pos: np.ndarray, direction: np.ndarray, radius: LengthType
) -> np.ndarray | None:
    """
    Find the intersection of a ray with a sphere (e.g., lunar surface).

    Args:
        pos: Starting position of the ray (3D vector)
        direction: Direction vector of the ray (3D vector)
        radius: Radius of the sphere to intersect with (unit: kilometers)

    Returns:
        Intersection point as 3D vector, or None if no intersection
    """
    # Normalize direction just in case
    v = direction / np.linalg.norm(direction)
    p = pos

    # Solve quadratic equation: |p + t*v|² = radius²
    a = np.dot(v, v)
    b = 2 * np.dot(p, v)
    c = np.dot(p, p) - radius.to(ureg.kilometer).magnitude ** 2

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


def get_intersections_or_none_batch(
    pos: np.ndarray, direction: np.ndarray, radius: LengthType
) -> tuple[np.ndarray, np.ndarray]:
    """
    Vectorized ray–sphere intersections for many rays.

    Args:
        pos: Array of starting positions, shape (N, 3).
        direction: Array of direction vectors, shape (N, 3).
        radius: Sphere radius (kilometers).

    Returns:
        points: Array of intersection points, shape (N, 3) with NaNs where no
            intersection.
        mask: Boolean array (N,) True where a valid forward intersection exists.
    """
    if pos.ndim != 2 or pos.shape[1] != 3:
        raise ValueError("pos must have shape (N, 3)")
    if direction.ndim != 2 or direction.shape[1] != 3:
        raise ValueError("direction must have shape (N, 3)")
    if pos.shape[0] != direction.shape[0]:
        raise ValueError("pos and direction must have the same leading dimension")

    p = pos.astype(float, copy=False)
    v_raw = direction.astype(float, copy=False)

    # Normalize directions row-wise; guard against zero-norm vectors
    norms = np.linalg.norm(v_raw, axis=1)
    valid_dir = norms > 0
    # Avoid division by zero by temporarily setting zeros to 1
    safe_norms = np.where(valid_dir, norms, 1.0)
    v = v_raw / safe_norms[:, None]

    r2 = float(radius.to(ureg.kilometer).magnitude) ** 2

    # Quadratic coefficients per row for |p + t*v|^2 = r^2
    a = np.einsum("ij,ij->i", v, v)
    b = 2.0 * np.einsum("ij,ij->i", p, v)
    c = np.einsum("ij,ij->i", p, p) - r2

    discriminant = b * b - 4.0 * a * c
    has_real = discriminant >= 0.0

    # Compute roots where discriminant is non-negative; clip to avoid NaN in sqrt
    sqrt_disc = np.sqrt(np.clip(discriminant, 0.0, None))
    denom = 2.0 * a
    # Avoid division by zero
    nonzero_a = denom != 0.0

    t1 = np.where(nonzero_a, (-b - sqrt_disc) / denom, np.nan)
    t2 = np.where(nonzero_a, (-b + sqrt_disc) / denom, np.nan)

    # Choose smallest positive t
    t1_pos = t1 > 0.0
    t2_pos = t2 > 0.0
    t = np.where(
        t1_pos & t2_pos,
        np.minimum(t1, t2),
        np.where(t1_pos, t1, np.where(t2_pos, t2, np.nan)),
    )

    valid = valid_dir & has_real & np.isfinite(t)

    points = np.full_like(p, np.nan, dtype=float)
    if np.any(valid):
        points[valid] = p[valid] + t[valid, None] * v[valid]

    return points, valid
