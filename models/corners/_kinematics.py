# third-party
import numpy as np


def euler_xyz(eul: np.ndarray) -> np.ndarray:
    """Rotation matrix for an extrinsic x-y-z Euler triple -- identical to
    scipy's ``Rotation.from_euler("xyz", eul).as_matrix()`` but ~15x cheaper per
    call, which matters because the corner solvers rebuild this on every residual
    evaluation."""
    cx, cy, cz = np.cos(eul)
    sx, sy, sz = np.sin(eul)
    return np.array([
        [cy * cz,  sx * sy * cz - cx * sz,  cx * sy * cz + sx * sz],
        [cy * sz,  sx * sy * sz + cx * cz,  cx * sy * sz - sx * cz],
        [-sy,      sx * cy,                 cx * cy],
    ])


def cross3(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Cross product of two length-3 vectors. ``np.cross`` spends most of its
    time in ``moveaxis``/``normalize_axis_tuple`` bookkeeping that a 3-vector
    doesn't need; the explicit form is several times faster in the solver loop."""
    return np.array([
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ])
