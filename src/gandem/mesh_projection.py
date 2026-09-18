import cv2
import numpy as np

_PROJECTION_RESOLUTION = 2048


def project_onto_axes(
    mesh,  # Type hint: trimesh.Geometry
    axes: list[str] = ["x", "y", "z"],
) -> list[np.ndarray]:
    """Rasterize a mesh's XY, YZ, and XZ outlines in mesh coordinates.

    Outlines are returned at full resolution. Tessellation-invariant
    measurement is the metric's business, not the projector's -- see
    :func:`~gandem.evopop.shape_factors.compute_sphericity`.
    """
    outlines = []
    triangles = mesh.triangles
    axis_map = {
        "x": [1, 2],
        "y": [0, 2],
        "z": [0, 1],
    }

    for axis in axes:
        projected = triangles[:, :, axis_map[axis]]
        first_edge = projected[:, 1] - projected[:, 0]
        second_edge = projected[:, 2] - projected[:, 0]
        twice_area = np.abs(
            first_edge[:, 0] * second_edge[:, 1] - first_edge[:, 1] * second_edge[:, 0]
        )
        projected = projected[twice_area > 2e-9]
        if not len(projected):
            outlines.append(np.zeros((0, 2)))
            continue

        minimum = projected.min(axis=(0, 1))
        extent = np.ptp(projected.reshape(-1, 2), axis=0).max()
        scale = (_PROJECTION_RESOLUTION - 3) / extent
        pixels = np.rint((projected - minimum) * scale + 1).astype(np.int32)

        mask = np.zeros(
            (_PROJECTION_RESOLUTION, _PROJECTION_RESOLUTION), dtype=np.uint8
        )
        for triangle in pixels:
            cv2.fillConvexPoly(mask, triangle, 255)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            outlines.append(np.zeros((0, 2)))
            continue

        outline = cv2.approxPolyDP(
            max(contours, key=cv2.contourArea), epsilon=1.0, closed=True
        ).reshape(-1, 2)
        if not np.array_equal(outline[0], outline[-1]):
            outline = np.vstack((outline, outline[0]))
        outlines.append((outline - 1) / scale + minimum)

    return outlines
