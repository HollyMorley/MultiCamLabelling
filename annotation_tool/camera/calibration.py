"""Camera pose estimation pipeline.

Reads camera intrinsics from project.yaml. Given labelled landmarks
(in image space, per camera), runs solvePnP per camera to recover
extrinsics. The result is the "initial" calibration; downstream
optimisation (camera/optimisation.py) can then refine the extrinsics
by minimising reprojection error against body-part observations.
"""

import cv2
import numpy as np


class CameraData:
    """Each camera's intrinsic matrix and the routine that recovers its
    extrinsics via solvePnP.

    Intrinsics are built at construction from project.intrinsics in
    project.yaml. Extrinsics need landmark positions to solve for - those
    are passed in to compute_cameras_extrinsics by InitialCalibration.
    """

    def __init__(self, project):
        self.project = project
        self.specs = project.require_intrinsics()
        self.intrinsic_matrices = self.get_cameras_intrinsics()

    def get_cameras_intrinsics(self) -> dict:
        """Build each camera's intrinsic matrix from its spec block.

        Each camera intrinsic matrix corresponds to matrix A in the first
        equation at https://docs.opencv.org/4.x/d5/d1f/calib3d_solvePnP.html
        """
        camera_intrinsics = dict()
        for cam in self.project.views:
            spec = self.specs[cam]
            fx = spec["focal_length_mm"] / spec["pixel_size_x_mm"]
            fy = spec["focal_length_mm"] / spec["pixel_size_y_mm"]

            cx = spec.get("principal_point_x_px", spec["x_size_px"] / 2.0)
            cy = spec.get("principal_point_y_px", spec["y_size_px"] / 2.0)

            cx -= spec.get("crop_offset_x", 0)
            cy -= spec.get("crop_offset_y", 0)

            camera_intrinsics[cam] = np.array([
                [fx, 0.0, cx],
                [0.0, fy, cy],
                [0.0, 0.0, 1.0],
            ])
        return camera_intrinsics

    def compute_cameras_extrinsics(
        self,
        coords_WCS: np.ndarray,
        coords_CCS: dict,
    ) -> dict:
        """Apply solvePnP to estimate the extrinsic matrix for each camera."""
        camera_extrinsics = dict()

        for cam in self.project.views:
            # solvePnP returns the camera's rotation (as a 3-element axis-angle
            # vector, rvec) and translation (tvec).
            retval, rvec, tvec = cv2.solvePnP(
                coords_WCS,
                coords_CCS[cam],
                self.intrinsic_matrices[cam],
                np.array([]),
                flags=cv2.SOLVEPNP_ITERATIVE,
            )
            assert retval, f"solvePnP failed for camera {cam!r}"

            # Convert the rotation vector into a 3x3 rotation matrix.
            rotm, _ = cv2.Rodrigues(rvec)
            camera_pose_full = np.vstack(
                [np.hstack([rotm, tvec]), np.flip(np.eye(1, 4))]
            )

            coords_CCS_repr, _ = cv2.projectPoints(
                coords_WCS, rvec, tvec,
                self.intrinsic_matrices[cam], np.array([]),
            )
            coords_CCS_repr = np.squeeze(coords_CCS_repr)
            error = np.mean(
                np.linalg.norm(coords_CCS_repr - coords_CCS[cam], axis=1)
            )

            camera_extrinsics[cam] = {
                "retval": retval,
                "rvec": rvec,
                "tvec": tvec,
                "rotm": rotm,
                "full": camera_pose_full,
                "repr_err": error,
            }

        return camera_extrinsics


class CalibrationLandmarks:
    """The matched 3D-2D point sets used during pose recovery - each
    landmark's world position paired with its labelled image position in
    every camera.

    World coordinates come from project.yaml; image coordinates come from
    the labelled-points CSV. Both arrays are built in the same order so
    they line up for solvePnP.
    """

    def __init__(self, observations_df, views, calibration_label_coordinates):
        self.views = list(views)
        self.label_coords = calibration_label_coordinates
        self.coords_WCS = self.get_points_in_WCS()
        self.coords_CCS = self.get_points_in_CCS(observations_df)

    def get_points_in_WCS(self) -> np.ndarray:
        """Stack the per-label (x, y, z) coords in label_coords order."""
        return np.array(
            [list(self.label_coords[label]) for label in self.label_coords],
            dtype=float,
        )

    def get_points_in_CCS(self, df) -> dict:
        """Per-camera (N, 2) array of pixel coords, rows in label_coords order.

        df is the calibration CSV: long-form with 'bodyparts', 'coords' (x/y),
        and one column per camera view.
        """
        coords_CCS: dict = {cam: [] for cam in self.views}
        for label in self.label_coords:
            x_row = df[(df["bodyparts"] == label) & (df["coords"] == "x")]
            y_row = df[(df["bodyparts"] == label) & (df["coords"] == "y")]
            for cam in self.views:
                coords_CCS[cam].append(
                    [x_row[cam].iloc[0], y_row[cam].iloc[0]]
                )
        return {cam: np.array(rows, dtype=float) for cam, rows in coords_CCS.items()}


class InitialCalibration:
    """Combines CameraData + CalibrationLandmarks to recover each camera's
    pose from the labelled calibration landmarks.

    This is the first-pass ("initial") result - downstream optimisation
    (camera/optimisation.py) can refine the extrinsics by minimising
    reprojection error against body-part labels.
    """

    def __init__(self, calibration_observations, project):
        self.project = project
        self.cameras = CameraData(project)
        self.cameras_intrinsics = self.cameras.intrinsic_matrices

        _, label_coords, _ = project.require_calibration_geometry()
        self.landmarks = CalibrationLandmarks(
            calibration_observations, project.views, label_coords,
        )

    def estimate_cams_pose(self):
        return self.cameras.compute_cameras_extrinsics(
            self.landmarks.coords_WCS, self.landmarks.coords_CCS,
        )

    def print_reprojection_errors(self, cameras_extrinsics):
        print("Reprojection errors:")
        for cam, data in cameras_extrinsics.items():
            print(f"  {cam}: {data['repr_err']}")
