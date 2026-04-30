import matplotlib.pyplot as plt
import numpy as np
import cv2


class CameraData:
    """Per-view camera intrinsics + extrinsics computed from project config.

    Reads intrinsic specs from project.intrinsics (one block per view in
    project.views). The extrinsics are computed by solvePnP against the
    landmark coordinates in BeltPoints during compute_cameras_extrinsics.
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
        belt_coords_WCS: np.ndarray,
        belt_coords_CCS: dict,
    ) -> dict:
        """Apply solvePnP to estimate the extrinsic matrix for each camera."""
        camera_extrinsics = dict()

        for cam in self.project.views:
            retval, rvec, tvec = cv2.solvePnP(
                belt_coords_WCS,
                belt_coords_CCS[cam],
                self.intrinsic_matrices[cam],
                np.array([]),
                flags=cv2.SOLVEPNP_ITERATIVE,
            )

            rotm, _ = cv2.Rodrigues(rvec)
            camera_pose_full = np.vstack(
                [np.hstack([rotm, tvec]), np.flip(np.eye(1, 4))]
            )

            belt_coords_CCS_repr, _ = cv2.projectPoints(
                belt_coords_WCS, rvec, tvec,
                self.intrinsic_matrices[cam], np.array([]),
            )
            belt_coords_CCS_repr = np.squeeze(belt_coords_CCS_repr)
            error = np.mean(
                np.linalg.norm(belt_coords_CCS_repr - belt_coords_CCS[cam], axis=1)
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


class BeltPoints:
    """Holds the calibration landmark positions in both world (WCS) and
    camera (CCS) coordinate systems.

    The canonical landmark order is the iteration order of
    `calibration_label_coordinates`. WCS and per-camera CCS arrays are built
    in that order so they line up for solvePnP.
    """

    def __init__(self, belt_coords_df, views, calibration_label_coordinates):
        self.views = list(views)
        self.label_coords = calibration_label_coordinates
        self.coords_WCS = self.get_points_in_WCS()
        self.coords_CCS = self.get_points_in_CCS(belt_coords_df)

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
        belt_coords_CCS: dict = {cam: [] for cam in self.views}
        for label in self.label_coords:
            x_row = df[(df["bodyparts"] == label) & (df["coords"] == "x")]
            y_row = df[(df["bodyparts"] == label) & (df["coords"] == "y")]
            for cam in self.views:
                belt_coords_CCS[cam].append(
                    [x_row[cam].iloc[0], y_row[cam].iloc[0]]
                )
        return {cam: np.array(rows, dtype=float) for cam, rows in belt_coords_CCS.items()}

    def plot_CCS(self, camera: CameraData):
        """Plot belt points in each camera coordinate system."""
        try:
            fig, axes = plt.subplots(len(self.views), 1)
            axes_list = list(axes) if hasattr(axes, "__len__") else [axes]
            for cam, ax in zip(camera.specs.keys(), axes_list):
                x_offset = camera.specs[cam].get("crop_offset_x", 0)
                y_offset = camera.specs[cam].get("crop_offset_y", 0)
                ax.imshow(
                    camera.views[cam],
                    extent=[x_offset, camera.views[cam].shape[1] + x_offset,
                            camera.views[cam].shape[0] + y_offset, y_offset],
                )
                ax.scatter(
                    x=self.coords_CCS[cam][:, 0] + x_offset,
                    y=self.coords_CCS[cam][:, 1] + y_offset,
                    s=50, c="r", marker="x", linewidth=0.5,
                    label=range(self.coords_CCS[cam].shape[0]),
                )
                ax.scatter(
                    x=camera.specs[cam]["principal_point_x_px"],
                    y=camera.specs[cam]["principal_point_y_px"],
                    s=50, c="b", marker="o", linewidth=0.5,
                    label=range(self.coords_CCS[cam].shape[0]),
                )
                for id in range(self.coords_CCS[cam].shape[0]):
                    ax.text(
                        x=self.coords_CCS[cam][id, 0] + x_offset,
                        y=self.coords_CCS[cam][id, 1] + y_offset,
                        s=id, c="r",
                    )
                ax.set_xlim(0, camera.specs[cam]["principal_point_x_px"] * 2)
                ax.set_ylim(0, camera.specs[cam]["principal_point_y_px"] * 2)
                ax.invert_yaxis()
                ax.set_title(cam)
                ax.set_xlabel("x (px)")
                ax.set_ylabel("y (px)")
                ax.axis("equal")
            return fig, axes
        except Exception as e:
            print(f"Error: {e}")
            return None, None

    def plot_WCS(self):
        """Plot belt points in the world coordinate system."""
        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")
        ax.scatter(
            self.coords_WCS[:, 0], self.coords_WCS[:, 1], self.coords_WCS[:, 2],
            s=50, c="r", marker=".", linewidth=0.5, alpha=1,
        )
        for id in range(self.coords_WCS.shape[0]):
            ax.text(
                self.coords_WCS[id, 0], self.coords_WCS[id, 1], self.coords_WCS[id, 2],
                s=id, c="r",
            )
        for row, col in zip(np.eye(3), ["r", "g", "b"]):
            ax.quiver(0, 0, 0, row[0], row[1], row[2],
                      color=col, length=500, arrow_length_ratio=0, normalize=True)
        ax.text(0, 0, 500, s="WCS", c="k")
        ax.set_xlabel("x (mm)")
        ax.set_ylabel("y (mm)")
        ax.set_zlabel("z (mm)")
        ax.axis("equal")
        return fig, ax


def plot_rotated_CS_in_WCS(fig, rot_cam, trans_cam):
    """Plot camera coordinate systems in the world coordinate system."""
    ax = fig.add_subplot(projection="3d")
    for row, col in zip(np.eye(3), ["r", "g", "b"]):
        ax.quiver(0, 0, 0, row[0], row[1], row[2],
                  color=col, arrow_length_ratio=0, normalize=True)
    for row, col in zip(rot_cam.T, ["r", "g", "b"]):
        ax.quiver(
            trans_cam[0], trans_cam[1], trans_cam[2],
            row[0], row[1], row[2],
            color=col, arrow_length_ratio=0, normalize=True,
            linestyle=":", linewidth=4,
        )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    return fig, ax
