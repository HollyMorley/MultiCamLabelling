import matplotlib.pyplot as plt
import numpy as np
import cv2


def triangulate(points_2d, projection_matrices):
    """Linear (DLT) triangulation of a 3D point from N >= 2 camera views.

    points_2d:           (N, 2) array of 2D points, one per camera
    projection_matrices: (N, 3, 4) array of projection matrices

    Returns the homogeneous 3D point (4,), normalised so the last entry is 1.
    """
    points_2d = np.asarray(points_2d, dtype=float)
    projection_matrices = np.asarray(projection_matrices, dtype=float)

    # Build the 2N x 4 system from the standard DLT projection constraints
    rows = []
    for (x, y), P in zip(points_2d, projection_matrices):
        rows.append(x * P[2, :] - P[0, :])
        rows.append(y * P[2, :] - P[1, :])
    A = np.stack(rows, axis=0)

    # Solution is the right singular vector with the smallest singular value
    _, _, vh = np.linalg.svd(A)
    X = vh[-1]
    return X / X[-1]


class CameraData:
    """Holds intrinsic/extrinsic data for all three cameras (side, front, overhead).

    TODO: probably a class per camera makes more sense.
    """

    def __init__(self, snapshot_paths=[], basic=True):
        if not basic:
            self.view_paths = snapshot_paths
            self.views = self.get_cameras_views()

        self.specs = self.get_cameras_specs()
        self.intrinsic_matrices = self.get_cameras_intrinsics()
        self.extrinsics_ini_guess = self.get_cameras_extrinsics_guess()

    def get_cameras_specs(self) -> dict:
        """Return a dictionary with the camera specs for each camera."""
        camera_specs = {
            "side": {
                "focal_length_mm": 16,
                "y_size_px": 230,
                "x_size_px": 1920,
                "pixel_size_x_mm": 4.8e-3,
                "pixel_size_y_mm": 4.8e-3,
                "principal_point_x_px": 960,
                "principal_point_y_px": 600,
                "crop_offset_x": 0,
                "crop_offset_y": 607,
            },
            "front": {
                "focal_length_mm": 12,
                "y_size_px": 320,
                "x_size_px": 296,
                "pixel_size_x_mm": 3.45e-3,
                "pixel_size_y_mm": 3.45e-3,
                "principal_point_x_px": 960,
                "principal_point_y_px": 600,
                "crop_offset_x": 652,
                "crop_offset_y": 477,
            },
            "overhead": {
                "focal_length_mm": 16,
                "y_size_px": 116,
                "x_size_px": 992,
                "pixel_size_x_mm": 4.8e-3,
                "pixel_size_y_mm": 4.8e-3,
                "principal_point_x_px": 960,
                "principal_point_y_px": 600,
                "crop_offset_x": 608,
                "crop_offset_y": 914,
            },
        }
        return camera_specs

    def get_cameras_views(self) -> dict:
        """Return loaded camera view images."""
        camera_views = dict()
        for cam, path in self.view_paths.items():
            camera_views[cam] = plt.imread(path)
        return camera_views

    def get_cameras_intrinsics(self) -> dict:
        """Define cameras' intrinsic matrices from technical specs data.

        Each camera intrinsic matrix corresponds to matrix A in the first equation at
        https://docs.opencv.org/4.x/d5/d1f/calib3d_solvePnP.html
        """
        camera_intrinsics = dict()
        for cam in self.specs.keys():
            fx = self.specs[cam]["focal_length_mm"] / self.specs[cam]["pixel_size_x_mm"]
            fy = self.specs[cam]["focal_length_mm"] / self.specs[cam]["pixel_size_y_mm"]

            cx = self.specs[cam].get("principal_point_x_px", self.specs[cam]["x_size_px"] / 2.0)
            cy = self.specs[cam].get("principal_point_y_px", self.specs[cam]["y_size_px"] / 2.0)

            offset_x = self.specs[cam].get("crop_offset_x", 0)
            offset_y = self.specs[cam].get("crop_offset_y", 0)
            cx -= offset_x
            cy -= offset_y

            camera_intrinsics[cam] = np.array([
                [fx, 0.0, cx],
                [0.0, fy, cy],
                [0.0, 0.0, 1.0],
            ])
        return camera_intrinsics

    def get_cameras_extrinsics_guess(self) -> dict:
        """Define an initial guess for the cameras' extrinsic matrices.

        Each camera extrinsic matrix corresponds to matrix T_w in the first equation at
        https://docs.opencv.org/4.x/d5/d1f/calib3d_solvePnP.html
        """
        box_length = 470  # mm
        box_width = 53.5  # mm
        box_height = 70  # mm

        tvec_guess = dict()
        tvec_guess["side"] = np.array([-box_length / 2, box_height / 2, 1050]).reshape(-1, 1)
        tvec_guess["front"] = np.array([-box_width / 2, box_height / 2, box_length + 760]).reshape(-1, 1)
        tvec_guess["overhead"] = np.array([-box_length / 2, box_width / 2, box_height + 1330]).reshape(-1, 1)

        # fmt: off
        rot_m_guess = dict()
        rot_m_guess['side'] = np.array([
            [1.0, 0.0, 0.0,],
            [0.0, 0.0, 1.0,],
            [0.0, -1.0, 0.0,],
        ])
        rot_m_guess['front'] = np.array([
            [0.0, 0.0, -1.0,],
            [1.0, 0.0, 0.0,],
            [0.0, -1.0, 0.0,],
        ])
        rot_m_guess['overhead'] = np.array([
            [1.0, 0.0, 0.0,],
            [0.0, -1.0, 0.0,],
            [0.0, 0.0, -1.0,],
        ])
        # fmt: on

        # NOTE: transpose rotm to match opencv's definition
        cameras_extrinsics_guess = dict()
        for cam in self.specs.keys():
            rodrigues_vec_opencv, _ = cv2.Rodrigues(rot_m_guess[cam].T)
            cameras_extrinsics_guess[cam] = {
                "rotm": rot_m_guess[cam].T,
                "rvec": rodrigues_vec_opencv,
                "tvec": tvec_guess[cam],
            }
        return cameras_extrinsics_guess

    def compute_cameras_extrinsics(
        self,
        belt_coords_WCS: np.ndarray,
        belt_coords_CCS: dict,
        use_extrinsics_ini_guess: bool = False,
    ) -> dict:
        """Apply solvePnP algorithm to estimate the extrinsic matrix for each camera."""
        camera_extrinsics = dict()

        for cam in self.specs.keys():
            if not use_extrinsics_ini_guess:
                retval, rvec, tvec = cv2.solvePnP(
                    belt_coords_WCS,
                    belt_coords_CCS[cam],
                    self.intrinsic_matrices[cam],
                    np.array([]),
                    flags=cv2.SOLVEPNP_ITERATIVE,
                )
            else:
                retval, rvec, tvec = cv2.solvePnP(
                    belt_coords_WCS,
                    belt_coords_CCS[cam],
                    self.intrinsic_matrices[cam],
                    np.array([]),
                    rvec=self.extrinsics_ini_guess[cam]["rvec"].copy(),
                    tvec=self.extrinsics_ini_guess[cam]["tvec"].copy(),
                    useExtrinsicGuess=True,
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
    """Holds data for the selected belt calibration points."""

    def __init__(self, belt_coords, views):
        self.views = list(views)
        self.points_str2int = {
            "StartPlatR": 0,
            "StartPlatL": 3,
            "TransitionR": 1,
            "TransitionL": 2,
            "Door": 4,
            "StepR": 5,
            "StepL": 6,
        }
        self.fn_points_str2int = np.vectorize(lambda x: self.points_str2int[x])

        self.coords_CCS = self.get_points_in_CCS(belt_coords)
        self.coords_WCS = self.get_points_in_WCS()

    def get_points_in_WCS(self) -> np.ndarray:
        """Express belt points in the world coordinate system (mm)."""
        return np.array([
            [0.0, 0.0, 0.0],
            [470.0, 0.0, 0.0],
            [470.0, 53.5, 0.0],
            [0.0, 53.5, 0.0],
            [-8.2, 26.0, 48.5],
            [0.0, 0.0, 5.0],
            [0.0, 53.5, 5.0],
        ])

    def get_points_in_CCS(self, df) -> dict:
        """Express points in each camera's coordinate system."""
        points_str_in_input_order = np.array(df.loc[df["coords"] == "x"]["bodyparts"])
        points_IDs_in_input_order = self.fn_points_str2int(points_str_in_input_order)
        sorted_idcs_by_pt_ID = np.argsort(points_IDs_in_input_order)

        sorted_kys = sorted(
            self.points_str2int.keys(), key=lambda ky: self.points_str2int[ky]
        )
        assert all(points_str_in_input_order[sorted_idcs_by_pt_ID] == sorted_kys)

        belt_coords_CCS = dict()
        for cam in self.views:
            imagePoints = np.array(
                [df.loc[df["coords"] == "x"][cam], df.loc[df["coords"] == "y"][cam]]
            ).T
            belt_coords_CCS[cam] = imagePoints[sorted_idcs_by_pt_ID, :]

        return belt_coords_CCS

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
