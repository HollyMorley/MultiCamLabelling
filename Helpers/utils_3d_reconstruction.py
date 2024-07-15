import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import cv2
from pathlib import Path


class CameraData:
    """A class holding the data for all three cameras.

    TODO: probably a class per camera makes more sense.
    """

    def __init__(self, snapshot_paths=[], basic=True):
        if not basic:
            self.view_paths = snapshot_paths
            self.views = self.get_cameras_views()

        self.specs = self.get_cameras_specs()
        self.intrinsic_matrices = self.get_cameras_intrinsics()
        self.extrinsics_ini_guess = self.get_cameras_extrinsics_guess()
        # todo might need to add conditions here so that only use views if necessary, ie if want more than just the intrinsic matrices

    def get_cameras_specs(self) -> dict:
        """
        Return a dictionary with the camera specs for each camera.
        """
        camera_specs = {
            "side": {
                "focal_length_mm": 16,
                "y_size_px": 230,
                "x_size_px": 1920,
                "pixel_size_x_mm": 4.8e-3,  # 4.8um = 4.8e-3 mm
                "pixel_size_y_mm": 4.8e-3,  # 4.8um = 4.8e-3 mm
                "principal_point_x_px": 960,
                "principal_point_y_px": 600,
                "crop_offset_x": 0,
                "crop_offset_y": 607,
            },
            "front": {
                "focal_length_mm": 12,
                "y_size_px": 320,  # ok?
                "x_size_px": 296,  # ok?
                "pixel_size_x_mm": 3.45e-3,
                "pixel_size_y_mm": 3.45e-3,
                "principal_point_x_px": 960,
                "principal_point_y_px": 600,
                "crop_offset_x": 652,
                "crop_offset_y": 477,
            },
            "overhead": {
                "focal_length_mm": 16,
                "y_size_px": 116,  # ok?
                "x_size_px": 992,  # ok?
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
        """
        Return a dictionary with the camera specs for each camera.
        """
        camera_views = dict()
        for cam, path in self.view_paths.items():
            camera_views[cam] = plt.imread(path)
        return camera_views

    def get_cameras_intrinsics(self) -> dict:
        """
        Define the cameras' intrinsic matrices from technical specs data.

        Each camera intrinsic matrix would correspond to matrix A in the first equation at
        https://docs.opencv.org/4.x/d5/d1f/calib3d_solvePnP.html

        """
        camera_intrinsics = dict()
        for cam in self.specs.keys():
            # focal length in pixels
            fx = self.specs[cam]["focal_length_mm"] / self.specs[cam]["pixel_size_x_mm"]
            fy = self.specs[cam]["focal_length_mm"] / self.specs[cam]["pixel_size_y_mm"]

            # Origin offset in the full camera frame
            cx = self.specs[cam].get("principal_point_x_px", self.specs[cam]["x_size_px"] / 2.0)
            cy = self.specs[cam].get("principal_point_y_px", self.specs[cam]["y_size_px"] / 2.0)

            # Adjust for cropping if crop offsets are provided
            offset_x = self.specs[cam].get("crop_offset_x", 0)
            offset_y = self.specs[cam].get("crop_offset_y", 0)
            cx -= offset_x
            cy -= offset_y

            # build intrinsics matrix
            camera_intrinsics[cam] = np.array(
                [
                    [fx, 0.0, cx],
                    [0.0, fy, cy],
                    [0.0, 0.0, 1.0],
                ]
            )
        return camera_intrinsics

    def get_cameras_extrinsics_guess(self) -> dict:
        """
        Define an initial guess for the cameras' extrinsic matrices.

        Each camera extrinsic matrix would correspond to matrix $T_w$ in the first equation at
        https://docs.opencv.org/4.x/d5/d1f/calib3d_solvePnP.html

        """
        # Estimated size of captured volume
        box_length = 470  # mm
        box_width = 53.5  # mm
        box_height = 70  # mm
        #cam2panel_distance = 1000  # mm; estimated distance between a CCS origin and the closest plane in the captured volume.

        # Initial guess for the translation vector (tvec) from each camera
        # tvec: is expressed in CCS; from the origin of the CCS to the origin of the WCS.
        # I assume each camera centre is ~1m from the centre of the captured volume's closest plane.
        tvec_guess = dict()
        tvec_guess["side"] = np.array(
            [-box_length / 2, box_height / 2, 1050]#cam2panel_distance]
        ).reshape(-1, 1)
        tvec_guess["front"] = np.array(
            [-box_width / 2, box_height / 2, box_length + 760] #cam2panel_distance]
        ).reshape(-1, 1)
        tvec_guess["overhead"] = np.array(
            [-box_length / 2, box_width / 2, box_height + 1330]#cam2panel_distance]
        ).reshape(-1, 1)

        # Initial guess for the rotation matrix
        # I use my definition, that is,
        # in columns, I have the rotated WCS versors
        # NOTE: fmt: off avoids the linter from formatting the following lines
        # fmt: off
        rot_m_guess = dict()
        rot_m_guess['side'] = np.array(
            [
                [1.0, 0.0, 0.0,],
                [0.0, 0.0, 1.0,],
                [0.0, -1.0, 0.0,],
            ]
        )
        rot_m_guess['front'] = np.array(
            [
                [0.0, 0.0, -1.0,],
                [1.0, 0.0, 0.0,],
                [0.0, -1.0, 0.0,],
            ]
        )
        rot_m_guess['overhead'] = np.array(
            [
                [1.0, 0.0, 0.0,],
                [0.0, -1.0, 0.0,],
                [0.0, 0.0, -1.0,],
            ]
        )
        # fmt: on
        # NOTE: fmt: on reactivates the linter for the following lines

        # Build initial guess for the extrinsic matrix [R|t]
        # NOTE: I need to transpose rotm to match opencv's definition
        cameras_extrinsics_guess = dict()
        for cam in self.specs.keys():
            # Compute Rodrigues vector on rotm with opencv convention
            rodrigues_vec_opencv, _ = cv2.Rodrigues(rot_m_guess[cam].T)

            # Save parameters
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
        """
        Apply solvePnP algorithm to estimate the extrinsic matrix for each camera.
        """
        # initialise dict with extrinsics
        camera_extrinsics = dict()

        # for every camera
        for cam in self.specs.keys():
            # if no guess for the intrinsic matrices is provided
            if not use_extrinsics_ini_guess:
                retval, rvec, tvec = cv2.solvePnP(
                    belt_coords_WCS,
                    belt_coords_CCS[cam],
                    self.intrinsic_matrices[cam],
                    np.array([]),  # no distorsion
                    flags=cv2.SOLVEPNP_ITERATIVE,
                )
            # else: use the initial guess
            else:
                retval, rvec, tvec = cv2.solvePnP(
                    belt_coords_WCS,
                    belt_coords_CCS[cam],
                    self.intrinsic_matrices[cam],
                    np.array([]),  # no distorsion
                    rvec=self.extrinsics_ini_guess[cam]["rvec"].copy(),
                    tvec=self.extrinsics_ini_guess[cam]["tvec"].copy(),
                    useExtrinsicGuess=True,
                    flags=cv2.SOLVEPNP_ITERATIVE,
                )

            # build the full extrinsic matrix [R|t]
            rotm, _ = cv2.Rodrigues(
                rvec
            )  # transform Rodrigues vector to opencv rotation matrix
            camera_pose_full = np.vstack(
                [np.hstack([rotm, tvec]), np.flip(np.eye(1, 4))]
            )

            # compute reprojection error
            belt_coords_CCS_repr, _ = cv2.projectPoints(
                belt_coords_WCS,
                rvec,
                tvec,
                self.intrinsic_matrices[cam],
                np.array([]),  # we assume no distorsion
            )
            belt_coords_CCS_repr = np.squeeze(belt_coords_CCS_repr)
            error = np.mean(
                np.linalg.norm(belt_coords_CCS_repr - belt_coords_CCS[cam], axis=1)
            )

            # add results to dict
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
    """A class to hold the data of the selected beltpoints"""

    def __init__(self, belt_coords):
        # map string to integer label for each point
        self.points_str2int = {
            "StartPlatR": 0,
            "StartPlatL": 3,
            "TransitionR": 1,
            "TransitionL": 2,
            "Door": 4,
            'StepR': 5,
            'StepL': 6,
        }
        self.fn_points_str2int = np.vectorize(lambda x: self.points_str2int[x])

        self.coords_CCS = self.get_points_in_CCS(belt_coords)
        self.coords_WCS = self.get_points_in_WCS()

    def get_points_in_WCS(self) -> np.ndarray:
        """
        Express belt points in the WCS.
        """
        return np.array(
            [
                [0.0, 0.0, 0.0],
                [470.0, 0.0, 0.0],
                [470.0, 53.5, 0.0],
                [0.0, 53.5, 0.0],
                [-8.2, 26.0, 48.5],
                [0.0, 0.0, 5.0],
                [0.0, 53.5, 5.0],
            ]
        )

    def get_points_in_CCS(self, df) -> dict:
        """
        Express points in each camera's coord system.
        """
        # compute idcs to sort points by ID
        points_str_in_input_order = np.array(df.loc[df["coords"] == "x"]["bodyparts"])
        points_IDs_in_input_order = self.fn_points_str2int(points_str_in_input_order)
        sorted_idcs_by_pt_ID = np.argsort(points_IDs_in_input_order)

        sorted_kys = sorted(
            self.points_str2int.keys(), key=lambda ky: self.points_str2int[ky]
        )
        assert all(points_str_in_input_order[sorted_idcs_by_pt_ID] == sorted_kys)

        # loop thru camera views and save points
        list_cameras = list(df.columns[-3:])
        belt_coords_CCS = dict()
        for cam in list_cameras:
            imagePoints = np.array(
                [df.loc[df["coords"] == "x"][cam], df.loc[df["coords"] == "y"][cam]]
            ).T

            # sort them by point ID
            belt_coords_CCS[cam] = imagePoints[sorted_idcs_by_pt_ID, :]

        return belt_coords_CCS

    def plot_CCS(self, camera: CameraData):
        """
        Plot belt points in each CCS.        """
        try:
            # Check belt points in CCS (1-3)
            fig, axes = plt.subplots(2, 2)

            for cam, ax in zip(camera.specs.keys(), axes.reshape(-1)):
                # add image
                #ax.imshow(camera.views[cam])
                x_offset = camera.specs[cam].get("crop_offset_x", 0)
                y_offset = camera.specs[cam].get("crop_offset_y", 0)

                # add image with crop offset
                ax.imshow(camera.views[cam], extent=[x_offset, camera.views[cam].shape[1] + x_offset, camera.views[cam].shape[0] + y_offset, y_offset])

                # add scatter offset by crop_offset_x and crop_offset_y
                ax.scatter(
                    x=self.coords_CCS[cam][:, 0] + camera.specs[cam].get("crop_offset_x", 0),
                    y=self.coords_CCS[cam][:, 1] + camera.specs[cam].get("crop_offset_y", 0),
                    s=50,
                    c="r",
                    marker="x",
                    linewidth=0.5,
                    label=range(self.coords_CCS[cam].shape[0]),
                )

                # add image center
                ax.scatter(
                    x = camera.specs[cam]["principal_point_x_px"],
                    y = camera.specs[cam]["principal_point_y_px"],
                    s=50,
                    c="b",
                    marker="o",
                    linewidth=0.5,
                    label=range(self.coords_CCS[cam].shape[0]),
                )

                # add text offset by crop_offset_x and crop_offset_y
                for id in range(self.coords_CCS[cam].shape[0]):
                    ax.text(
                        x=self.coords_CCS[cam][id, 0] + camera.specs[cam].get("crop_offset_x", 0),
                        y=self.coords_CCS[cam][id, 1] + camera.specs[cam].get("crop_offset_y", 0),
                        s=id,
                        c="r",
                    )

                # set axes limits,
                ax.set_xlim(0, camera.specs[cam]["principal_point_x_px"] * 2)
                ax.set_ylim(0, camera.specs[cam]["principal_point_y_px"] * 2)
                ax.invert_yaxis()

                # add labels
                ax.set_title(cam)
                ax.set_xlabel("x (px)")
                ax.set_ylabel("y (px)")
                ax.axis("equal")

            return fig, axes
        except Exception as e:
            print(f"Error: {e}")
            return None, None

    def plot_WCS(self):
        """
        Plot belt points in WCS.
        """

        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")

        # add scatter
        ax.scatter(
            self.coords_WCS[:, 0],
            self.coords_WCS[:, 1],
            self.coords_WCS[:, 2],
            s=50,
            c="r",
            marker=".",
            linewidth=0.5,
            alpha=1,
        )

        # add text
        for id in range(self.coords_WCS.shape[0]):
            ax.text(
                self.coords_WCS[id, 0],
                self.coords_WCS[id, 1],
                self.coords_WCS[id, 2],
                s=id,
                c="r",
            )

        for row, col in zip(np.eye(3), ["r", "g", "b"]):
            ax.quiver(
                0,
                0,
                0,
                row[0],
                row[1],
                row[2],
                color=col,
                length=500,
                arrow_length_ratio=0,
                normalize=True,
            )

        # add text
        ax.text(
            0,
            0,
            500,
            s="WCS",
            c="k",
        )

        ax.set_xlabel("x (mm)")
        ax.set_ylabel("y (mm)")
        ax.set_zlabel("z (mm)")
        ax.axis("equal")

        return fig, ax


def plot_rotated_CS_in_WCS(fig, rot_cam, trans_cam):
    """
    Plot camera coordinate systems in the WCS.
    """

    ax = fig.add_subplot(projection="3d")

    # WCS
    for row, col in zip(np.eye(3), ["r", "g", "b"]):
        ax.quiver(
            0,
            0,
            0,
            row[0],
            row[1],
            row[2],
            color=col,
            # length=1,
            arrow_length_ratio=0,
            normalize=True,
        )

    # camera coordinate systems
    for row, col in zip(rot_cam.T, ["r", "g", "b"]):
        ax.quiver(
            trans_cam[0],
            trans_cam[1],
            trans_cam[2],
            row[0],
            row[1],
            row[2],
            color=col,
            arrow_length_ratio=0,
            normalize=True,
            linestyle=":",
            linewidth=4,
        )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    return fig, ax