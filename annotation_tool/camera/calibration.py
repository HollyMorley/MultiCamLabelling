from annotation_tool.camera.reconstruction import CameraData, BeltPoints


class BasicCalibration:
    def __init__(self, calibration_coords, project):
        self.project = project
        self.cameras = CameraData(project)
        self.cameras_intrinsics = self.cameras.intrinsic_matrices

        _, label_coords, _ = project.require_calibration_geometry()
        self.belt_pts = BeltPoints(calibration_coords, project.views, label_coords)
        self.belt_coords_CCS = self.belt_pts.coords_CCS
        self.belt_coords_WCS = self.belt_pts.coords_WCS

    def estimate_cams_pose(self):
        cameras_extrinsics = self.cameras.compute_cameras_extrinsics(
            self.belt_coords_WCS, self.belt_coords_CCS
        )
        return cameras_extrinsics

    def print_reprojection_errors(self, cameras_extrinsics, with_guess=False):
        if with_guess:
            print("Reprojection errors (w/ initial guess):")
        else:
            print("Reprojection errors:")
        for cam, data in cameras_extrinsics.items():
            print(f"  {cam}: {data['repr_err']}")
