from annotation_tool.camera.reconstruction import CameraData, BeltPoints


class BasicCalibration:
    def __init__(self, calibration_coords, views):
        self.cameras = CameraData()
        self.cameras_intrinsics = self.cameras.intrinsic_matrices

        self.belt_pts = BeltPoints(calibration_coords, views)
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
