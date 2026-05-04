"""Body part labelling tool with 3D projection assistance."""

import os
import tkinter as tk
from tkinter import messagebox

import cv2
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.backend_bases import MouseButton

from annotation_tool import paths
from annotation_tool.constants import (
    DEFAULT_BRIGHTNESS, DEFAULT_CONTRAST, DEFAULT_MARKER_SIZE,
)
from annotation_tool.camera.calibration import InitialCalibration
from annotation_tool.camera.geometry import (
    back_project_2d_to_3d, build_projection_matrix,
    camera_center_from_extrinsics, clip_ray_to_aabb,
    project_3d_to_views,
)
from annotation_tool.camera.optimisation import optimize_extrinsics
from annotation_tool.gui.base import LabellingBase
from annotation_tool.gui.utils import (
    generate_label_colors, apply_contrast_brightness, debounce,
)


class LabelFramesTool(LabellingBase):
    def __init__(self, root, main_tool, project, recording):
        super().__init__(root, main_tool, project, recording)
        self.body_part_labels = project.require_body_part_labels()
        self.calibration_labels = project.require_calibration_labels()
        # Subset of calibration_labels whose position varies per frame (e.g.
        # doors). Treated like body parts during labelling — placeable, draggable,
        # and not auto-propagated across frames. Optional; default is empty.
        self.movable_calibration_labels = project.movable_calibration_labels or []

        self.extracted_frames_path = {}
        self.calibration_data = None
        # Existing code uses self.labels for the body-part labels list (it's
        # later mutated in setup_labeling_ui); preserve that convention.
        self.labels = list(self.body_part_labels)
        self.label_colors = generate_label_colors(self.labels, self.calibration_labels)
        self.current_label = tk.StringVar(value=self.labels[0] if self.labels else "")
        self.projection_view = tk.StringVar(value=project.reference_view)
        self.body_part_points = {}
        self.calibration_points_static = {
            label: {v: None for v in project.views}
            for label in self.calibration_labels
        }
        self.cam_reprojected_points = {"near": {}, "far": {}}
        self.frames = {v: [] for v in project.views}
        self.projection_lines = {v: None for v in project.views}
        self.tooltip = None
        self.label_buttons = []
        self.tooltip_window = None
        self.frame_names = {v: [] for v in project.views}
        self.frame_numbers = {v: [] for v in project.views}

        self.spacer_lines_active = False
        self.spacer_lines_points = []
        self.spacer_lines = []

        self.last_update_time = 0

        self.label_frames_menu()

    def label_frames_menu(self):
        self.main_tool.clear_root()

        self.calibration_file_path = paths.calibration_csv(self.project, self.recording)
        enhanced_calibration_file = paths.calibration_csv_enhanced(self.project, self.recording)

        if not os.path.exists(self.calibration_file_path) and not os.path.exists(enhanced_calibration_file):
            messagebox.showerror(
                "Error",
                "No calibration found for this recording. Run Calibrate first.",
            )
            self.main_tool.go_project_view()
            return

        # Bug fix: previously hardcoded "Side"/"Front"/"Overhead" segments;
        # now driven by project.views via paths.frames_dir.
        self.extracted_frames_path = {
            v: paths.frames_dir(self.project, self.recording, v)
            for v in self.project.views
        }

        if not all(os.path.exists(p) for p in self.extracted_frames_path.values()):
            messagebox.showerror(
                "Error",
                "One or more extracted-frames folders not found. Run Extract first.",
            )
            self.main_tool.go_project_view()
            return

        self.current_frame_index = 0
        self.show_loading_popup()
        self.frames = {v: [] for v in self.project.views}
        self.root.after(100, self.load_frames, enhanced_calibration_file)

    def show_loading_popup(self):
        self.loading_popup = tk.Toplevel(self.root)
        self.loading_popup.geometry("300x100")
        self.loading_popup.title("Loading")
        tk.Label(self.loading_popup, text="Loading frames, please wait...").pack(pady=20, padx=20)
        self.root.update_idletasks()

    def load_frames(self, enhanced_calibration_file):
        valid_extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}

        for view in self.project.views:
            frame_files = sorted(
                (f for f in os.listdir(self.extracted_frames_path[view])
                 if os.path.splitext(f)[1].lower() in valid_extensions),
                key=lambda x: os.path.getctime(
                    os.path.join(self.extracted_frames_path[view], x)
                ),
            )
            for file in frame_files:
                frame = cv2.imread(os.path.join(self.extracted_frames_path[view], file))
                self.frames[view].append(frame)
                self.frame_names[view].append(file)
                image_number = int(os.path.splitext(file)[0].replace("img", ""))
                self.frame_numbers[view].append(image_number)

        min_frame_count = min(len(self.frames[v]) for v in self.project.views if self.frames[v])
        for view in self.project.views:
            self.frames[view] = self.frames[view][:min_frame_count]
            self.frame_numbers[view] = self.frame_numbers[view][:min_frame_count]

        print("Frames loaded - " + ", ".join(
            f"{v.capitalize()}: {len(self.frames[v])}" for v in self.project.views
        ))

        if min_frame_count == 0:
            messagebox.showerror("Error", "No frames found in the directories.")
            self.loading_popup.destroy()
            return

        self.loading_popup.destroy()

        self.body_part_points = {
            frame_idx: {label: {v: None for v in self.project.views} for label in self.labels}
            for frame_idx in range(min_frame_count)
        }

        self.matched_frames = [tuple(i for _ in self.project.views) for i in range(min_frame_count)]

        self.setup_labeling_ui()

        for view in self.project.views:
            label_file_path = paths.labels_csv(self.project, self.recording, view)
            if os.path.exists(label_file_path):
                self.load_existing_labels(label_file_path, view)

        if os.path.exists(enhanced_calibration_file):
            self.load_calibration_data(enhanced_calibration_file)
        else:
            self.load_calibration_data(self.calibration_file_path)

        self.display_frame()

    def setup_labeling_ui(self):
        self.main_tool.clear_root()

        main_frame = tk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True)

        control_frame = tk.Frame(main_frame)
        control_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)

        self.create_settings_controls(control_frame)

        frame_control = tk.Frame(control_frame)
        frame_control.pack(side=tk.LEFT, padx=20)

        self.frame_label = tk.Label(
            frame_control,
            text=f"Frame: {self.current_frame_index + 1}/{len(self.matched_frames)}",
        )
        self.frame_label.pack()

        tk.Button(frame_control, text="<<",
                  command=lambda: self.skip_frames(-1)).pack(side=tk.LEFT, padx=5)
        tk.Button(frame_control, text=">>",
                  command=lambda: self.skip_frames(1)).pack(side=tk.LEFT, padx=5)

        button_frame = tk.Frame(control_frame)
        button_frame.pack(side=tk.LEFT, padx=20)

        tk.Button(button_frame, text="Reset View", command=self.reset_view).pack(pady=5)
        tk.Button(button_frame, text="Spacer Lines",
                  command=self.toggle_spacer_lines).pack(pady=5)
        tk.Button(button_frame, text="Optimize Calibration",
                  command=self.optimize_calibration).pack(pady=5)

        # View selectors. self.current_view and self.projection_view are
        # already initialized in __init__ (to project.reference_view); the radio
        # buttons bind to those existing StringVars.
        view_frame = tk.Frame(control_frame)
        view_frame.pack(side=tk.RIGHT, padx=10, pady=5)
        tk.Label(view_frame, text="Label View").pack()
        for view in self.project.views:
            tk.Radiobutton(view_frame, text=view.capitalize(),
                           variable=self.current_view, value=view).pack(side=tk.TOP, pady=2)

        projection_frame = tk.Frame(control_frame)
        projection_frame.pack(side=tk.RIGHT, padx=10, pady=5)
        tk.Label(projection_frame, text="Projection View").pack()
        for view in self.project.views:
            tk.Radiobutton(projection_frame, text=view.capitalize(),
                           variable=self.projection_view, value=view).pack(side=tk.TOP, pady=2)

        control_frame_right = tk.Frame(control_frame)
        control_frame_right.pack(side=tk.RIGHT, padx=20)
        tk.Button(control_frame_right, text="Save Labels", command=self.save_labels).pack(pady=5)
        tk.Button(control_frame_right, text="Back to Project View",
                  command=self.main_tool.go_project_view).pack(pady=5)
        tk.Button(control_frame_right, text="Exit", command=self.confirm_exit).pack(pady=5)

        # Scrollable label buttons
        control_frame_labels = tk.Frame(main_frame)
        control_frame_labels.pack(side=tk.RIGHT, fill=tk.Y, padx=3, pady=1)

        self.labels = list(self.body_part_labels)
        self.label_colors = generate_label_colors(self.labels, self.calibration_labels)
        self.current_label = tk.StringVar(value=self.labels[0])

        self.label_canvas = tk.Canvas(control_frame_labels, width=100)
        self.label_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=False)
        self.label_scrollbar = tk.Scrollbar(
            control_frame_labels, orient=tk.VERTICAL, command=self.label_canvas.yview
        )
        self.label_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.label_canvas.configure(yscrollcommand=self.label_scrollbar.set)

        self.label_frame_widget = tk.Frame(self.label_canvas)
        self.label_canvas.create_window((0, 0), window=self.label_frame_widget, anchor="nw")
        self.label_frame_widget.bind(
            "<Configure>",
            lambda e: self.label_canvas.configure(scrollregion=self.label_canvas.bbox("all")),
        )

        for label in self.labels:
            color = self.label_colors[label]
            if label in self.calibration_labels and label not in self.movable_calibration_labels:
                continue
            label_button = tk.Radiobutton(
                self.label_frame_widget, text=label, variable=self.current_label,
                value=label, indicatoron=0, width=15, bg=color, font=("Helvetica", 7),
                command=lambda l=label: self.on_label_select(l),
            )
            label_button.pack(fill=tk.X, pady=1)
            self.label_buttons.append(label_button)

        self.current_label.set(self.body_part_labels[0])
        self.update_label_button_selection()

        # Canvas — one row per view, sized by project config
        self.create_per_view_canvas(main_frame, figsize=(10, 10))
        self.fig.subplots_adjust(left=0.02, right=0.999, top=0.99, bottom=0.01,
                                 wspace=0.01, hspace=0.005)
        for ax in self.axs:
            ax.tick_params(axis="both", which="major", labelsize=8)
            ax.xaxis.set_major_locator(plt.MultipleLocator(50))
            ax.yaxis.set_major_locator(plt.MultipleLocator(50))

        self.tooltip = self.fig.text(
            0, 0, "", va="bottom", ha="left", fontsize=8,
            bbox=dict(boxstyle="round,pad=0.3", fc="yellow", ec="black", lw=1), zorder=10,
        )
        self.tooltip.set_visible(False)

        self.connect_mouse_events()
        self.canvas.mpl_connect("motion_notify_event", self.show_tooltip)

        self.display_frame()

    def update_label_button_selection(self):
        for button in self.label_buttons:
            if button.cget("text") == self.current_label.get():
                button.select()

    def load_existing_labels(self, label_file_path, view):
        # Existing data is always read from the .h5 sibling. The csv path is
        # passed in as a discoverability check.
        h5_path = paths.labels_h5(self.project, self.recording, view)
        df = paths.load_labels_h5(h5_path)
        # The DLC label-file multi-index column header. Uses the user's
        # project.name (DLC convention calls this slot 'scorer'); falls back
        # to the literal string 'scorer' when no name is set.
        scorer = self.project.name or "scorer"

        for frame_idx_pos, frame_idx in enumerate(df.index):
            for label in self.labels:
                if (scorer, label, "x") in df.columns and (scorer, label, "y") in df.columns:
                    x = df.loc[frame_idx, (scorer, label, "x")]
                    y = df.loc[frame_idx, (scorer, label, "y")]
                    if not np.isnan(x) and not np.isnan(y):
                        self.body_part_points[frame_idx_pos][label][view] = (x, y)
                else:
                    print(f"Label '{label}' not found in DataFrame for frame {frame_idx}.")

    # ----- Display -----

    def display_frame(self):
        frame_nums = dict(zip(self.project.views, self.matched_frames[self.current_frame_index]))
        imgs = {view: self.frames[view][frame_nums[view]] for view in self.project.views}

        self.display_views(imgs)
        # display_views runs ax.cla() on every axis, which orphans any Line2D
        # stashed elsewhere. Drop those refs so don't try to .remove()
        # detached artists later (NotImplementedError).
        self.projection_lines = {v: None for v in self.project.views}
        self.spacer_lines = []
        for ax in self.axs:
            ax.set_title(ax.get_title(), fontsize=8)
            ax.tick_params(axis="both", which="both", direction="in",
                           top=True, right=True, labelsize=8)

        self.show_body_part_points()
        self.canvas.draw()

        self.current_label.set(self.body_part_labels[0])
        self.update_label_button_selection()

    def refresh_display(self):
        self.display_frame()

    def show_body_part_points(self, draw=True):
        for ax in self.axs:
            for collection in ax.collections:
                collection.remove()

        current_points = self.body_part_points[self.current_frame_index]
        for label, views in current_points.items():
            for view, coords in views.items():
                if coords is not None:
                    x, y = coords
                    ax = self.axs[self.project.views.index(view)]
                    color = self.label_colors[label]
                    if label in self.calibration_labels:
                        ax.scatter(x, y, c=color, s=self.marker_size_var.get() * 10,
                                   label=label, edgecolors="red", linewidths=1)
                    else:
                        ax.scatter(x, y, c=color, s=self.marker_size_var.get() * 10,
                                   label=label)
        if draw:
            self.canvas.draw_idle()

    # ----- Contrast/brightness with debounce + caching -----

    @debounce(0.1)
    def update_contrast_brightness(self, val):
        self.redraw_frame()

    def redraw_frame(self):
        current_xlim = [ax.get_xlim() for ax in self.axs]
        current_ylim = [ax.get_ylim() for ax in self.axs]
        self.display_frame()
        for ax, xlim, ylim in zip(self.axs, current_xlim, current_ylim):
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
        self.canvas.draw_idle()

    @debounce(0.1)
    def on_scroll(self, event):
        super().on_scroll(event)

    # ----- Tooltip -----

    def show_tooltip(self, event):
        if event.inaxes in self.axs:
            marker_size = self.marker_size_var.get() * 10
            for label, views in self.body_part_points[self.current_frame_index].items():
                for view, coords in views.items():
                    if view == self.current_view.get() and coords is not None:
                        x, y = coords
                        if np.hypot(x - event.xdata, y - event.ydata) < marker_size:
                            self.show_custom_tooltip(self.canvas.get_tk_widget(), label)
                            return
        self.hide_custom_tooltip()

    def show_custom_tooltip(self, wdgt, text):
        if self.tooltip_window:
            self.tooltip_window.destroy()
        self.tooltip_window = tk.Toplevel(wdgt)
        self.tooltip_window.overrideredirect(True)
        tk.Label(self.tooltip_window, text=text, background="yellow").pack()
        self.tooltip_window.update_idletasks()
        x_center = wdgt.winfo_pointerx() + 20
        y_center = wdgt.winfo_pointery() + 20
        self.tooltip_window.geometry(f"+{x_center}+{y_center}")
        wdgt.bind("<Leave>", self.hide_custom_tooltip)

    def hide_custom_tooltip(self, event=None):
        if self.tooltip_window:
            self.tooltip_window.destroy()
            self.tooltip_window = None

    # ----- Mouse interaction -----

    def on_click(self, event):
        if event.inaxes not in self.axs:
            return

        view = self.current_view.get()
        ax = self.axs[self.project.views.index(view)]
        label = self.current_label.get()
        color = self.label_colors[label]
        marker_size = self.marker_size_var.get()
        frame_points = self.body_part_points[self.current_frame_index]

        if event.button == MouseButton.RIGHT:
            if self.spacer_lines_active:
                if len(self.spacer_lines_points) < 2:
                    self.spacer_lines_points.append((event.xdata, event.ydata))
                    if len(self.spacer_lines_points) == 2:
                        self.draw_spacer_lines(
                            ax, self.spacer_lines_points[0], self.spacer_lines_points[1]
                        )
                return
            if event.key == "shift":
                self.delete_closest_point(ax, event, frame_points)
            else:
                if label in self.movable_calibration_labels or label not in self.calibration_labels:
                    frame_points[label][view] = (event.xdata, event.ydata)
                    ax.scatter(event.xdata, event.ydata, c=color, s=marker_size * 10,
                               label=label)
                    self.canvas.draw_idle()
                    self.advance_label()
                    self.draw_reprojected_points()
        elif event.button == MouseButton.LEFT:
            if label in self.movable_calibration_labels or label not in self.calibration_labels:
                self.dragging_point = self.find_closest_point(ax, event, frame_points)

    def on_drag(self, event):
        if self.dragging_point is None or event.inaxes not in self.axs:
            return
        label, view, _ = self.dragging_point
        if event.button == MouseButton.LEFT:
            self.body_part_points[self.current_frame_index][label][view] = (
                event.xdata, event.ydata
            )
            self.show_body_part_points()
            self.draw_reprojected_points()

    def on_mouse_release(self, event):
        super().on_mouse_release(event)
        if event.button == MouseButton.LEFT:
            self.dragging_point = None
        elif event.button == MouseButton.RIGHT and self.spacer_lines_active:
            if len(self.spacer_lines_points) == 2:
                self.spacer_lines_points = []
                self.spacer_lines_active = False

    def find_closest_point(self, ax, event, frame_points):
        min_dist = float("inf")
        closest_point = None
        for label, views in frame_points.items():
            for view, coords in views.items():
                if coords is not None:
                    x, y = coords
                    dist = np.hypot(x - event.xdata, y - event.ydata)
                    if dist < min_dist:
                        min_dist = dist
                        closest_point = (label, view, coords)
        return closest_point if min_dist < 10 else None

    def delete_closest_point(self, ax, event, frame_points):
        min_dist = float("inf")
        closest_label = None
        closest_view = None
        for label, views in frame_points.items():
            for view, coords in views.items():
                if coords is not None:
                    x, y = coords
                    dist = np.hypot(x - event.xdata, y - event.ydata)
                    if dist < min_dist:
                        min_dist = dist
                        closest_label = label
                        closest_view = view

        if closest_label and closest_view:
            if (closest_label in self.movable_calibration_labels
                    or closest_label not in self.calibration_labels):
                frame_points[closest_label][closest_view] = None
                self.display_frame()

    def advance_label(self):
        current_index = self.labels.index(self.current_label.get())
        next_index = (current_index + 1) % len(self.labels)
        if next_index != 0 or len(self.labels) == 1:
            self.current_label.set(self.labels[next_index])
        else:
            self.current_label.set("")
        self.draw_reprojected_points()

    def skip_frames(self, step):
        self.current_frame_index += step
        self.current_frame_index = max(0, min(
            self.current_frame_index, len(self.matched_frames) - 1
        ))
        self.frame_label.config(
            text=f"Frame: {self.current_frame_index + 1}/{len(self.matched_frames)}"
        )
        self.display_frame()
        self.current_label.set(self.body_part_labels[0])

    def on_label_select(self, label):
        self.current_label.set(label)
        self.draw_reprojected_points()

    def reset_view(self):
        for ax in self.axs:
            ax.set_xlim(0, ax.get_images()[0].get_array().shape[1])
            ax.set_ylim(ax.get_images()[0].get_array().shape[0], 0)
        self.contrast_var.set(DEFAULT_CONTRAST)
        self.brightness_var.set(DEFAULT_BRIGHTNESS)
        self.marker_size_var.set(DEFAULT_MARKER_SIZE)
        self.display_frame()

    # ----- Spacer lines -----

    def toggle_spacer_lines(self):
        self.spacer_lines_active = not self.spacer_lines_active
        if not self.spacer_lines_active:
            self.remove_spacer_lines()
            self.spacer_lines_points = []
        else:
            self.spacer_lines_points = []

    def remove_spacer_lines(self):
        for line in self.spacer_lines:
            line.remove()
        self.spacer_lines = []
        self.canvas.draw_idle()

    def draw_spacer_lines(self, ax, start_point, end_point):
        if self.spacer_lines:
            self.remove_spacer_lines()
        x_values = np.linspace(start_point[0], end_point[0], num=12)
        for x in x_values:
            line = ax.axvline(x=x, color="pink", linestyle=":", linewidth=1)
            self.spacer_lines.append(line)
        self.canvas.draw_idle()

    # ----- 3D projection -----

    def get_camera_center(self, view):
        if not self.calibration_data:
            return None
        e = self.calibration_data["extrinsics"][view]
        return camera_center_from_extrinsics(e["rotm"], e["tvec"])

    def back_project_label(self, view, bp):
        """Back-project the labelled 2D point for `bp` in `view` to a 3D point
        on the camera ray. Returns None if the label has not been placed yet."""
        pt = self.body_part_points[self.current_frame_index][bp][view]
        if pt is None or pt[0] is None or pt[1] is None:
            return None
        e = self.calibration_data["extrinsics"][view]
        K = self.calibration_data["intrinsics"][view]
        P = build_projection_matrix(K, e["rotm"], e["tvec"])
        return back_project_2d_to_3d(pt, P)

    def find_3d_edges(self, view, bp):
        """Back-project the labelled 2D point through the camera and clip the
        resulting 3D ray to project.imaging_area. Returns the entry/exit
        points of the ray as it passes through the imaging volume — these
        are then re-projected into the other views as the epipolar segment.
        """
        point_3d = self.back_project_label(view, bp)
        camera_center = self.get_camera_center(view)
        if point_3d is None or camera_center is None:
            return None, None
        direction = point_3d - camera_center
        return clip_ray_to_aabb(camera_center, direction, self.project.imaging_area)

    def reproject_3d_to_2d(self):
        view = self.projection_view.get()
        bp = self.current_label.get()

        self.cam_reprojected_points["near"] = {}
        self.cam_reprojected_points["far"] = {}

        near_edge, far_edge = self.find_3d_edges(view, bp)
        if near_edge is not None and far_edge is not None:
            other_views = [v for v in self.project.views if v != view]
            for wcs in [near_edge, far_edge]:
                key = "near" if wcs is near_edge else "far"
                self.cam_reprojected_points[key].update(project_3d_to_views(
                    wcs,
                    self.calibration_data["extrinsics"],
                    self.calibration_data["intrinsics"],
                    other_views,
                ))

    def draw_reprojected_points(self):
        self.reproject_3d_to_2d()
        for view in self.project.views:
            if view != self.projection_view.get():
                ax = self.axs[self.project.views.index(view)]
                if self.projection_lines[view] is not None:
                    self.projection_lines[view].remove()
                    self.projection_lines[view] = None

                frame = cv2.cvtColor(
                    self.frames[view][self.current_frame_index], cv2.COLOR_BGR2RGB
                )
                frame = apply_contrast_brightness(
                    frame, self.contrast_var.get(), self.brightness_var.get()
                )
                ax.imshow(frame)
                ax.set_title(f"{view.capitalize()} View", fontsize=8)
                ax.axis("on")

                if (view in self.cam_reprojected_points["near"]
                        and view in self.cam_reprojected_points["far"]):
                    near_point = self.cam_reprojected_points["near"][view]
                    far_point = self.cam_reprojected_points["far"][view]
                    self.projection_lines[view], = ax.plot(
                        [near_point[0], far_point[0]], [near_point[1], far_point[1]],
                        color="red", linestyle="--", linewidth=0.5,
                    )

                ax.tick_params(axis="both", which="both", direction="in",
                               top=True, right=True, labelsize=8)

        self.show_body_part_points()
        self.canvas.draw_idle()

    # ----- Calibration -----

    def load_calibration_data(self, calibration_data_path):
        try:
            calibration_coordinates = pd.read_csv(calibration_data_path)
            calib = InitialCalibration(calibration_coordinates, self.project)
            cameras_extrinsics = calib.estimate_cams_pose()

            self.calibration_data = {
                "extrinsics": cameras_extrinsics,
                "intrinsics": calib.cameras_intrinsics,
            }

            for label in self.calibration_labels:
                for view in self.project.views:
                    x_vals = calibration_coordinates[
                        (calibration_coordinates["bodyparts"] == label)
                        & (calibration_coordinates["coords"] == "x")
                    ][view].values
                    y_vals = calibration_coordinates[
                        (calibration_coordinates["bodyparts"] == label)
                        & (calibration_coordinates["coords"] == "y")
                    ][view].values

                    if len(x_vals) > 0 and len(y_vals) > 0:
                        x, y = x_vals[0], y_vals[0]
                        self.calibration_points_static[label][view] = (x, y)
                        if label not in self.movable_calibration_labels:
                            for frame in self.body_part_points.keys():
                                self.body_part_points[frame][label][view] = (x, y)
                    else:
                        self.calibration_points_static[label][view] = None
                        print(f"Missing data for {label} in {view} view")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to load calibration data: {e}")
            print(f"Error loading calibration data: {e}")

    # ----- Calibration optimization -----

    def optimize_calibration(self):
        """Refine camera extrinsics by minimising body-part reprojection
        error. The math lives in camera/optimisation.py; this method
        bridges GUI state in / out and prints/saves the result."""
        result = optimize_extrinsics(
            project=self.project,
            body_part_labels_at_frame=self.body_part_points[self.current_frame_index],
            calibration_points=self.calibration_points_static,
            intrinsics=self.calibration_data["intrinsics"],
            extrinsics=self.calibration_data["extrinsics"],
        )

        # Apply optimised positions back onto the GUI state
        for label, views in result["optimized_calibration_points"].items():
            for view, point in views.items():
                self.calibration_points_static[label][view] = point
        self.calibration_data = result["calibration_data"]

        print(f"Initial total reprojection error: {result['initial_total_error']}")
        for label, views in result["initial_errors"].items():
            print(f"  {label}: {views}")
        print(f"Optimized total reprojection error: {result['final_total_error']}")
        for label, views in result["final_errors"].items():
            print(f"  {label}: {views}")

        self.save_optimized_calibration_points()
        self.update_calibration_labels_and_projection()

    def save_optimized_calibration_points(self):
        calibration_path = paths.calibration_csv_enhanced(self.project, self.recording)

        data = {"bodyparts": [], "coords": []}
        for v in self.project.views:
            data[v] = []
        for label, coords in self.calibration_points_static.items():
            if label in self.calibration_labels:
                for coord in ["x", "y"]:
                    data["bodyparts"].append(label)
                    data["coords"].append(coord)
                    for view in self.project.views:
                        if coords[view] is not None:
                            x, y = coords[view]
                            data[view].append(x if coord == "x" else y)
                        else:
                            data[view].append(None)

        df = pd.DataFrame(data)
        try:
            paths.save_calibration_csv(df, calibration_path)
            messagebox.showinfo("Info", "Optimized calibration points saved successfully")
        except PermissionError:
            print(f"Permission denied: {calibration_path}")
            messagebox.showerror("Error", f"Unable to save: {calibration_path}")

    def update_calibration_labels_and_projection(self):
        enhanced_path = paths.calibration_csv_enhanced(self.project, self.recording)
        if os.path.exists(enhanced_path):
            self.load_calibration_data(enhanced_path)
            self.display_frame()
        else:
            print(f"Enhanced calibration file not found: {enhanced_path}")

    # ----- Save / Exit -----

    def save_labels(self):
        # The DLC label-file multi-index column header. Uses the user's
        # project.name (DLC convention calls this slot 'scorer'); falls back
        # to the literal string 'scorer' when no name is set.
        scorer = self.project.name or "scorer"
        # Per-view DataFrame second-level identifier in the MultiIndex.
        # Mirrors the per-recording-per-view naming the old code produced
        # via the directory basename.
        video_id_for_view = {v: f"{self.recording.name}_{v}" for v in self.project.views}

        data = {view: [] for view in self.project.views}
        for frame_idx, labels in self.body_part_points.items():
            for label, views in labels.items():
                for view, coords in views.items():
                    if coords is not None:
                        x, y = coords
                        frame_number = self.matched_frames[frame_idx][self.project.views.index(view)]
                        filename = self.frame_names[view][frame_number]
                        data[view].append((
                            frame_idx, label, x, y, scorer,
                            video_id_for_view[view], filename,
                        ))

        for view, view_data in data.items():
            df_view = pd.DataFrame(
                view_data,
                columns=["frame_index", "label", "x", "y", "scorer",
                         "video_filename", "frame_filename"],
            )

            multi_cols = pd.MultiIndex.from_product(
                [[scorer], self.labels, ["x", "y"]],
                names=["scorer", "bodyparts", "coords"],
            )
            multi_idx = pd.MultiIndex.from_tuples([
                ("labeled_data", video_id_for_view[view], filename)
                for filename in df_view["frame_filename"].unique()
            ])
            df_ordered = pd.DataFrame(index=multi_idx, columns=multi_cols)

            for _, row in df_view.iterrows():
                df_ordered.loc[
                    ("labeled_data", row.video_filename, row.frame_filename),
                    (scorer, row.label, "x"),
                ] = row.x
                df_ordered.loc[
                    ("labeled_data", row.video_filename, row.frame_filename),
                    (scorer, row.label, "y"),
                ] = row.y

            df_ordered = df_ordered.apply(pd.to_numeric)

            try:
                csv_path, h5_path = paths.save_labels(
                    df_ordered, self.project, self.recording, view, scorer=scorer,
                )
                print(f"Saved {csv_path}")
            except PermissionError as e:
                print(f"PermissionError: {e}")
                messagebox.showerror(
                    "Error", f"Unable to save labels for {view}. Check file permissions.",
                )

        print("Labels saved successfully")
        messagebox.showinfo("Info", "Labels saved successfully")

    def confirm_exit(self):
        answer = messagebox.askyesnocancel("Exit",
                                            "Do you want to save labels before exiting?")
        if answer is not None:
            if answer:
                self.save_labels()
            self.root.quit()
