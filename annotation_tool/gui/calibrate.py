"""Camera calibration tool — label known landmarks for camera pose estimation."""

import os
import tkinter as tk
from tkinter import messagebox

import cv2
import numpy as np
import pandas as pd
from matplotlib.backend_bases import MouseButton

from annotation_tool import paths
from annotation_tool.gui.base import BaseAnnotationTool
from annotation_tool.gui.utils import generate_label_colors
from annotation_tool.gui.sync import load_synced_video_captures


class CalibrateCamerasTool(BaseAnnotationTool):
    def __init__(self, root, main_tool, project, recording):
        super().__init__(root, main_tool, project, recording)
        self.caps = {}
        self.total_frames = 0
        self.mode = "calibration"

        self.labels = self.project.require_calibration_labels()
        self.calibration_points_static = {
            label: {v: None for v in self.project.views}
            for label in self.labels
        }
        self.label_colors = generate_label_colors(self.labels)
        self.current_label = tk.StringVar(value=self.labels[0])

        self._setup()

    def _setup(self):
        self.main_tool.clear_root()

        views = self.project.views

        self.caps, self.total_frames, self.matched_frames = load_synced_video_captures(
            self.project, self.recording,
        )
        self.current_frame_index = 0

        self.calibration_file_path = paths.calibration_csv(self.project, self.recording)
        enhanced_calibration_file = paths.calibration_csv_enhanced(self.project, self.recording)

        # Build UI
        main_frame = tk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True)

        control_frame = tk.Frame(main_frame)
        control_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)

        self.create_settings_controls(control_frame)

        frame_control = tk.Frame(control_frame)
        frame_control.pack(side=tk.LEFT, padx=20)

        self.frame_label = tk.Label(frame_control, text="Frame: 0")
        self.frame_label.pack()

        self.slider = tk.Scale(
            frame_control, from_=0, to=len(self.matched_frames) - 1,
            orient=tk.HORIZONTAL, length=400, command=self.update_frame_label,
        )
        self.slider.pack()

        skip_frame = tk.Frame(frame_control)
        skip_frame.pack()
        self.add_skip_buttons(skip_frame)

        button_frame = tk.Frame(control_frame)
        button_frame.pack(side=tk.LEFT, padx=20)

        save_column = tk.Frame(button_frame)
        save_column.pack(side=tk.LEFT, padx=5)
        tk.Button(save_column, text="Save Calibration Points",
                  command=self.save_calibration_points).pack(pady=5)
        tk.Button(save_column, text="Save + Set as Default",
                  command=self.save_calibration_points_and_set_as_default).pack(pady=5)

        nav_column = tk.Frame(button_frame)
        nav_column.pack(side=tk.LEFT, padx=5)
        tk.Button(nav_column, text="Reset View", command=self.reset_view).pack(pady=5)
        tk.Button(nav_column, text="Back to Project View",
                  command=self.main_tool.go_project_view).pack(pady=5)
        tk.Button(nav_column, text="Exit",
                  command=self.root.quit).pack(pady=5)

        # Label selector
        control_frame_right = tk.Frame(main_frame)
        control_frame_right.pack(side=tk.RIGHT, padx=10, pady=10, fill=tk.Y)

        for label in self.labels:
            color = self.label_colors[label]
            label_frame = tk.Frame(control_frame_right)
            label_frame.pack(fill=tk.X, pady=2)
            tk.Label(label_frame, bg=color, width=2).pack(side=tk.LEFT, padx=5)
            tk.Radiobutton(
                label_frame, text=label, variable=self.current_label,
                value=label, indicatoron=0, width=20,
            ).pack(side=tk.LEFT)

        for view in views:
            tk.Radiobutton(
                control_frame_right, text=view.capitalize(),
                variable=self.current_view, value=view,
            ).pack(pady=2)

        self.create_per_view_canvas(main_frame)
        self.connect_mouse_events()

        # Load existing calibration if available. Recording-level files take
        # precedence over the project-wide default.
        default_calibration_file = paths.default_calibration_csv(self.project)
        if os.path.exists(enhanced_calibration_file):
            if messagebox.askyesno(
                "Enhanced Calibration Found",
                "Enhanced calibration labels found for this recording. Load them?",
            ):
                self.load_calibration_points(enhanced_calibration_file)
        elif os.path.exists(self.calibration_file_path):
            if messagebox.askyesno(
                "Calibration Found",
                "Calibration labels found for this recording. Load them?",
            ):
                self.load_calibration_points(self.calibration_file_path)
        elif os.path.exists(default_calibration_file):
            if messagebox.askyesno(
                "Project Default Found",
                "No calibration for this recording yet. "
                "Load the project default as a starting point?",
            ):
                self.load_calibration_points(default_calibration_file)

        self.show_frames()

    def load_calibration_points(self, file_path):
        try:
            df = paths.load_calibration_csv(file_path)
            df.set_index(["bodyparts", "coords"], inplace=True)
            for label in df.index.levels[0]:
                for view in self.project.views:
                    if not pd.isna(df.loc[(label, "x"), view]):
                        x, y = df.loc[(label, "x"), view], df.loc[(label, "y"), view]
                        self.calibration_points_static[label][view] = self.axs[
                            self.project.views.index(view)
                        ].scatter(
                            x, y, c=self.label_colors[label],
                            s=self.marker_size_var.get() * 10, label=label,
                        )
            self.canvas.draw()
            messagebox.showinfo("Info", "Calibration points loaded successfully")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load calibration points: {e}")

    def update_frame_label(self, val):
        self.current_frame_index = int(val)
        self.show_frames()

    def skip_frames(self, step):
        new_index = self.current_frame_index + step
        new_index = max(0, min(new_index, len(self.matched_frames) - 1))
        self.current_frame_index = new_index
        self.slider.set(new_index)
        self.frame_label.config(text=f"Frame: {new_index}/{len(self.matched_frames) - 1}")
        self.show_frames()

    def show_frames(self, val=None):
        frame_number = self.current_frame_index
        self.frame_label.config(text=f"Frame: {frame_number}/{len(self.matched_frames) - 1}")

        views = self.project.views
        frame_nums = dict(zip(views, self.matched_frames[frame_number]))
        imgs = {}
        for view in views:
            self.caps[view].set(cv2.CAP_PROP_POS_FRAMES, frame_nums[view])
            ret, img = self.caps[view].read()
            if not ret:
                return
            imgs[view] = img

        self.display_views(imgs)
        self.show_static_points()
        self.canvas.draw_idle()

    def refresh_display(self):
        self.show_frames()

    def show_static_points(self):
        for label, points in self.calibration_points_static.items():
            for view, point in points.items():
                if point is not None:
                    ax = self.axs[self.project.views.index(view)]
                    ax.add_collection(point)
                    point.set_sizes([self.marker_size_var.get() * 10])
        self.canvas.draw()

    def _build_calibration_dataframe(self):
        """Serialise the current scatter points into a {bodyparts, coords, *views}
        DataFrame ready for save_calibration_csv."""
        data = {"bodyparts": [], "coords": []}
        for v in self.project.views:
            data[v] = []
        for label, coords in self.calibration_points_static.items():
            for coord in ["x", "y"]:
                data["bodyparts"].append(label)
                data["coords"].append(coord)
                for view in self.project.views:
                    if coords[view] is not None:
                        x, y = coords[view].get_offsets()[0]
                        data[view].append(x if coord == "x" else y)
                    else:
                        data[view].append(None)
        return pd.DataFrame(data)

    def save_calibration_points(self):
        df = self._build_calibration_dataframe()
        paths.save_calibration_csv(df, paths.calibration_csv(self.project, self.recording))
        messagebox.showinfo("Info", "Calibration points saved successfully")

    def save_calibration_points_and_set_as_default(self):
        """Save to the recording-level CSV and also write to the project-level
        default, so future recordings can be offered this calibration as a
        starting point. Confirms before overwriting an existing default."""
        default_path = paths.default_calibration_csv(self.project)
        if os.path.exists(default_path):
            if not messagebox.askyesno(
                "Overwrite Default",
                "A project default calibration already exists. Overwrite it?",
            ):
                return
        df = self._build_calibration_dataframe()
        paths.save_calibration_csv(df, paths.calibration_csv(self.project, self.recording))
        paths.save_calibration_csv(df, default_path)
        messagebox.showinfo("Info", "Calibration saved and set as project default")

    # ----- Mouse interaction -----

    def on_click(self, event):
        if event.inaxes not in self.axs:
            return

        view = self.current_view.get()
        ax = self.axs[self.project.views.index(view)]
        color = self.label_colors[self.current_label.get()]
        marker_size = self.marker_size_var.get()

        if event.button == MouseButton.RIGHT:
            if event.key == "shift":
                self.delete_closest_point(ax, event)
            else:
                label = self.current_label.get()
                if self.calibration_points_static[label][view] is not None:
                    self.calibration_points_static[label][view].remove()
                self.calibration_points_static[label][view] = ax.scatter(
                    event.xdata, event.ydata, c=color, s=marker_size * 10, label=label,
                )
                self.canvas.draw()
        elif event.button == MouseButton.LEFT:
            self.dragging_point = self.find_closest_point(ax, event)

    def on_drag(self, event):
        if self.dragging_point is None or event.inaxes not in self.axs:
            return
        if event.button == MouseButton.LEFT:
            self.dragging_point.set_offsets((event.xdata, event.ydata))
            self.canvas.draw()

    def find_closest_point(self, ax, event):
        min_dist = float("inf")
        closest_point = None
        for label, points in self.calibration_points_static.items():
            if points[self.current_view.get()] is not None:
                x, y = points[self.current_view.get()].get_offsets()[0]
                dist = np.hypot(x - event.xdata, y - event.ydata)
                if dist < min_dist:
                    min_dist = dist
                    closest_point = points[self.current_view.get()]
        return closest_point if min_dist < 10 else None

    def delete_closest_point(self, ax, event):
        min_dist = float("inf")
        closest_label = None
        for label, points in self.calibration_points_static.items():
            if points[self.current_view.get()] is not None:
                x, y = points[self.current_view.get()].get_offsets()[0]
                dist = np.hypot(x - event.xdata, y - event.ydata)
                if dist < min_dist:
                    min_dist = dist
                    closest_label = label

        if closest_label:
            self.calibration_points_static[closest_label][self.current_view.get()].remove()
            self.calibration_points_static[closest_label][self.current_view.get()] = None
            self.canvas.draw()
