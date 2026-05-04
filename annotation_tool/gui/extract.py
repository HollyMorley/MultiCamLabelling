"""Frame extraction tool — synchronize and extract frames from multi-camera videos."""

import tkinter as tk
from tkinter import messagebox

import cv2

from annotation_tool import paths
from annotation_tool.gui.base import FrameDisplayBase
from annotation_tool.sync import load_synced_video_captures


class ExtractFramesTool(FrameDisplayBase):
    def __init__(self, root, main_tool, project, recording):
        super().__init__(root, main_tool, project, recording)
        self.caps = {}
        self.total_frames = 0
        self._open_videos_and_sync()

    def _open_videos_and_sync(self):
        """Open video captures for every view and build the matched-frame
        list using timestamp synchronisation."""
        self.main_tool.clear_root()

        self.caps, self.total_frames, self.matched_frames = load_synced_video_captures(
            self.project, self.recording,
        )
        self.current_frame_index = 0

        frame_counts = {
            v: int(self.caps[v].get(cv2.CAP_PROP_FRAME_COUNT))
            for v in self.project.views
        }
        print("Total frames - " + ", ".join(
            f"{v.capitalize()}: {frame_counts[v]}" for v in self.project.views
        ))

        self._show_frames_extraction()

    def _show_frames_extraction(self):
        self.main_tool.clear_root()
        views = self.project.views
        ref = self.project.reference_view

        control_frame = tk.Frame(self.root)
        control_frame.pack(side=tk.TOP, pady=10)

        self.slider = tk.Scale(
            control_frame, from_=0, to=len(self.matched_frames) - 1,
            orient=tk.HORIZONTAL, length=600, command=self.update_frame_label,
        )
        self.slider.pack(side=tk.LEFT, padx=5)

        ref_idx = views.index(ref)
        self.frame_label = tk.Label(
            control_frame, text=f"Frame: {self.matched_frames[0][ref_idx]}"
        )
        self.frame_label.pack(side=tk.LEFT, padx=5)

        skip_frame = tk.Frame(self.root)
        skip_frame.pack(side=tk.TOP, pady=10)
        self.add_skip_buttons(skip_frame)

        control_frame_right = tk.Frame(self.root)
        control_frame_right.pack(side=tk.RIGHT, padx=10, pady=10)

        tk.Button(control_frame_right, text="Extract Frames",
                  command=self.save_extracted_frames).pack(pady=5)
        tk.Button(control_frame_right, text="Back to Project View",
                  command=self.main_tool.go_project_view).pack(pady=5)
        tk.Button(control_frame_right, text="Exit",
                  command=self.root.quit).pack(pady=5)

        self.create_per_view_canvas(self.root)
        self.display_frame(0)

    def skip_frames(self, step):
        new_index = self.current_frame_index + step
        new_index = max(0, min(new_index, len(self.matched_frames) - 1))
        self.slider.set(new_index)
        self.display_frame(new_index)

    def read_matched(self, index):
        """Read the matched frame for every view at the given matched index.

        Returns dict[view -> BGR image] or None if any view failed to read.
        """
        views = self.project.views
        frame_nums = dict(zip(views, self.matched_frames[index]))
        imgs = {}
        for view in views:
            self.caps[view].set(cv2.CAP_PROP_POS_FRAMES, frame_nums[view])
            ret, img = self.caps[view].read()
            if not ret:
                return None, frame_nums
            imgs[view] = img
        return imgs, frame_nums

    def display_frame(self, index):
        self.current_frame_index = index
        imgs, _ = self.read_matched(index)
        if imgs is None:
            return
        self.display_views(imgs)
        self.canvas.draw()

    def refresh_display(self):
        self.display_frame(self.current_frame_index)

    def update_frame_label(self, val):
        index = int(val)
        ref_idx = self.project.views.index(self.project.reference_view)
        self.frame_label.config(text=f"Frame: {self.matched_frames[index][ref_idx]}")
        self.display_frame(index)

    def save_extracted_frames(self):
        imgs, frame_nums = self.read_matched(self.current_frame_index)
        if imgs is None:
            return
        for view in self.project.views:
            paths.save_frame_image(
                self.project, self.recording, view, frame_nums[view], imgs[view],
            )
        messagebox.showinfo("Info", "Frames saved successfully")
