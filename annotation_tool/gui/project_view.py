"""Project view — shows recordings in the loaded project, with per-row
Extract / Calibrate / Label actions."""

import tkinter as tk

from annotation_tool import paths


class ProjectView:
    def __init__(self, root, main_tool, project):
        self.root = root
        self.main_tool = main_tool
        self.project = project
        self.build()

    def build(self):
        self.main_tool.clear_root()
        self.root.title(f"Project: {self.project.project_name}")

        outer = tk.Frame(self.root)
        outer.pack(expand=True, fill=tk.BOTH, padx=20, pady=20)

        # Header
        header = tk.Frame(outer)
        header.pack(fill=tk.X, pady=(0, 10))
        tk.Label(
            header, text=f"Project: {self.project.project_name}", font=("Helvetica", 16, "bold"),
        ).pack(side=tk.LEFT)
        tk.Label(
            header,
            text=f"  ({len(self.project.views)} cameras, "
                 f"reference: {self.project.reference_view})",
            fg="#666",
        ).pack(side=tk.LEFT)
        tk.Button(header, text="Back to Home", command=self.main_tool.go_home).pack(side=tk.RIGHT, padx=2)
        tk.Button(header, text="Add Videos", command=self.main_tool.go_add_videos).pack(side=tk.RIGHT, padx=2)

        # Recordings
        body = tk.Frame(outer)
        body.pack(fill=tk.BOTH, expand=True)

        tk.Label(body, text="Recordings", font=("Helvetica", 12, "bold")).pack(anchor="w", pady=(8, 4))

        if not self.project.recordings:
            tk.Label(
                body,
                text="No recordings yet. Click \"Add Videos\" to add one.",
                fg="#666",
            ).pack(anchor="w", pady=20)
            return

        # Header row
        cols = tk.Frame(body)
        cols.pack(fill=tk.X)
        tk.Label(cols, text="Recording", width=30, anchor="w", font=("Helvetica", 10, "bold")).pack(side=tk.LEFT)
        tk.Label(cols, text="Status", width=30, anchor="w", font=("Helvetica", 10, "bold")).pack(side=tk.LEFT)
        tk.Label(cols, text="Actions", anchor="w", font=("Helvetica", 10, "bold")).pack(side=tk.LEFT)

        # One row per recording
        for recording in self.project.recordings:
            self._build_row(body, recording)

    def _build_row(self, parent, recording):
        row = tk.Frame(parent, pady=2)
        row.pack(fill=tk.X)

        tk.Label(row, text=recording.name, width=30, anchor="w").pack(side=tk.LEFT)

        # Status
        has_frames = paths.has_extracted_frames(self.project, recording)
        has_calib = paths.has_calibration(self.project, recording)
        has_lbls = paths.has_labels(self.project, recording)
        bits = []
        if has_frames:
            bits.append("frames")
        if has_calib:
            bits.append("calibration")
        if has_lbls:
            bits.append("labels")
        status = ", ".join(bits) if bits else "—"
        tk.Label(row, text=status, width=30, anchor="w", fg="#666").pack(side=tk.LEFT)

        # Actions — greyed out where prerequisites are missing
        can_calibrate = bool(self.project.calibration_labels)   # only need calib labels in yaml
        can_label = (                                           # need all labels defined, calib done, and frames extracted
            has_frames and has_calib
            and bool(self.project.body_part_labels)
            and bool(self.project.calibration_labels)
        )

        tk.Button(
            row, text="Extract",
            command=lambda r=recording: self.main_tool.go_extract(r),
        ).pack(side=tk.LEFT, padx=2)
        tk.Button(
            row, text="Calibrate",
            command=lambda r=recording: self.main_tool.
            go_calibrate(r),
            state=tk.NORMAL if can_calibrate else tk.DISABLED,
        ).pack(side=tk.LEFT, padx=2)
        tk.Button(
            row, text="Label",
            command=lambda r=recording: self.main_tool.go_label(r),
            state=tk.NORMAL if can_label else tk.DISABLED,
        ).pack(side=tk.LEFT, padx=2)
