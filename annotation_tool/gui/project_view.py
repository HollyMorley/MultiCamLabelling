"""Project view — shows recordings in the loaded project, with per-row
Extract / Calibrate / Label actions."""

import tkinter as tk

from annotation_tool import paths
from annotation_tool.gui.utils import attach_tooltip


class ProjectView:
    def __init__(self, root, navigator, project):
        self.root = root
        self.navigator = navigator
        self.project = project
        self.build()

    def build(self):
        self.navigator.clear_root()
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
        tk.Button(header, text="Back to Home", command=self.navigator.go_home).pack(side=tk.RIGHT, padx=2)
        tk.Button(header, text="Add Videos", command=self.navigator.go_add_videos).pack(side=tk.RIGHT, padx=2)

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

        # Actions — greyed out where prerequisites are missing. Each readiness
        # message combines project.yaml config (project.*_ready()) with
        # recording-level state (frames extracted, calibration saved).
        extract_msg = self.project.extract_ready()
        calibrate_msg = self.project.calibrate_ready()
        label_msg = self.project.label_ready()
        if label_msg is None and not has_frames:
            label_msg = "needs frames — run Extract first"
        if label_msg is None and not has_calib:
            label_msg = "needs calibration — run Calibrate first"

        self._tool_button(
            row, "Extract", lambda r=recording: self.navigator.go_extract(r),
            extract_msg,
        )
        self._tool_button(
            row, "Calibrate", lambda r=recording: self.navigator.go_calibrate(r),
            calibrate_msg,
        )
        self._tool_button(
            row, "Label", lambda r=recording: self.navigator.go_label(r),
            label_msg,
        )

    def _tool_button(self, parent, text, on_click, readiness_msg):
        """Build one tool button. If readiness_msg is non-None, the button is
        disabled and the message appears as a hover tooltip."""
        btn = tk.Button(
            parent, text=text, command=on_click,
            state=tk.NORMAL if readiness_msg is None else tk.DISABLED,
        )
        btn.pack(side=tk.LEFT, padx=2)
        attach_tooltip(btn, readiness_msg)
