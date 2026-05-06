"""Create Project dialog — collects project name, root, camera setup, and
optional label schemas. On submit, instantiates a Project and writes
project.yaml."""

import os
import tkinter as tk
from tkinter import filedialog, messagebox

from annotation_tool.constants import (
    DEFAULT_NAME, DEFAULT_REFERENCE_VIEW, DEFAULT_VIEWS,
)
from annotation_tool.help import FIELD_HELP
from annotation_tool.project import Project
from annotation_tool.gui.utils import help_button, make_scrollable


class CreateProjectDialog:
    def __init__(self, root, navigator):
        self.root = root
        self.navigator = navigator
        self.build()

    def build(self):
        self.navigator.clear_root()
        self.root.title("Create Project")

        # Pinned title at the top
        title_bar = tk.Frame(self.root)
        title_bar.pack(side=tk.TOP, fill=tk.X, padx=20, pady=(20, 6))
        tk.Label(
            title_bar, text="Create Project", font=("Helvetica", 14, "bold"),
        ).pack(anchor="w")

        # Pinned button bar at the bottom (packed before the scroll area so
        # it always reserves its space even on short windows)
        btns = tk.Frame(self.root)
        btns.pack(side=tk.BOTTOM, fill=tk.X, padx=20, pady=(6, 16))
        tk.Button(btns, text="Cancel", command=self.navigator.go_home).pack(side=tk.RIGHT, padx=4)
        tk.Button(btns, text="Create", command=self._submit).pack(side=tk.RIGHT, padx=4)

        # Scrollable middle: holds the form
        scroll_holder = tk.Frame(self.root)
        scroll_holder.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=20, pady=(0, 6))
        frame = make_scrollable(scroll_holder)

        # ---- Required ----
        req = tk.LabelFrame(frame, text="Required")
        req.pack(fill=tk.X, pady=(0, 12))

        self.project_name_var = tk.StringVar()
        self._add_text_row(req, "Project name", self.project_name_var, field_key="project_name")

        self.root_var = tk.StringVar()
        self._add_path_row(req, "Project root", self.root_var, field_key="project_root")

        self.name_var = tk.StringVar(value=DEFAULT_NAME)
        self._add_text_row(req, "Name", self.name_var, field_key="name")

        self.views_var = tk.StringVar(value=", ".join(DEFAULT_VIEWS))
        self._add_text_row(req, "Views (comma-separated)", self.views_var, field_key="views")

        self.ref_view_var = tk.StringVar(value=DEFAULT_REFERENCE_VIEW)
        self._add_text_row(req, "Reference view", self.ref_view_var, field_key="reference_view")

        self.num_cams_var = tk.StringVar(value=str(len(DEFAULT_VIEWS)))
        self._add_text_row(req, "Number of cameras", self.num_cams_var, field_key="num_cameras")

        self.framerate_var = tk.StringVar()
        self._add_text_row(
            req, "Frame rate (fps)", self.framerate_var, field_key="framerate_fps",
        )

        # ---- Optional at creation, required for Calibrate / Label ----
        opt = tk.LabelFrame(
            frame,
            text="Optional at creation — required before Calibrate / Label "
                 "(can be edited later in project.yaml)",
        )
        opt.pack(fill=tk.BOTH, expand=True, pady=(0, 12))

        self.calibration_text = self._add_textarea(
            opt, "Calibration labels (one per line)", height=4,
            field_key="calibration_labels",
        )
        self.movable_calib_text = self._add_textarea(
            opt, "Movable calibration labels (one per line)", height=2,
            field_key="movable_calibration_labels",
        )
        self.bodyparts_text = self._add_textarea(
            opt, "Body part labels (one per line)", height=6,
            field_key="body_part_labels",
        )
        self.optref_text = self._add_textarea(
            opt, "Optimisation reference labels (one per line)", height=3,
            field_key="optimisation_reference_labels",
        )
        self.weights_text = self._add_textarea(
            opt, "Reference label weights (label: weight per line)", height=3,
            field_key="reference_label_weights",
        )

    def _add_text_row(self, parent, label, var, field_key=None):
        block = tk.Frame(parent)
        block.pack(fill=tk.X, padx=8, pady=(4, 0))
        row = tk.Frame(block)
        row.pack(fill=tk.X)
        tk.Label(row, text=label, width=24, anchor="w").pack(side=tk.LEFT)
        tk.Entry(row, textvariable=var).pack(side=tk.LEFT, fill=tk.X, expand=True)
        self._maybe_add_help(row, label, field_key)
        self._maybe_add_inline_hint(block, field_key)

    def _add_path_row(self, parent, label, var, field_key=None):
        block = tk.Frame(parent)
        block.pack(fill=tk.X, padx=8, pady=(4, 0))
        row = tk.Frame(block)
        row.pack(fill=tk.X)
        tk.Label(row, text=label, width=24, anchor="w").pack(side=tk.LEFT)
        tk.Entry(row, textvariable=var).pack(side=tk.LEFT, fill=tk.X, expand=True)
        tk.Button(row, text="...", command=lambda: self._pick_dir(var)).pack(side=tk.LEFT, padx=4)
        self._maybe_add_help(row, label, field_key)
        self._maybe_add_inline_hint(block, field_key)

    def _pick_dir(self, var):
        chosen = filedialog.askdirectory(title="Choose a parent directory for the project")
        if chosen:
            var.set(chosen)

    def _add_textarea(self, parent, label, height=4, field_key=None):
        block = tk.Frame(parent)
        block.pack(fill=tk.BOTH, expand=True, padx=8, pady=(4, 0))
        header = tk.Frame(block)
        header.pack(fill=tk.X)
        tk.Label(header, text=label, anchor="w").pack(side=tk.LEFT)
        self._maybe_add_help(header, label, field_key)
        self._maybe_add_inline_hint(block, field_key)
        text = tk.Text(block, height=height, wrap=tk.NONE)
        text.pack(fill=tk.BOTH, expand=True)
        return text

    def _maybe_add_help(self, row, label, field_key):
        """Append a '?' help button to `row` if FIELD_HELP has a long-form
        entry for this field. Skipped silently if absent."""
        if not field_key:
            return
        entry = FIELD_HELP.get(field_key)
        if entry is None or not entry.long:
            return
        help_button(row, label, entry.long).pack(side=tk.LEFT, padx=(4, 0))

    def _maybe_add_inline_hint(self, block, field_key):
        """Add the short one-liner hint underneath the field if FIELD_HELP
        has one. Skipped silently if absent."""
        if not field_key:
            return
        entry = FIELD_HELP.get(field_key)
        if entry is None or not entry.short:
            return
        tk.Label(
            block, text=entry.short, fg="#666",
            font=("Helvetica", 8), anchor="w", justify=tk.LEFT, wraplength=600,
        ).pack(fill=tk.X, pady=(0, 2))

    # ----- Parsing helpers -----

    @staticmethod
    def _parse_csv_list(s):
        return [p.strip() for p in s.split(",") if p.strip()]

    @staticmethod
    def _parse_lines(text_widget):
        raw = text_widget.get("1.0", tk.END).strip()
        return [line.strip() for line in raw.splitlines() if line.strip()] or None

    @staticmethod
    def _parse_float(string_var):
        raw = string_var.get().strip()
        if not raw:
            return None
        try:
            return float(raw)
        except ValueError:
            raise ValueError(f"Frame rate must be a number — got {raw!r}")

    @staticmethod
    def _parse_weights(text_widget):
        raw = text_widget.get("1.0", tk.END).strip()
        if not raw:
            return None
        out = {}
        for line in raw.splitlines():
            line = line.strip()
            if not line:
                continue
            if ":" not in line:
                raise ValueError(f"Weight line missing ':' — got {line!r}")
            label, weight = line.split(":", 1)
            try:
                out[label.strip()] = float(weight.strip())
            except ValueError:
                raise ValueError(f"Weight for {label.strip()!r} is not a number: {weight!r}")
        return out or None

    # ----- Submit -----

    def _submit(self):
        try:
            project_name = self.project_name_var.get().strip()
            parent_dir = self.root_var.get().strip()
            views = self._parse_csv_list(self.views_var.get())
            ref_view = self.ref_view_var.get().strip()
            name = self.name_var.get().strip() or None

            num_cams_str = self.num_cams_var.get().strip()
            try:
                num_cams = int(num_cams_str)
            except ValueError:
                raise ValueError(f"Number of cameras is not an integer: {num_cams_str!r}")

            if not project_name:
                raise ValueError("Project name is required.")
            if not parent_dir:
                raise ValueError("Project root is required.")
            if not views:
                raise ValueError("At least one view is required.")
            if ref_view not in views:
                raise ValueError(
                    f"Reference view {ref_view!r} must be one of views {views}."
                )
            if num_cams != len(views):
                raise ValueError(
                    f"Number of cameras ({num_cams}) must match the number of views ({len(views)})."
                )

            framerate_fps = self._parse_float(self.framerate_var)
            if framerate_fps is None:
                raise ValueError("Frame rate (fps) is required.")

            calibration_labels = self._parse_lines(self.calibration_text)
            movable_calibration_labels = self._parse_lines(self.movable_calib_text)
            body_part_labels = self._parse_lines(self.bodyparts_text)
            optimisation_reference_labels = self._parse_lines(self.optref_text)
            reference_label_weights = self._parse_weights(self.weights_text)

            project_dir = os.path.join(parent_dir, project_name)
            project = Project.create(
                dir=project_dir, project_name=project_name, views=views,
                reference_view=ref_view, num_cameras=num_cams,
                name=name,
                calibration_labels=calibration_labels,
                body_part_labels=body_part_labels,
                optimisation_reference_labels=optimisation_reference_labels,
                reference_label_weights=reference_label_weights,
                movable_calibration_labels=movable_calibration_labels,
                framerate_fps=framerate_fps,
            )
        except (ValueError, FileExistsError, OSError) as e:
            messagebox.showerror("Cannot create project", str(e))
            return

        self.navigator.project = project
        self.navigator.go_project_view()
