"""Add Videos screen — pick video files for one new recording.

Each picked file is auto-tagged with its view (via filename heuristic) and
the recording name is auto-suggested from the filenames. The user can
override both before submitting.
"""

import os
import tkinter as tk
from tkinter import filedialog, messagebox

from annotation_tool import paths
from annotation_tool.project import Project


class AddVideosScreen:
    def __init__(self, root, main_tool, project: Project):
        self.root = root
        self.main_tool = main_tool
        self.project = project

        # rows = list of dicts: {path, view_var, basename}
        self.rows = []
        self.recording_name_var = tk.StringVar()

        self.build()

    def build(self):
        self.main_tool.clear_root()
        self.root.title("Add Videos")

        outer = tk.Frame(self.root)
        outer.pack(expand=True, fill=tk.BOTH, padx=20, pady=20)

        tk.Label(outer, text="Add Videos", font=("Helvetica", 14, "bold")).pack(anchor="w")
        tk.Label(
            outer,
            text=(
                "Pick one video per camera view for this recording. "
                "Each video and its sibling _Timestamps.csv will be copied into "
                "the project's Videos/ folder."
            ),
            wraplength=600, justify=tk.LEFT, fg="#666",
        ).pack(anchor="w", pady=(0, 12))

        # File selection
        pick_row = tk.Frame(outer)
        pick_row.pack(fill=tk.X, pady=(0, 8))
        tk.Button(pick_row, text="Choose video files...", command=self._pick_files).pack(side=tk.LEFT)
        tk.Label(
            pick_row,
            text=f"  (need one per chosen views: {', '.join(self.project.views)})",
            fg="#666",
        ).pack(side=tk.LEFT)

        # Table of picked files
        self.table_frame = tk.Frame(outer, relief=tk.SUNKEN, borderwidth=1)
        self.table_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 12))
        self._render_table()

        # Recording name
        rec_row = tk.Frame(outer)
        rec_row.pack(fill=tk.X, pady=(0, 12))
        tk.Label(rec_row, text="Recording name", width=18, anchor="w").pack(side=tk.LEFT)
        tk.Entry(rec_row, textvariable=self.recording_name_var).pack(
            side=tk.LEFT, fill=tk.X, expand=True,
        )

        # Buttons
        btns = tk.Frame(outer)
        btns.pack(fill=tk.X)
        tk.Button(btns, text="Cancel", command=self.main_tool.go_project_view).pack(side=tk.RIGHT, padx=4)
        tk.Button(btns, text="Add Recording", command=self._submit).pack(side=tk.RIGHT, padx=4)

    def _pick_files(self):
        chosen = filedialog.askopenfilenames(
            title="Select video files (one per view)",
            initialdir=paths.videos_initial_dir(self.project),
            filetypes=[("Video files", "*.avi *.mp4 *.mov *.mkv"), ("All files", "*.*")],
        )
        if not chosen:
            return
        for path in chosen:
            base = os.path.basename(path)
            name, view = paths.parse_video_filename(base, self.project.views)
            self.rows.append({
                "path": path,
                "basename": base,
                "view_var": tk.StringVar(value=view or self.project.views[0]),
                "name_suggestion": name,
            })

        # Suggest a recording name from the first detected name (sans view).
        if not self.recording_name_var.get():
            for r in self.rows:
                if r["name_suggestion"]:
                    self.recording_name_var.set(r["name_suggestion"])
                    break

        self._render_table()

    def _render_table(self):
        for w in self.table_frame.winfo_children():
            w.destroy()

        if not self.rows:
            tk.Label(
                self.table_frame, text="No files selected yet.", fg="#888",
            ).pack(padx=10, pady=10)
            return

        # Header
        head = tk.Frame(self.table_frame)
        head.pack(fill=tk.X)
        tk.Label(head, text="File", width=40, anchor="w", font=("Helvetica", 10, "bold")).pack(side=tk.LEFT, padx=4, pady=2)
        tk.Label(head, text="View", width=12, anchor="w", font=("Helvetica", 10, "bold")).pack(side=tk.LEFT, padx=4, pady=2)
        tk.Label(head, text="", width=8, anchor="w").pack(side=tk.LEFT)

        for i, row in enumerate(self.rows):
            r = tk.Frame(self.table_frame)
            r.pack(fill=tk.X)
            tk.Label(r, text=row["basename"], width=40, anchor="w").pack(side=tk.LEFT, padx=4, pady=2)
            tk.OptionMenu(r, row["view_var"], *self.project.views).pack(side=tk.LEFT, padx=4, pady=2)
            # Note: `idx=i` snapshots the current i as a default arg so each
            # button knows its own row. Without it, all buttons would share
            # the loop's final i.
            tk.Button(
                r, text="Remove", command=lambda idx=i: self._remove_row(idx),
            ).pack(side=tk.LEFT, padx=4)

    def _remove_row(self, idx):
        self.rows.pop(idx)
        self._render_table()

    def _submit(self):
        rec_name = self.recording_name_var.get().strip()
        if not rec_name:
            messagebox.showerror("Missing input", "Recording name is required.")
            return
        if not self.rows:
            messagebox.showerror("Missing input", "Pick at least one video file.")
            return

        # Build view -> path map
        video_paths = {}
        for r in self.rows:
            view = r["view_var"].get()
            if view in video_paths:
                messagebox.showerror(
                    "Duplicate view",
                    f"Two files are tagged as view {view!r}. Each view must "
                    "have exactly one video.",
                )
                return
            video_paths[view] = r["path"]

        missing = set(self.project.views) - set(video_paths.keys())
        if missing:
            messagebox.showerror(
                "Missing views",
                f"Recording is missing videos for: {sorted(missing)}.",
            )
            return

        try:
            self.project.add_recording(rec_name, video_paths)
        except (ValueError, FileNotFoundError, OSError) as e:
            messagebox.showerror("Cannot add recording", str(e))
            return

        self.main_tool.go_project_view()
