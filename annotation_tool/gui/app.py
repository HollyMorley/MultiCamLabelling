"""
Main application — top-level state machine.
"""

from annotation_tool.project import Project


class Navigator:
    def __init__(self, root):
        self.root = root
        self.root.title("3D Annotation GUI")
        self.screen_width = self.root.winfo_screenwidth()
        self.screen_height = self.root.winfo_screenheight()
        self.project: Project | None = None
        self.go_home()

    # ----- Helpers -----

    def clear_root(self):
        for widget in self.root.winfo_children():
            widget.destroy()

    # ----- Navigation -----

    def go_home(self):
        from annotation_tool.gui.home import HomeScreen
        self.clear_root()
        HomeScreen(self.root, self)

    def go_create_project(self):
        from annotation_tool.gui.create_project import CreateProjectDialog
        CreateProjectDialog(self.root, self)

    def load_project(self, dir):
        self.project = Project.load(dir)
        self.go_project_view()

    def go_project_view(self):
        from annotation_tool.gui.project_view import ProjectView
        if self.project is None:
            self.go_home()
            return
        self.clear_root()
        ProjectView(self.root, self, self.project)

    def go_add_videos(self):
        from annotation_tool.gui.add_videos import AddVideosScreen
        if self.project is None:
            self.go_home()
            return
        self.clear_root()
        AddVideosScreen(self.root, self, self.project)

    def go_extract(self, recording):
        from annotation_tool.gui.extract import ExtractFramesTool
        if self.project is None:
            self.go_home()
            return
        self.clear_root()
        ExtractFramesTool(self.root, self, self.project, recording)

    def go_calibrate(self, recording):
        from annotation_tool.gui.calibrate import CalibrateCamerasTool
        if self.project is None:
            self.go_home()
            return
        self.clear_root()
        CalibrateCamerasTool(self.root, self, self.project, recording)

    def go_label(self, recording):
        from annotation_tool.gui.label import LabelFramesTool
        if self.project is None:
            self.go_home()
            return
        self.clear_root()
        LabelFramesTool(self.root, self, self.project, recording)
