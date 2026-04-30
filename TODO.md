# TODO

Loose list of next steps and improvements. Roughly grouped into smaller fixes and larger
features; not strictly prioritised.

## UX & small fixes

- Add an "Edit project file" button on the project home so `project.yaml` can be edited
  from the GUI.
- Add a "Save labels" button in the Label tool. Currently, labels are only saved on exit
  (via prompt).
- Add consistent Exit / save-check flow to all tools.
- In Extract, the user currently has to confirm every frame trio individually via pop-up - 
  bulk-save or silent-save would be much faster.
- Add unsaved-state checks in calibrate.py and label.py so the user is warned before
  losing work (for both 'Exit' and 'Back to Main Menu'.
- Make Spacer Lines configurable (count and axis) instead of the current 12-on-x default.
  Maybe allow multiples, accessible via dropdown.
- Refresh projection lines when the active view is switched in Label.

## Larger / structural

- Add type hints across the entire project.
- Make timestamp adjustment optional in `project.yaml`, or report a "how bad is the drift?"
  number/graph instead of always running the linear fit.
- Split `BaseAnnotationTool` into a `FrameBrowserTool` (used by Extract) and an
  annotation subclass (used by Calibrate / Label). Currently, Extract reimplements
  some of the base-class scaffolding but in limited scope (no labelling).
- Generalise the multi-view layout. Equal-height stacked rows won't suit
  every setup where view counts and aspect ratios will vary.

