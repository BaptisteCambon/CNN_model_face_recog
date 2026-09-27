# Project Task List & Technical Debt (`TODO.md`)

This document tracks immediate actionable tasks, active development priorities, and known technical debt for the face detection model pipeline.

---

## Priority 1: High Priority / In Progress

- [ ] Finish the line-by-line walkthrough in progress: `architecture.py`
- [ ] Finish the line-by-line walkthrough in progress: `loss.py`
- [ ] Finish the line-by-line walkthrough in progress: `model_training.py`
- [ ] Finish the line-by-line walkthrough in progress: `face_tracker.py`

## Priority 2: Medium Priority / Next Up

- [ ] The README references **`prepare_widerface_dataset.py`** but it doesn't exist in
      this repo.
- [ ] Reconcile `README.md` with the actual codebase (filename casing,
      the missing prep script) so it stops describing files that don't exist.

## Priority 3: Low Priority / End Stage

- [ ] Translate/replace the stray French comment in `architecture.py`.
- [ ] Add a couple of basic unit tests for `Preprocessing.py`'s pure
      functions (`compute_letterbox_params`, `_iou`, `encode_grid_targets`
      round-tripping with `decode_grid_predictions`)
- [ ] Add more negative (no-face) samples. There is currently not enough negative 
      images so the model allucinate faces.