# Data notes

Raw videos are NOT in the repo. Put them under `data/raw/` (git-ignored).

Primary dataset: KU Leuven High Quality Fall Simulation Dataset (HQFSD) — see the top-level README, section "Dataset".
Supplementary (poor-light / sitting-on-bed clips): GMDCSA-24 (Hugging Face: Voxel51/GMNCSA24-FO).

Workflow
1. Download HQFSD, put videos in `data/raw/`.
2. `python scripts/make_manifest.py data/raw` -> `data/manifest.csv`; fill `camera`, `split` (dev/test), `has_bed`.
3. `python scripts/select_bed_polygon.py --video <v> --out config/bed_polygons/<v>.json` (once per camera view).
4. Label the held-out videos -> `data/labels/<video>.csv` (+ optional `<video>_events.csv`). See `example_gt.csv`.
5. `python scripts/validate_labels.py data/labels/<video>.csv`

Label format — segments: `start,end,state` (seconds, mm:ss or hh:mm:ss); contiguous, no gaps; states from the 8-state list.
Events (recommended): `time,event` with event in {bed_exit, bed_return}; `time` = when the person starts leaving / approaching.
Label rules: label what is visible; use UNKNOWN when you truly cannot tell; sitting up in bed without hips leaving the
mattress is SITTING_ON_BED; a person who has left camera view after leaving the bed is OUT_OF_BED.
