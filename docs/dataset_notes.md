# Dataset Notes & Acquisition Verification

## 1. KU Leuven High Quality Fall Simulation Dataset (HQFSD)

### Citation & Provenance
* **Publication**: Baldewijns, G., Debard, G., Mertes, G., Vanrumste, B., & Croonenborghs, T. (2016). *Bridging the gap between real-life data and simulated data by providing a highly realistic fall dataset for evaluating camera-based fall detection algorithms*. Healthcare Technology Letters, 3(1), 6–11.
* **Recording Setup**: Real nursing-home room, 5 synchronized network cameras (Cam1–Cam5), 640×480 resolution at 12–30 fps, 10 volunteer actors re-enacting real fall incidents and continuous daily living activities.

### Download & Link Verification
1. **Official AdvISe Portal**: Successfully accessed at `https://iiw.kuleuven.be/onderzoek/advise/datasets`.
2. **Metadata Workbook**: Successfully fetched and archived locally at `data/HQFSD_metadata.xlsx`.
3. **Exemplary Full Video Downloads**:
   - `fall-1` (`data/raw/hqfsd_fall_1.avi`, 27.8 MB, 800×480 @ 30 fps, 269.9s): Downloaded and validated. Features a continuous nursing-home scenario with bed present, walker support, and a backwards fall to the floor.
   - `fall-2` (`data/raw/hqfsd_fall_2.avi`, 11.0 MB, 800×480 @ 30 fps, 140.1s): Downloaded and validated. Features a fall adjacent to the bed.
4. **Historical Box.com Archive**: The legacy URL cited in earlier literature (`http://kuleuven.box.com/s/dyo66et36l2lqvl19i9i7p66761sy0s6`) was verified and returns **HTTP 404 (Not Found)**.

### Ground Truth Annotations in HQFSD: Do Per-Second ADL Labels Exist?
* **Finding**: **NO per-second activity state labels exist for the ADL videos in the official metadata.**
* **Details**:
  - The `ADL` sheet in `HQFSD_metadata.xlsx` catalogs 17 scenarios (durations ranging from 11m 38s to 35m 30s).
  - Each entry lists only overall scenario duration, actor ID, walking aid, and a coarse free-text list of actions performed (e.g. *“in/out of bed, sleeping, walking, sitting down”*). No frame-by-frame or second-by-second timestamps are provided for these transitions.
  - The `Fall` sheet in contrast contains explicit timestamp intervals (`Start`, `Fall`, `End`, `Length`), actor IDs, starting/ending postures (e.g. `Standing -> Lying`), and camera visibility rankings (`Cam1`–`Cam5`).
* **Conclusion**: Held-out ADL test evaluation strictly requires manual annotation of state boundaries per the ground-truth protocol defined in `data/README.md`.

---

## 2. Supplementary Dataset: GMDCSA-24

### Provenance
* **Publication**: Alam, E., Sufian, A., Dutta, P., Leo, M., & Hameed, I. A. (2024). *GMDCSA-24: A dataset for human fall detection in videos*. Data in Brief, 54, 110452.
* **Repackaged Distribution**: Hugging Face `Voxel51/GMNCSA24-FO` and Zenodo DOI `10.5281/zenodo.12921216`.
* **Characteristics**: Natural home bedroom/living setups, 720p resolution (1280×720 @ 30 fps), 4 actors, varying daytime/nighttime lighting conditions, 335 annotated action segments.

### Suitability
* Excellent coverage of key target states: `Sitting` on bed edge, `Sleeping` in bed, `Standing`, `Walking`, and directional `Falling` (`FW`, `BW`, `SW`).
* Serves as an ideal paired suite with HQFSD: provides high-clarity bedroom geometry alongside the low-resolution continuous nursing-home recordings.

---

## 3. Video Manifest & Dataset Splits

The manifest is generated at `data/manifest.csv` with the following designated splits:

| Video | Split | Camera | Duration | Description & Justification |
|---|---|---|---|---|
| `gmdcsa_01.mp4` | **dev** | `gmdcsa_room1` | 11.4s | **Development video**: Clear bed-exit sequence (`SITTING_ON_BED` $\rightarrow$ `STANDING` $\rightarrow$ `WALKING`). Used strictly for threshold tuning. |
| `gmdcsa_03.mp4` | **test** | `gmdcsa_room1` | 11.9s | **Held-out test video 1**: Bed-return sequence (`WALKING` $\rightarrow$ `SITTING_ON_BED` $\rightarrow$ `LYING_IN_BED`). |
| `gmdcsa_02.mp4` | **test** | `gmdcsa_room1` | 11.8s | **Held-out test video 2**: Sustained sitting on bed edge / drinking (`SITTING_ON_BED`). |
| `gmdcsa_04.mp4` | **test** | `gmdcsa_room1` | 7.8s | **Held-out test video 3**: Upright posture in bedroom (`STANDING` / exercising). |
| `gmdcsa_01-2.mp4` | **fall** | `gmdcsa_room1` | 6.4s | **Fall test clip 1**: `STANDING` $\rightarrow$ sideways fall $\rightarrow$ `LYING_ON_FLOOR`. |
| `gmdcsa_02-2.mp4` | **fall** | `gmdcsa_room1` | 8.4s | **Fall test clip 2**: `WALKING` $\rightarrow$ sideways fall $\rightarrow$ `LYING_ON_FLOOR`. |
| `gmdcsa_04-2.mp4` | **fall** | `gmdcsa_room1` | 10.2s | **Fall test clip 3**: `STANDING` $\rightarrow$ backward fall $\rightarrow$ `LYING_ON_FLOOR`. |
| `hqfsd_fall_1.avi` | **fall** | `hqfsd_cam3` | 269.9s | **Fall test clip 4**: Continuous nursing-home scenario with bed present; walker fall to floor at ~48s, remaining horizontal on floor through 269s (`ALERT` trigger test). |
| `hqfsd_fall_2.avi` | **fall** | `hqfsd_cam3` | 140.1s | **Fall test clip 5**: Bed-adjacent fall in nursing-home environment (`ALERT` trigger test). |

---

## 4. Bed Polygons

Normalized coordinates $[0, 1]$ stored under `config/bed_polygons/`:
* `gmdcsa_room1.json`: $[ [0.22, 0.28], [0.68, 0.28], [0.68, 0.88], [0.22, 0.88] ]$
* `hqfsd_cam3.json`: $[ [0.55, 0.25], [0.98, 0.25], [0.98, 0.85], [0.55, 0.85] ]$
* Per-video JSON files created for seamless CLI invocation via `--polygon config/bed_polygons/<video>.json`.
