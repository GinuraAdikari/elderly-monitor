# Perception & Tracking Inspection Notes

## 1. Overview & Setup
* **Perception Engine**: YOLO11s-pose (`yolo11s-pose.pt`) coupled with ByteTrack tracker and CLAHE low-light enhancement.
* **Sampling Rate**: 5 fps (`stride = round(source_fps / 5)`), caching results to `outputs/<video>/perception.parquet` (66 columns: bbox coordinates, detection confidence, 17 keypoint $(x, y, c)$ coordinates).
* **Visualization Tool**: Created [`scripts/render_overlay.py`](file:///D:/elderly-monitor/elderly-monitor/elderly-monitor/scripts/render_overlay.py) which renders:
  - Yellow bed polygon boundary with `BED ZONE` marker.
  - Bounding box with track ID (neon green for patient, orange for others) and detection confidence.
  - 17 COCO keypoints and color-coded skeleton limbs.
  - Real-time Heads-Up Display (HUD) displaying timestamp, current state classification, posture probabilities (`Lie`, `Sit`, `Stand`), bed proximity (`p_bed`, `bed_dist_bl`), and image-plane speed (`BL/s`).

---

## 2. Dev Video Analysis (`gmdcsa_01.mp4`)

Rendered video: `outputs/gmdcsa_01/overlay.mp4` (960×540, 340 frames, 11.4s).

| Time Window | Visual Ground Truth | Extracted Features | Pipeline Output (`rules` mode) |
|---|---|---|---|
| **00:00 – 00:06** | Patient sitting upright on the edge of the mattress, feet resting near floor. | Torso angle $<20^\circ$, bbox aspect $w/h \approx 0.62$, knee-hip ratio $hk \approx 0.18$ (compressed), hips inside bed polygon (`p_bed = 1.0`). | **`SITTING_ON_BED`** (mean confidence: 0.85) |
| **00:06 – 00:08** | Patient pushes off the bed and extends knees to stand upright. | Torso vertical, knee-hip ratio extends to $hk > 0.65$, standing probability crosses from $0.26$ to $0.55$. | Transition to **`STANDING`** |
| **00:08 – 00:11** | Patient upright, stepping forward. | Torso angle $8^\circ$, hip velocity $\approx 0.08\text{ BL/s}$, standing probability $\approx 0.69$. | **`STANDING`** (mean confidence: 0.68) |

### Tracking Continuity
* The patient is detected in **100% of sampled frames** (presence score $= 1.0$).
* Keypoint confidence across all 17 joints averages **0.875** (shoulders $>0.96$, hips $>0.98$, knees $>0.90$).

---

## 3. Inspection of Complex Scenarios

### A. Lying Under Blankets (`gmdcsa_03.mp4`)
Rendered video: `outputs/gmdcsa_03/overlay.mp4` (11.9s).
* **Observation**: Patient walks to bed, sits down, and transitions to lying horizontal on the bed.
* **Pose Behavior**:
  - Upper body (head, shoulders, elbows) remains clearly detectable (confidence $>0.80$).
  - Lower body keypoints (knees, ankles) experience occlusion when blankets/bed covers conceal the legs.
  - Because shoulders and hips remain partially visible and the bounding box aspect ratio flips horizontally ($w/h > 1.3$), the geometry module successfully computes a horizontal torso orientation ($angle > 65^\circ$).
  - When legs are completely occluded, the feature extractor falls back to bounding box aspect ratio, correctly keeping `lying` probability $>0.70$.

### B. High-Velocity Fall & Static Floor Lying (`hqfsd_fall_1.avi`)
Processed from the KU Leuven dataset (269.9s, nursing home setup).
* **Observation**: Patient walks with a walker from $t=0\text{s}$ to $t=47\text{s}$, experiences a backward fall at $t=48\text{s}$, and remains static on the floor until $t=269\text{s}$.
* **Pose & Tracker Behavior**:
  - **Upright Phase ($0\text{s}$–$48\text{s}$)**: High keypoint confidence ($>0.88$), patient tracked continuously as Track ID 1.
  - **Dynamic Fall Phase ($48\text{s}$–$52\text{s}$)**: Fast descent introduces motion blur. Person box aspect changes rapidly.
  - **Post-Fall Floor Lying ($52\text{s}$–$269\text{s}$)**: Track ID drops during the fall blur and is reacquired as Track ID 8 once stationary.
  - **State Detection**: The system identifies horizontal posture on the floor outside the bed polygon, registering **`LYING_ON_FLOOR` for 2m 38s**, which promptly triggers the **`ALERT` level with reason: `Lying on floor for 158s (limit 20s)`**.

---

## 4. Where Pose Fails: Identified Failure Modes & Mitigations

Through visual overlay and feature extraction across the test suite, four distinct failure modes of 2D pose estimation were identified:

1. **Tracker ID Switches on Rapid Falls**:
   - *Failure*: When a patient falls abruptly, the sudden position displacement and motion blur can cause ByteTrack's Kalman filter to lose track continuity, assigning a new track ID upon impact.
   - *Mitigation in place*: `patient.py` implements temporal-spatial reacquisition (`reacquire_sec = 3.0s`, `reacquire_dist = 1.5` body heights) to link dropped tracks back to the primary resident.
2. **Lower-Limb Blanket Occlusion**:
   - *Failure*: Heavy blankets completely obscure hip, knee, and ankle keypoints. Hip keypoints fall below confidence threshold (`kp_conf = 0.30`).
   - *Mitigation in place*: `features.py` falls back to bbox center when hip midpoints are missing and uses bbox aspect ratio ($w/h$) when knee-hip ratios are unavailable. Furthermore, `states.py` implements an occlusion hold (`occ_hold = 0.5`) when a previously detected patient is in the bed zone.
3. **Perspective Foreshortening Along Camera Axis**:
   - *Failure*: If the bed is aligned along the camera's line of sight, a lying person appears vertically compressed in 2D image coordinates, leading to a square-ish bounding box ($w/h \approx 1.0$) rather than wide aspect.
   - *Mitigation in place*: The lying probability combines both torso angle ($\arctan(\Delta x, \Delta y)$) and aspect ratio with adaptive weighting ($0.6 \times lie\_angle + 0.4 \times lie\_aspect$).
4. **Bed Edge Ambiguity**:
   - *Failure*: When sitting on the extreme edge of the mattress with feet extended onto the floor, hips may sit close to the polygon border.
   - *Mitigation in place*: `edge_margin_bl = 0.25` body lengths creates a fuzzy margin where hips near the polygon boundary still classify as `SITTING_ON_BED`. Ambiguous edge cases are flagged by `triage.py` for agent review.
