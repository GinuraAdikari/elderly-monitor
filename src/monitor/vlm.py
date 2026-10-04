"""VLM helper: sample cropped frames around the patient and ask one constrained question. Cached on disk."""
from __future__ import annotations
import base64
import hashlib
import json
import pathlib
import numpy as np


class VLM:
    def __init__(self, client, cfg, video, bins, bed_px, cache_path):
        self.client, self.cfg, self.video, self.bins, self.bed_px = client, cfg, video, bins, bed_px
        self.cache_path = pathlib.Path(cache_path)
        self.cache = json.load(open(self.cache_path)) if self.cache_path.exists() else {}

    def frames(self, t0, t1):
        import cv2
        a, n = self.cfg["agent"], self.cfg["agent"]["vlm_frames"]
        cap = cv2.VideoCapture(str(self.video))
        out = []
        for t in np.linspace(t0, max(t0, t1 - 0.5), n):
            cap.set(cv2.CAP_PROP_POS_MSEC, float(t) * 1000)
            ok, fr = cap.read()
            if not ok:
                continue
            h0, w0 = fr.shape[:2]
            scale = min(1.0, self.cfg["video"]["max_width"] / w0)
            fr = cv2.resize(fr, (int(w0 * scale), int(h0 * scale)))
            h, w = fr.shape[:2]
            if a["draw_bed_polygon"]:
                cv2.polylines(fr, [(self.bed_px).astype(np.int32).reshape(-1, 1, 2)], True, (0, 255, 255), 2)
            row = self.bins.iloc[min(int(t), len(self.bins) - 1)]
            if row.present > 0.3 and row.body_len > 0:      # crop around patient, otherwise send the full frame
                half = int(row.body_len * a["vlm_crop_scale"] / 2)
                x0, y0 = int(max(0, row.cx - half)), int(max(0, row.cy - half))
                fr = fr[y0:min(h, int(row.cy + half)), x0:min(w, int(row.cx + half))]
            if fr.size == 0:
                continue
            s = 512 / max(fr.shape[:2])
            fr = cv2.resize(fr, (max(1, int(fr.shape[1] * s)), max(1, int(fr.shape[0] * s))))
            out.append(cv2.imencode(".jpg", fr, [cv2.IMWRITE_JPEG_QUALITY, 85])[1].tobytes())
        cap.release()
        return out

    def ask(self, t0, t1, question, options):
        jpgs = self.frames(t0, t1)
        if not jpgs:
            return {"choice": "uncertain", "reason": "no frames could be read"}
        key = hashlib.sha256((self.cfg["agent"]["model"] + question + "|".join(options)).encode()
                             + b"".join(hashlib.sha256(j).digest() for j in jpgs)).hexdigest()
        if key in self.cache:
            return self.cache[key]
        content = [{"type": "image", "source": {"type": "base64", "media_type": "image/jpeg",
                                                "data": base64.b64encode(j).decode()}} for j in jpgs]
        content.append({"type": "text", "text": (
            f"These {len(jpgs)} frames are in time order from {t0:.0f}s to {t1:.0f}s of a fixed indoor camera. "
            "The yellow outline (if drawn) is the bed. Be conservative; if the evidence is insufficient answer 'uncertain'.\n"
            f"Question: {question}\nOptions: {options + ['uncertain']}\n"
            'Reply with JSON only: {"choice": <one option>, "reason": <one short sentence>}')})
        resp = self.client.messages.create(model=self.cfg["agent"]["model"], max_tokens=300,
                                           messages=[{"role": "user", "content": content}])
        text = "".join(b.text for b in resp.content if b.type == "text")
        try:
            res = json.loads(text[text.index("{"): text.rindex("}") + 1])
        except Exception:
            res = {"choice": "uncertain", "reason": f"unparseable reply: {text[:80]}"}
        self.cache[key] = res
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        json.dump(self.cache, open(self.cache_path, "w"))
        return res
