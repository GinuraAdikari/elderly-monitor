def probe(video) -> dict:
    import cv2
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {video} (try: ffmpeg -i in.avi -c:v libx264 out.mp4)")
    fps = cap.get(cv2.CAP_PROP_FPS) or 12.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return {"fps": fps, "n_frames": n, "width": w, "height": h, "duration": n / fps}
