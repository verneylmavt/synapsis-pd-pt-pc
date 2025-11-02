from __future__ import annotations


from typing import List
from datetime import datetime, timezone, timedelta
from pathlib import Path


import cv2
import numpy as np
from fastapi import FastAPI, Depends, HTTPException, UploadFile, File, Form, Response as FastAPIResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse, Response
from sqlalchemy import select, func, case
from sqlalchemy.orm import Session
from ultralytics import YOLO


from app.core.config import settings

from app.db.models import VideoSource, Area, Event, Detection
from app.db.session import engine, get_db, SessionLocal

from app.schemas import AreaCreate, AreaOut, VideoSourceOut

from app.logic.geometry import point_in_polygon
from app.logic.hysteresis import HysteresisState



app = FastAPI(title=settings.app_name)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/healthz")
def health():
    return {"ok": True, "app": settings.app_name}


@app.get("/api/areas", response_model=List[AreaOut])
def list_areas(
    video_source_id: int | None = None,
    db: Session = Depends(get_db)
):
    stmt = select(Area)
    if video_source_id is not None:
        stmt = stmt.where(Area.video_source_id == video_source_id)
    stmt = stmt.order_by(Area.id.asc())
    rows = db.execute(stmt).scalars().all()
    return rows


@app.post("/api/areas", response_model=AreaOut, status_code=201)
def create_area(payload: AreaCreate, db: Session = Depends(get_db)):
    # Ensure video source exists
    vs = db.get(VideoSource, payload.video_source_id)
    if vs is None:
        raise HTTPException(status_code=400, detail="video_source_id not found")

    area = Area(
        video_source_id=payload.video_source_id,
        name=payload.name,
        polygon=[p.model_dump() for p in payload.polygon],
        active=True,
    )
    db.add(area)
    db.commit()
    db.refresh(area)
    return area


@app.get("/api/video-sources", response_model=List[VideoSourceOut])
def list_video_sources(db: Session = Depends(get_db)):
    stmt = select(VideoSource).order_by(VideoSource.id.asc())
    rows = db.execute(stmt).scalars().all()
    return rows


@app.get("/api/stats/live")
def stats_live(
    video_source_id: int,
    area_id: int,
    window_seconds: float = 10.0,   # <-- float so we can pass 2.5
    db: Session = Depends(get_db),
    response: FastAPIResponse = None,
):
    # prevent caching of live stats in intermediaries/browsers
    if response is not None:
        response.headers["Cache-Control"] = "no-store"

    now_utc = datetime.now(timezone.utc)
    start_window = now_utc - timedelta(seconds=window_seconds)

    # recent window
    recent = db.execute(
        select(
            func.sum(case((Event.type == "enter", 1), else_=0)),
            func.sum(case((Event.type == "exit", 1), else_=0)),
        ).where(
            Event.video_source_id == video_source_id,
            Event.area_id == area_id,
            Event.timestamp >= start_window,
            Event.timestamp <= now_utc,
        )
    ).one()
    in_recent = int(recent[0] or 0)
    out_recent = int(recent[1] or 0)

    # full net occupancy = total enters - total exits
    totals = db.execute(
        select(
            func.sum(case((Event.type == "enter", 1), else_=0)),
            func.sum(case((Event.type == "exit", 1), else_=0)),
        ).where(
            Event.video_source_id == video_source_id,
            Event.area_id == area_id
        )
    ).one()
    total_in = int(totals[0] or 0)
    total_out = int(totals[1] or 0)
    currently_inside = max(0, total_in - total_out)

    return {
        "video_source_id": video_source_id,
        "area_id": area_id,
        "ts": now_utc.isoformat(),
        "window_seconds": window_seconds,
        "in_recent": in_recent,
        "out_recent": out_recent,
        "currently_inside": currently_inside,
        "totals": {"in": total_in, "out": total_out},
    }

def _parse_iso8601_utc(s: str) -> datetime:
    # Minimal ISO8601 parser that handles trailing 'Z'
    if not s:
        raise ValueError("empty datetime")
    s = s.strip()
    if s.endswith("Z"):
        s = s[:-1]
        dt = datetime.fromisoformat(s)
        return dt.replace(tzinfo=timezone.utc)
    dt = datetime.fromisoformat(s)
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)

@app.get("/api/stats")
def stats(
    video_source_id: int,
    area_id: int,
    start: str | None = None,
    end: str | None = None,
    granularity: str = "minute",
    db: Session = Depends(get_db),
):
    """
    Historical counts for a given video_source_id + area_id,
    bucketed by minute|hour|day.
    """
    granularity = granularity.lower().strip()
    if granularity not in {"minute", "hour", "day"}:
        raise HTTPException(status_code=400, detail="granularity must be 'minute'|'hour'|'day'")

    now_utc = datetime.now(timezone.utc)
    end_dt = _parse_iso8601_utc(end) if end else now_utc

    if start is None:
        defaults = {"minute": timedelta(minutes=60), "hour": timedelta(hours=24), "day": timedelta(days=30)}
        start_dt = end_dt - defaults[granularity]
    else:
        start_dt = _parse_iso8601_utc(start)

    if start_dt > end_dt:
        raise HTTPException(status_code=400, detail="'start' must be <= 'end'")

    bucket = func.date_trunc(granularity, Event.timestamp).label("bucket")
    q = (
        select(
            bucket,
            func.sum(case((Event.type == "enter", 1), else_=0)).label("in_count"),
            func.sum(case((Event.type == "exit", 1), else_=0)).label("out_count"),
        )
        .where(
            Event.video_source_id == video_source_id,
            Event.area_id == area_id,
            Event.timestamp >= start_dt,
            Event.timestamp <= end_dt,
        )
        .group_by(bucket)
        .order_by(bucket.asc())
    )

    rows = db.execute(q).all()
    buckets = [
        {
            "window_start": (r.bucket if isinstance(r.bucket, datetime) else r.bucket).replace(tzinfo=timezone.utc).isoformat(),
            "in_count": int(r.in_count or 0),
            "out_count": int(r.out_count or 0),
        }
        for r in rows
    ]

    total_in = sum(b["in_count"] for b in buckets)
    total_out = sum(b["out_count"] for b in buckets)

    return {
        "video_source_id": video_source_id,
        "area_id": area_id,
        "granularity": granularity,
        "start": start_dt.isoformat(),
        "end": end_dt.isoformat(),
        "buckets": buckets,
        "summary": {"total_in": total_in, "total_out": total_out},
    }

@app.post("/api/upload-video")
async def upload_video(
    file: UploadFile = File(...),
    name: str = Form(None)
):
    # Ensure upload dir exists
    upload_dir = Path("data") / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)

    # Save file
    safe_name = file.filename.replace("\\", "_").replace("/", "_")
    out_path = upload_dir / f"{int(datetime.now().timestamp())}_{safe_name}"
    with open(out_path, "wb") as f:
        while True:
            chunk = await file.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)

    # Create VideoSource row
    db: Session = SessionLocal()
    try:
        vs = VideoSource(
            name=name or safe_name,
            uri=str(out_path.as_posix()),
            enabled=True,
        )
        db.add(vs)
        db.commit()
        db.refresh(vs)
        return {
            "id": vs.id,
            "name": vs.name,
            "uri": vs.uri
        }
    finally:
        db.close()


@app.get("/api/video/first-frame/{video_source_id}")
def first_frame(video_source_id: int):
    db: Session = SessionLocal()
    try:
        vs = db.get(VideoSource, video_source_id)
        if not vs:
            raise HTTPException(status_code=404, detail="video source not found")
        cap = cv2.VideoCapture(vs.uri)
        if not cap.isOpened():
            raise HTTPException(status_code=500, detail="failed to open video")
        ok, frame = cap.read()
        cap.release()
        if not ok or frame is None:
            raise HTTPException(status_code=500, detail="failed to read first frame")
        ok, jpg = cv2.imencode(".jpg", frame)
        if not ok:
            raise HTTPException(status_code=500, detail="failed to encode image")
        return Response(content=jpg.tobytes(), media_type="image/jpeg")
    finally:
        db.close()


def _normalize_xyxy(xyxy: np.ndarray, w: float, h: float) -> np.ndarray:
    """
    Normalize bounding box coordinates to the [0, 1] range
    relative to image width and height.

    Input:
        xyxy: N x 4 NumPy array of bounding boxes in pixel coordinates
            Each row represents [x1, y1, x2, y2] in absolute pixels.
        w: image width in pixels
        h: image height in pixels

    Output:
        N x 4 NumPy array of normalized bounding boxes where
        all coordinates are between 0 and 1.
    """
    # Make a float copy of the original array to avoid modifying it directly
    out = xyxy.copy().astype(np.float32)

    # Normalize x-coordinates (x1 and x2) by dividing by image width.
    # Use max(w, 1.0) to avoid division by zero in case width is invalid.
    out[:, [0, 2]] /= max(w, 1.0)

    # Normalize y-coordinates (y1 and y2) by dividing by image height.
    # Use max(h, 1.0) for the same reason as above.
    out[:, [1, 3]] /= max(h, 1.0)

    # Clip all normalized coordinates to [0, 1]
    # This ensures no value falls outside the valid range
    # due to rounding or floating-point inaccuracies.
    np.clip(out, 0.0, 1.0, out=out)

    # Return the normalized bounding boxes
    return out


def _poly_to_pixels(poly_norm: list[dict], w: int, h: int) -> np.ndarray:
    """
    Convert a polygon’s normalized coordinates (in range [0, 1])
    back to absolute pixel coordinates based on image size.

    Input:
        poly_norm: list of dictionaries, each with {"x": float, "y": float}
                describing polygon vertices normalized to [0, 1].
        w: image width in pixels
        h: image height in pixels

    Output:
        NumPy array of integer pixel coordinates (N x 2)
        suitable for drawing polygons on images (e.g., cv2.polylines()).
    """
    pts = []  # list to hold all converted pixel coordinates

    for p in poly_norm:
        # Convert normalized x, y values back to pixel coordinates.
        # Multiply by width and height respectively, then round to nearest integer.
        # This effectively scales polygon coordinates to match the image resolution.
        #
        # Example: if p["x"] = 0.5 and w = 640, → x ≈ 320
        #          if p["y"] = 0.25 and h = 480, → y ≈ 120
        #
        # After rounding, we ensure the coordinates remain inside image bounds.

        # Scale normalized x to pixels and clamp between [0, w - 1]
        x = min(w - 1, max(0, int(round(p["x"] * w))))

        # Scale normalized y to pixels and clamp between [0, h - 1]
        y = min(h - 1, max(0, int(round(p["y"] * h))))

        # Append the clamped pixel coordinates to the list
        pts.append([x, y])

    # Convert the list of points into a NumPy array of shape (N, 2)
    # with integer data type — required by OpenCV for drawing functions.
    return np.array(pts, dtype=np.int32)


@app.get("/stream/{video_source_id}")
def stream(video_source_id: int, conf: float = 0.25, iou: float = 0.45, device: str = ""):
    """
    Stream endpoint for real-time people detection, tracking, and counting.

    This function:
    - Loads the configured video source (camera or file) and its active detection zones.
    - Runs a YOLOv8 + ByteTrack pipeline to detect and track people.
    - For each frame, logs detections and entry/exit events to the database.
    - Applies hysteresis filtering to ensure stable entry/exit decisions.
    - Streams annotated MJPEG frames (video with bounding boxes and zones) to the client.

    Parameters:
        video_source_id (int): The database ID of the video source.
        conf (float): Minimum detection confidence threshold for YOLO (default = 0.25).
        iou (float): Intersection-over-Union threshold for non-max suppression (default = 0.45).
        device (str): Compute device ("cpu", "cuda", etc.); empty string = auto-detect.

    Returns:
        StreamingResponse: A live MJPEG stream containing annotated video frames.
    """

    # Create a new database session
    db: Session = SessionLocal()
    try:
        # Retrieve the video source record (camera stream or file)
        vs = db.get(VideoSource, video_source_id)
        if not vs:
            db.close()
            raise HTTPException(status_code=404, detail="video source not found")

        # Fetch all "active" polygon areas associated with this video source.
        # Each area defines a region of interest for people counting.
        areas: List[Area] = db.execute(
            select(Area).where(Area.video_source_id == video_source_id, Area.active == True)  # noqa: E712
        ).scalars().all()

        # If no areas are configured, there's nothing to monitor.
        if not areas:
            db.close()
            raise HTTPException(status_code=400, detail="no active areas for this video source")

        # Initialize the YOLOv8 detector (Nano version for speed; can be replaced by larger variants).
        model = YOLO("yolov8n.pt")

        # Begin streaming with ByteTrack tracker enabled.
        # This continuously yields frames and detection results.
        result_iter = model.track(
            source=vs.uri,            # Stream URI (e.g., camera URL or file path)
            stream=True,              # Enable live streaming
            tracker="bytetrack.yaml", # Use ByteTrack multi-object tracker
            classes=[0],              # Detect only "person" class (ID=0)
            device=device or None,    # Auto-detect device if not specified
            conf=conf,                # Detection confidence threshold
            iou=iou,                  # IoU threshold for overlapping detections
            verbose=False,            # Disable verbose YOLO output
            persist=True,             # Keep track IDs persistent across frames
        )
        print(device)

        # Maintain per-person-per-area hysteresis states
        # Key: (tracker_id, area_id) → Value: HysteresisState
        state: dict[tuple[int, int], HysteresisState] = {}

        # Prepare headers for MJPEG (multipart image stream)
        boundary = b"--frame"
        headers = {
            "Content-Type": "multipart/x-mixed-replace; boundary=frame",
            "Cache-Control": "no-store",
            "Pragma": "no-cache",
            "Expires": "0",
        }

        # Generator that yields frames as JPEG chunks in real-time
        def gen():
            nonlocal db, state
            try:
                for result in result_iter:
                    # Skip invalid frames or frames without detections
                    if result is None or result.boxes is None:
                        continue

                    # Extract original image (BGR format) and its dimensions
                    frame = result.orig_img
                    h, w = frame.shape[:2]
                    ts = datetime.now(timezone.utc)  # Current UTC timestamp for logging

                    # Extract YOLO detection outputs: boxes, confidences, and tracking IDs
                    xyxy = result.boxes.xyxy   # Bounding boxes (x1, y1, x2, y2)
                    confs = result.boxes.conf  # Confidence scores
                    ids = result.boxes.id      # Unique track IDs from ByteTrack

                    # If no detections, draw only area polygons and send frame as-is
                    if xyxy is None or len(xyxy) == 0:
                        for area in areas:
                            pts = _poly_to_pixels(area.polygon, w, h)
                            cv2.polylines(frame, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
                        ok, jpg = cv2.imencode(".jpg", frame)
                        if not ok: 
                            continue
                        # Construct MJPEG chunk for streaming
                        chunk = (boundary + b"\r\n" +
                                b"Content-Type: image/jpeg\r\n" +
                                b"Content-Length: " + str(len(jpg)).encode() + b"\r\n\r\n" +
                                jpg.tobytes() + b"\r\n")
                        yield chunk
                        continue

                    # Convert tensors to NumPy for easier manipulation (if needed)
                    if hasattr(xyxy, "cpu"):
                        xyxy = xyxy.cpu().numpy()
                        confs = confs.cpu().numpy() if confs is not None else None
                        track_ids = (ids.cpu().numpy().astype(int).tolist() 
                                    if ids is not None else [None] * xyxy.shape[0])
                    else:
                        xyxy = np.array(xyxy)
                        confs = np.array(confs) if confs is not None else None
                        track_ids = list(map(int, ids)) if ids is not None else [None] * xyxy.shape[0]

                    # Normalize bounding boxes to [0,1] for database storage (resolution-independent)
                    nxyxy = _normalize_xyxy(xyxy, w, h)

                    # Save each detection to the database
                    for i in range(nxyxy.shape[0]):
                        x1, y1, x2, y2 = map(float, nxyxy[i].tolist())
                        tid = track_ids[i] if track_ids[i] is not None else None

                        # Compute center point of detection (used for inside-area testing)
                        cx = (x1 + x2) * 0.5
                        cy = (y1 + y2) * 0.5
                        inside_area_id = None

                        # Determine if this detection center lies within any defined area
                        for area in areas:
                            if point_in_polygon((cx, cy), area.polygon):
                                inside_area_id = area.id
                                break

                        # Create Detection record for logging
                        det = Detection(
                            video_source_id=video_source_id,
                            timestamp=ts,
                            tracker_id=str(tid) if tid is not None else None,
                            cls="person",
                            conf=float(confs[i]) if confs is not None else None,
                            x1=x1, y1=y1, x2=x2, y2=y2,
                            inside_area_id=inside_area_id,
                        )
                        db.add(det)

                    # ---- Entry/Exit Event Detection (People Counting) ----
                    for i in range(nxyxy.shape[0]):
                        if track_ids[i] is None:
                            continue
                        tid = int(track_ids[i])
                        x1, y1, x2, y2 = map(float, nxyxy[i].tolist())
                        cx = (x1 + x2) * 0.5
                        cy = (y1 + y2) * 0.5

                        for area in areas:
                            # Each (person, area) pair gets its own hysteresis tracker
                            key = (tid, area.id)
                            st = state.get(key) or HysteresisState(inside=False)

                            # Update hysteresis with current inside/outside reading
                            # "enter" triggers after k consecutive inside frames
                            # "exit" triggers after k consecutive outside frames
                            ev_type = st.update(now_inside=point_in_polygon((cx, cy), area.polygon), k=3)
                            state[key] = st

                            # Log entry/exit events in database
                            if ev_type == "enter":
                                db.add(Event(video_source_id=video_source_id, area_id=area.id,
                                            timestamp=ts, tracker_id=str(tid), type="enter"))
                            elif ev_type == "exit":
                                db.add(Event(video_source_id=video_source_id, area_id=area.id,
                                            timestamp=ts, tracker_id=str(tid), type="exit"))

                    # Commit all detections and events for this frame
                    db.commit()

                    # ---- Visualization (for MJPEG output) ----
                    annotated = frame.copy()  # Copy original frame for annotation

                    N = nxyxy.shape[0]
                    inside_mask = [False] * N

                    # Determine which detections are currently inside any polygon
                    for i in range(N):
                        x1n, y1n, x2n, y2n = map(float, nxyxy[i].tolist())
                        cx = (x1n + x2n) * 0.5
                        cy = (y1n + y2n) * 0.5
                        inside_any = False
                        for area in areas:
                            if point_in_polygon((cx, cy), area.polygon):
                                inside_any = True
                                break
                        inside_mask[i] = inside_any

                    # Draw bounding boxes and labels only for people inside a monitored area
                    for i in range(N):
                        if not inside_mask[i]:
                            continue
                        x1, y1, x2, y2 = map(int, xyxy[i].tolist())
                        tid = track_ids[i] if track_ids[i] is not None else None
                        label = f"id:{tid}" if tid is not None else "id:-"
                        if confs is not None:
                            label += f"  {float(confs[i]):.2f}"

                        # Draw blue bounding box
                        cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 0, 0), 2)
                        # Add text label above the box
                        cv2.putText(annotated, label, (x1, max(0, y1 - 6)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)

                    # Draw the defined polygon zones in green
                    for area in areas:
                        pts = _poly_to_pixels(area.polygon, w, h)
                        cv2.polylines(annotated, [pts], isClosed=True, color=(0, 255, 0), thickness=2)

                    # Convert annotated frame to JPEG and yield as MJPEG chunk
                    ok, jpg = cv2.imencode(".jpg", annotated)
                    if not ok:
                        continue

                    yield (boundary + b"\r\n" +
                        b"Content-Type: image/jpeg\r\n" +
                        b"Content-Length: " + str(len(jpg)).encode() + b"\r\n\r\n" +
                        jpg.tobytes() + b"\r\n")
            finally:
                # Ensure DB session is closed when client disconnects or stream ends
                db.close()

        # Return HTTP streaming response (live video)
        return StreamingResponse(gen(), headers=headers)

    # Handle HTTP-specific exceptions
    except HTTPException:
        db.close()
        raise

    # Handle unexpected runtime errors
    except Exception as e:
        db.close()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/dashboard", response_class=HTMLResponse)
def dashboard():
    html = """
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>Synapsis AI Engineer Challenge</title>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<style>
    body { font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif; margin: 20px; }
    .row { display: flex; gap: 16px; align-items: flex-start; flex-wrap: wrap; }
    .col { display: flex; flex-direction: column; gap: 12px; }
    .card { border: 1px solid #ddd; border-radius: 6px; padding: 12px 16px; }
    label { font-size: 14px; }
    select, input, button { padding: 6px 8px; }
    canvas { border: 1px solid #ddd; }
    #streamImg { max-width: 900px; border: 1px solid #ddd; display:block; }
    /* Make the chart container wider and the canvas taller */
    #chartWrap { width: 100%; max-width: 1800px; }
    #histChart { width: 100% !important; height: 480px !important; }
</style>
</head>
<body>
<h2>People Detection + People Tracking + People Counting</h2>

<div class="card">
    <div class="row">
    <div class="col">
        <div><strong>1) Upload Video</strong></div>
        <input type="file" id="videoFile" accept="video/mp4,video/*"/>
        <input type="text" id="videoName" placeholder="Video Name"/>
        <button id="btnUpload">Upload</button>
        <div id="uploadStatus" style="color:#666;"></div>
    </div>

    <div class="col">
        <div><strong>2) Pick Your Shape of Polygon by Adding Points! </strong> (Click to Add Points; Points ≥3)</div>
        <canvas id="frameCanvas" width="640" height="360"></canvas>
        <div class="row">
        <input type="text" id="areaName" placeholder="Area Name (e.g., Pentagon)"/>
        <button id="btnClear">Clear Points</button>
        <button id="btnClose">Close Points</button>
        <button id="btnSavePoly">Save Polygon</button>
        </div>
        <div id="polyStatus" style="color:#666;"></div>
    </div>

    <div class="col">
        <div><strong>3) Start CV!</strong></div>
        <button id="btnStartStream">Start</button>
        <img id="streamImg"/>
    </div>
    </div>
</div>

<div class="card" style="margin-top:16px;">
    <div class="row">
    <div class="col">
        <label><strong>Polygon</strong></label>
        <select id="area"></select>
        <div id="areaLabel" style="color:#666;"></div>
    </div>
    <div class="col" id="chartWrap">
        <label><strong>Chart of Counts (Sampled per 2.5s)</strong></label>
        <canvas id="histChart"></canvas>
    </div>
    </div>
</div>

<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<script>
    const $ = (id) => document.getElementById(id);

    // --- Upload / polygon canvas state ---
    let uploadedVSid = null;
    let firstFrameLoaded = false;
    const canvas = $('frameCanvas');
    const ctx = canvas.getContext('2d');
    let points = [];
    let polygonClosed = false;
    let imgForCanvas = null;

    // last saved area (auto-select)
    let lastSavedAreaId = null;
    let lastSavedAreaName = null;

    // streaming & chart state
    let streaming = false;
    let pollTimer = null;         // 2.5s interval
    const SAMPLE_SECONDS = 2.5;   // window size for /api/stats/live
    // end-of-stream / idle detection
    let consecutiveIdle = 0;      // how many consecutive polls had no activity
    let consecutiveFetchFailures = 0;
    let lastInside = null;        // last "currently_inside" value we saw

    // Chart series
    let chart = null;
    let labels = [];
    let seriesIn = [];
    let seriesOut = [];
    let seriesInside = [];

    function drawCanvas() {
    if (!imgForCanvas) return;
    ctx.clearRect(0,0,canvas.width, canvas.height);
    ctx.drawImage(imgForCanvas, 0, 0, canvas.width, canvas.height);
    if (points.length > 0) {
        ctx.beginPath();
        ctx.moveTo(points[0][0], points[0][1]);
        for (let i=1;i<points.length;i++) ctx.lineTo(points[i][0], points[i][1]);
        if (polygonClosed && points.length >= 3) {
        ctx.lineTo(points[0][0], points[0][1]);
        ctx.strokeStyle = 'lime'; ctx.lineWidth = 2; ctx.stroke();
        ctx.globalAlpha = 0.12; ctx.fillStyle = 'lime'; ctx.fill(); ctx.globalAlpha = 1.0;
        } else {
        ctx.strokeStyle = 'lime'; ctx.lineWidth = 2; ctx.stroke();
        }
        ctx.fillStyle = 'red';
        for (const [x,y] of points) { ctx.beginPath(); ctx.arc(x,y,3,0,2*Math.PI); ctx.fill(); }
    }
    }

    canvas.addEventListener('click', (e) => {
    if (!firstFrameLoaded || polygonClosed) return;
    const r = canvas.getBoundingClientRect();
    points.push([e.clientX - r.left, e.clientY - r.top]);
    drawCanvas();
    });

    $('btnClear').addEventListener('click', () => {
    points = []; polygonClosed = false; drawCanvas(); $('polyStatus').textContent = '';
    });

    $('btnClose').addEventListener('click', () => {
    if (points.length < 3) { $('polyStatus').textContent = 'Need at least 3 points to close.'; return; }
    polygonClosed = true; $('polyStatus').textContent = 'Polygon is closed. You can now click "Save Polygon".'; drawCanvas();
    });

    $('btnUpload').addEventListener('click', async () => {
    const f = $('videoFile').files[0];
    if (!f) { $('uploadStatus').textContent = 'Please choose a file.'; return; }
    const fd = new FormData(); fd.append('file', f);
    const name = $('videoName').value || ''; if (name) fd.append('name', name);
    $('uploadStatus').textContent = 'Uploading...';
    const res = await fetch('/api/upload-video', { method: 'POST', body: fd });
    if (!res.ok) { $('uploadStatus').textContent = 'Upload failed.'; return; }
    const data = await res.json();
    uploadedVSid = data.id;
    $('uploadStatus').textContent = `Video is uploaded as: video_source_id=${uploadedVSid}`;

    // first frame
    const img = new Image();
    img.onload = () => { imgForCanvas = img; firstFrameLoaded = true; points=[]; polygonClosed=false; drawCanvas(); $('polyStatus').textContent = 'If u want to clear the Shape, click "Clear Points". If u want to close the Shape, click "Close Points".'; };
    img.onerror = () => { $('polyStatus').textContent = 'Failed to load first frame.'; };
    img.src = `/api/video/first-frame/${uploadedVSid}`;

    // load areas for this video
    await loadAreas(uploadedVSid);
    });

    $('btnSavePoly').addEventListener('click', async () => {
    if (!uploadedVSid) { $('polyStatus').textContent = 'Upload a video first.'; return; }
    if (points.length < 3) { $('polyStatus').textContent = 'Need at least 3 points.'; return; }
    if (!polygonClosed) { $('polyStatus').textContent = 'Please click "Close Points" before saving.'; return; }

    const poly = points.map(([x,y]) => ({ x: x/canvas.width, y: y/canvas.height }));
    const areaName = $('areaName').value || `area-${uploadedVSid}`;
    const res = await fetch('/api/areas', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({ video_source_id: uploadedVSid, name: areaName, polygon: poly })
    });
    if (!res.ok) { const t = await res.text(); $('polyStatus').textContent = 'Failed to save polygon: ' + t; return; }
    const saved = await res.json();
    lastSavedAreaId = saved.id; lastSavedAreaName = saved.name;
    $('polyStatus').textContent = `Polygon is saved as: area_id=${saved.id}`;

    await loadAreas(uploadedVSid);
    $('area').value = String(saved.id);
    $('areaLabel').textContent = `(Polygon: area_id=${saved.id})`;
    });

    async function loadAreas(vsid) {
    const res = await fetch(`/api/areas?video_source_id=${vsid}`);
    const areas = await res.json();
    const sel = $('area'); sel.innerHTML = '';
    for (const a of areas) {
        const opt = document.createElement('option');
        opt.value = a.id; opt.textContent = `${a.id} – ${a.name}`;
        sel.appendChild(opt);
    }
    if (areas.length > 0) {
        const idToUse = lastSavedAreaId ? String(lastSavedAreaId) : String(areas[0].id);
        sel.value = idToUse;
        const picked = areas.find(x => String(x.id) === idToUse);
        $('areaLabel').textContent = `(selected area_id=${idToUse}${picked ? ', ' + picked.name : ''})`;
    } else {
        $('areaLabel').textContent = '(no areas yet)';
    }
    }

    // --- streaming & chart ---
    $('btnStartStream').addEventListener('click', async () => {
    if (!uploadedVSid) { alert('Upload a video first.'); return; }
    if (lastSavedAreaId) { $('area').value = String(lastSavedAreaId); $('areaLabel').textContent = `(selected area_id=${lastSavedAreaId}${lastSavedAreaName ? ', ' + lastSavedAreaName : ''})`; }

    // reset chart series
    labels = []; seriesIn = []; seriesOut = []; seriesInside = [];
    initChart(); // clears canvas
    consecutiveIdle = 0;
    consecutiveFetchFailures = 0;
    lastInside = null;

    // start stream
    const img = $('streamImg');
    // Stop polling when the stream ends or is aborted. Different browsers behave differently,
    // so register several signals.
    img.onerror = handleStreamEnd;
    img.onabort = handleStreamEnd;
    // Not universally supported for <img>, but harmless if ignored:
    img.onloadend = handleStreamEnd;

    img.src = `/stream/${uploadedVSid}`;

    // start polling every 2.5 s
    streaming = true;
    if (pollTimer) clearInterval(pollTimer);
    await sampleOnce(); // first sample immediately
    pollTimer = setInterval(sampleOnce, 2500);
    });

    async function sampleOnce() {
    if (!streaming || !uploadedVSid) return;
    const areaId = $('area').value;
    if (!areaId) return;

    let data;
    try {
        const res = await fetch(
        `/api/stats/live?video_source_id=${uploadedVSid}&area_id=${areaId}&window_seconds=${SAMPLE_SECONDS}`,
        { cache: 'no-store' }
        );
        if (!res.ok) {
        consecutiveFetchFailures++;
        if (consecutiveFetchFailures >= 2) handleStreamEnd();
        return;
        }
        data = await res.json();
        consecutiveFetchFailures = 0; // reset on success
    } catch (e) {
        // Network-level failure; assume stream died if it happens twice
        consecutiveFetchFailures++;
        if (consecutiveFetchFailures >= 2) handleStreamEnd();
        return;
    }

    const tsLabel = new Date().toLocaleTimeString();
    labels.push(tsLabel);
    seriesIn.push(data.in_recent || 0);
    seriesOut.push(data.out_recent || 0);
    seriesInside.push(data.currently_inside || 0);

    // Idle watchdog: if no activity (no enters/exits) and occupancy hasn't changed
    // for ~10 seconds, assume the stream ended and stop polling.
    const hadDelta = (data.in_recent || 0) > 0 || (data.out_recent || 0) > 0;
    const insideChanged = lastInside === null || data.currently_inside !== lastInside;
    lastInside = data.currently_inside;
    if (!hadDelta && !insideChanged) {
        consecutiveIdle++;
    } else {
        consecutiveIdle = 0;
    }
    // 10s / 2.5s = 4 polls with no activity
    if (consecutiveIdle >= Math.ceil(10 / SAMPLE_SECONDS)) {
        handleStreamEnd();
    }

    updateChart();
    }

    function handleStreamEnd() {
    // Server closed MJPEG (video ended)
    streaming = false;
    if (pollTimer) { clearInterval(pollTimer); pollTimer = null; }
    // Explicitly release the image stream
    const img = $('streamImg');
    if (img) img.src = '';
    // Nothing else to do; chart already holds full session (start -> end)
    }

    function initChart() {
    const ctx2 = $('histChart').getContext('2d');
    if (chart) { chart.destroy(); chart = null; }
    chart = new Chart(ctx2, {
        type: 'line',
        data: {
        labels: labels,
        datasets: [
            { label: 'In (Δ per 2.5s)', data: seriesIn },
            { label: 'Out (Δ per 2.5s)', data: seriesOut },
            { label: 'Currently Inside', data: seriesInside },
        ]
        },
        options: {
        responsive: true,
        animation: false,
        maintainAspectRatio: false, // allow CSS height to control the canvas
        interaction: { mode: 'index', intersect: false },
        scales: { y: { beginAtZero: true } }
        }
    });
    }

    function updateChart() {
    if (!chart) return;
    chart.update();
    }

    $('area').addEventListener('change', () => {
    const opt = $('area').selectedOptions[0];
    if (opt) $('areaLabel').textContent = `(selected area_id=${opt.value}, ${opt.text.split('–').slice(1).join('–').trim()})`;
    // If streaming, we’ll continue sampling for the newly selected area
    });
</script>
</body>
</html>
    """
    return HTMLResponse(html)