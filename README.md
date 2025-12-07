# Synapsis Jr. AI Engineer Challenge: People Detection + People Tracking + People Counting

This project implements a complete people counting system that integrates people detection and people tracking within user-defined polygonal regions of a video feed. Built with FastAPI, YOLOv8, and ByteTrack, it enables users to upload videos, define custom areas of interest, and monitor in/out movement patterns in real time. The system persists detection and event data to a PostgreSQL database, exposing live analytics through an interactive dashboard.

[Click here to learn more about the project: synapsis-pd-pt-pc/assets/AI Engineer - Challenge.pdf](https://github.com/verneylmavt/synapsis-pd-pt-pc/blob/3f5b52ef32e7e98cc8bc825fdbe27fe2644dff9e/assets/AI%20Engineer%20-%20Challenge.pdf).

## 📁 Project Structure

```
synapsis-pd-pt-pc
│
├─ alembic/                    # Alembic folder
│  ├─ env.py                   # Alembic entry
│  └─ versions/                # Database migration scripts
│
├─ app/
│  ├─ main.py                  # FastAPI app
│  ├─ schemas.py               # Pydantic request/response schemas
│  │
│  ├─ core/
│  │  └─ config.py             # Central settings
│  │
│  ├─ db/
│  │  ├─ models.py             # SQLAlchemy ORM models
│  │  ├─ session.py            # Engine/session helpers
│  │
│  └─ logic/
│     ├─ geometry.py           # _point_on_segment and point_in_polygon
│     └─ hysteresis.py         # Enter/exit debouncing
│
├─ .env                        # Env variables
├─ alembic.ini                 # Alembic config
├─ docker-compose.yml          # Docker compose for PostgreSQL
└─ requirements.txt            # Python deps

```

## 🛢 Database Design

![ER Diagram](https://raw.githubusercontent.com/verneylmavt/synapsis-pd-pt-pc/refs/heads/main/assets/ER%20Diagram.jpg)

- VideoSource: Input Video Stream or File
- Area: Polygon-Shaped Region for Specific Video Source
- Detection: Detected Object per Frame
- Event: “Enter” and “Exit” Occurrence of Tracked Object

To handle video input and detection, the system is designed so that each VideoSource entry in the database represents one input stream. During runtime, the application loads these sources and processes frames using a pretrained YOLOv8 model to perform real-time people detection. Each detected object is stored in the Detection table, which records bounding box coordinates, confidence scores, timestamps, and tracker IDs assigned by ByteTrack. This ensures every frame’s results are traceable to a specific video source.

The relationship between the detection and area configuration tables is established through the inside_area_id foreign key in the Detection table, which references the Area table that stores polygon configurations for each video source. Each Area defines a specific region of interest in polygonal form, allowing the system to determine whether a detection’s centroid lies inside that region. When a detected person enters or exits an area, corresponding records are created in the Event table linked to both the Area and VideoSource.

<!-- ## 🔀 Workflow -->

## 🔌 API

1. **Health Check**
   - `GET /healthz`: to verify that the API server is running and reachable
     ```bash
     curl "http://localhost:8000/healthz"
     ```
2. **Video Source Management**
   - `GET /api/video-sources`: to return a list of all uploaded or registered video sources
     ```bash
     curl "http://localhost:8000/api/video-sources"
     ```
   - `POST /api/upload-video`: to upload a new video file to the server and creates a corresponding VideoSource record in the database
     ```bash
     curl -X POST "http://localhost:8000/api/upload-video" \
     -F "file=@./data/{video_source}.mp4" \
     -F "name={video_source_name}"
     ```
   - `GET /api/video/first-frame/{video_source_id}`: to return the first frame of a specific video as a JPEG image
3. **Polygon Area Management**
   - `GET /api/areas`: to list all defined polygon areas used for people counting
     ```bash
     curl "http://localhost:8000/api/areas"
     ```
   - `POST /api/areas`: to create a new polygon detection area for a given video source
     ```bash
     curl -X POST "http://localhost:8000/api/areas" \
      -H "Content-Type: application/json" \
      -d '{
         "video_source_id": {video_source_id},
         "name": "{area_name}",
         "polygon": [
            {"x":{x1},"y":{y1}},
            {"x":{x2},"y":{y2}},
            {"x":{x3},"y":{y3}},
            {"x":{x4},"y":{y4}}
         ]
      }'
     ```
4. **Real-Time Detection & Streaming**
   - `GET /stream/{video_source_id}`: to run live object detection, tracking, and people counting on a selected video source
5. **People Counting Statistics**
   - `GET /api/stats/live`: to return live statistics of people movement for a specific video source and polygon area
     ```bash
     curl "http://localhost:8000/api/stats/live?video_source_id={video_source_id}&area_id={area_id}&window_seconds={window_seconds}"
     ```
   - `GET /api/stats`: to provide historical statistics for a given video source and area, aggregated over time buckets
     ```bash
     curl "http://localhost:8000/api/stats?video_source_id={video_source_id}&area_id={area_id}&granularity=minute&start={start_ISO_8601}&end={end_ISO_8601}"
     ```
6. **Dashboard**
   - `GET /dashboard`: to return an interactive HTML dashboard for testing and visualization

## 🖥️ Dashboard

![Dashboard](https://raw.githubusercontent.com/verneylmavt/synapsis-pd-pt-pc/refs/heads/main/assets/Dashboard.gif)

## ✨ Features

1. **Database Design** ✔️

   - How It Works?  
     The database serves is built using PostgreSQL and managed through SQLAlchemy ORM with Alembic for version-controlled schema migrations. The schema follows a relational structure with four main tables: video_sources (managing video feed metadata), areas (storing polygonal detection zones in JSONB format), detections (logging per-frame person detections with bounding boxes and confidence scores), and events (recording entry and exit actions within defined areas). Each table is linked by foreign keys to maintain referential integrity, while timestamp fields enable temporal analytics. All database operations are handled through a scoped session manager that ensures transactional safety, and Docker Compose provides a reproducible PostgreSQL environment with an integrated Adminer interface for quick inspection.
   - Challenges  
     Building a database for a real-time video analytics pipeline introduced several key challenges. Managing the high-frequency insertion of detections and events required optimizing session handling and connection pooling to maintain throughput without data loss. Storing large volumes of time-series detection data also raised issues of scalability, necessitating careful indexing and potential partitioning strategies. The use of JSONB for polygon storage offered flexibility but limited spatial query performance compared to dedicated GIS extensions like PostGIS. Furthermore, keeping ORM models and Alembic migrations in sync during iterative development was error-prone. Ensuring that real-time API endpoints reflected the latest database state without overwhelming query loads required a balance between consistency, performance, and efficiency.

2. **Pengumpulan Dataset** ❌  
   No custom dataset was used in this project. Instead, the system relies entirely on a pre-trained YOLOv8n model (yolov8n.pt) from the Ultralytics library. This model is trained on the COCO dataset and specifically configured to detect objects belonging to class 0 (person), which is sufficient for the project's human detection requirements. To handle tracking, the system integrates ByteTrack, a real-time multi-object tracking algorithm that associates detections across consecutive frames. This approach was chosen due to time and computational constraints, training or fine-tuning a custom model would have required significant resources that were unavailable within the project’s schedule.
3. **Object Detection & Tracking** ✔️

   - How It Works?  
     The system uses a pretrained YOLOv8 model to perform real-time people detection on each incoming video frame. YOLO identifies bounding boxes that contain humans and outputs their positions and confidence scores. These detections are then passed into ByteTrack, a lightweight multi-object tracker, which assigns unique IDs to each person and maintains their identity across frames, even when movement or temporary occlusion occurs. Together, YOLO handles object recognition while ByteTrack ensures consistent tracking, allowing the system to follow multiple individuals smoothly throughout the video stream.
   - Challenges  
     Achieving stable tracking can be difficult under real-world conditions. Frequent occlusions, overlapping individuals, and sudden movements may cause the tracker to lose or switch identities. Lighting changes, camera shake, or low-resolution footage can also degrade detection accuracy. Balancing speed and precision is another challenge, as higher accuracy models can slow down real-time performance on limited hardware.

4. **Counting & Polygon Area** ✔️

   - How It Works?  
     After detection and tracking, each person’s bounding box center is checked against pre-defined polygon zones (areas of interest) using a geometric function called point-in-polygon. This determines whether a person is currently inside a monitored area. To prevent flickering or false triggers, a hysteresis mechanism is applied: a person must remain consistently inside or outside for several consecutive frames before being marked as “entered” or “exited.” These events are recorded in the database, allowing the system to count how many people are currently inside or how many have entered or left a zone over time.
   - Challenges  
     Accurately detecting entry and exit events depends heavily on zone geometry and tracking stability. If a person moves along the boundary or the tracker briefly loses them, the system can miscount. Defining complex polygon shapes also introduces computational and geometric edge cases, especially when zones overlap. Maintaining consistent counts under real-world movement patterns and minimizing false positives remain key challenges.

5. **Prediksi (Forecasting)** ❌  
   The documentation and project implementation do not include or describe any forecasting component. Therefore, this section is not applicable.
6. **Integrasi API** ✔

   - How It Works?  
     The system exposes a set of RESTful API endpoints built using FastAPI, allowing interaction between the backend logic, database, and front-end dashboard. These endpoints serve multiple purposes: retrieving stored data, uploading new video sources, defining detection zones, and streaming live analytics. For instance, /api/upload-video saves an uploaded video to the server and records its metadata in the database; /api/areas manages polygon zones tied to each video; and /api/stats/live returns real-time counts of people entering and exiting monitored areas by querying the latest Event records. A specialized /stream/{video_source_id} endpoint continuously processes video frames through YOLOv8 and ByteTrack, performs counting logic, and sends back annotated frames as an MJPEG stream for the web dashboard. This modular design ensures that each API route directly corresponds to a specific function in the detection, tracking, and counting workflow.
   - Challenges  
     Integrating multiple real-time components into a unified API was non-trivial. Since video streaming, detection, and database operations occur simultaneously, synchronization and performance management were key issues. Ensuring the MJPEG stream remained stable while the system performed continuous YOLO inference and database writes required careful handling of asynchronous operations and resource cleanup. Additionally, designing the API to remain stateless, secure, and efficient was challenging. Balancing readability, modularity, and real-time responsiveness without causing latency or blocking operations demanded rigorous testing and optimization.

7. **Deployment** ❌  
   The deployment phase was not completed due to time limitations. While the core system functions correctly in a local environment, there was insufficient time to set up deployment infrastructure, such as hosting the application on cloud platforms like Vercel or AWS.

## ⚙️ Local Setup

0. Make sure to have the prerequisites:

   - Git
   - Python
   - Conda or venv
   - PostgreSQL and Docker
   - FFmpeg
   - NVIDIA Driver + CUDA Toolkit (optional)

1. Clone the repository:

   ```bash
    git clone https://github.com/verneylmavt/synapsis-pd-pt-pc.git
    cd synapsis-pd-pt-pc
   ```

2. Create environment and install dependencies:

   ```bash
   conda create -n synapsis-pd-pt-pc python=3.10 -y
   conda activate synapsis-pd-pt-pc

   pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126
   pip install -r requirements.txt
   ```

3. Start Docker Desktop and run PostgreSQL:

   ```bash
   docker compose up -d
   ```

4. Apply database migrations:

   ```bash
   alembic upgrade head
   ```

5. Run the server:

   ```bash
   uvicorn app.main:app --reload
   ```

6. Open the dashboard:
   ```bash
   start "http://127.0.0.1:8000/dashboard"
   ```
