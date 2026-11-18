# PathSnap: Map-Matching & Route Reconstruction

UE24CS352A Machine Learning Hackathon, PES University

PathSnap takes noisy GPS points from a vehicle and works out which roads it actually drove on. Each GPS point gets a shortlist of nearby road segments, an ML classifier scores them, and an HMM with Viterbi decoding picks the most likely connected route. The final system runs as a FastAPI backend (Laptop 1) and a UI (Laptop 2).

## Team

| Role | Member | Owns |
|---|---|---|
| P1 – Data & features | _name_ | loaders, candidate generation, labels, features |
| P2 – EDA & models | _name_ | EDA, 5+ models, comparison, Bayesian optimisation |
| P3 – HMM/Viterbi & maps | _name_ | connectivity graph, Viterbi decoding, Folium maps |
| P4 – API, UI & integration | _name_ | FastAPI, UI, two-laptop deployment, README/report |

## Dataset

Each trip folder (`train/<id>/`, `student_test/<id>/`) contains:

| File | Columns (no header) | Meaning |
|---|---|---|
| `.track` | lon, lat, timestamp (s) | GPS observations, ~1 Hz |
| `.nodes` | lon, lat | road nodes; row index = node id |
| `.arcs` | from_node, to_node | directed road segments; row index = arc id |
| `.route` | arc_id | ground-truth arcs in travel order |

Partitions: 80 train trips, 10 student-test trips, 10 evaluator trips (held by the organisers and never used during development).

The raw data is not committed. Place it at `data/raw/train/` and `data/raw/student_test/`.

## Repository structure

```
pathsnap/
├── data/              # raw data (gitignored) + generated candidate CSVs
├── eda/               # EDA notebook, plots, Folium maps
├── features/          # loaders, candidate generation, labelling, feature engineering
├── models/            # training scripts, CV results, comparison plots, saved weights
├── tuning/            # Bayesian optimisation (Optuna) code + trial logs
├── hmm/               # connectivity graph, Viterbi decoder, route maps
├── api/               # FastAPI backend
├── ui/                # evaluator-facing UI
├── results/           # CSVs, metrics, experiment logs
├── report/            # technical report PDF + figures
├── requirements.txt
└── README.md
```

## Pipeline

1. **Load & project.** Read the trip files and convert lon/lat to local metres (equirectangular projection around the trip centre).
2. **Candidates.** For each GPS point, take up to K road arcs within R metres (KD-tree over densified arcs).
3. **Labels.** Mark the candidate that matches the reference route arc for that point.
4. **Features.** Perpendicular distance, heading difference, speed, arc length, distance rank, etc.
5. **Classifier.** Outputs P(candidate is the true arc). At least 5 models are compared with GroupKFold (grouped by trip).
6. **Bayesian optimisation.** Optuna tuning of the selected model on the 80 training trips.
7. **HMM + Viterbi.** Hidden states are candidate arcs. Emissions come from classifier probabilities, and transitions from road connectivity (arc hops). Viterbi returns the most likely connected arc sequence.
8. **Serve.** FastAPI runs the pipeline for a chosen trip and computes accuracy server-side. The UI shows the route on a map.

## Setup

```bash
git clone <repo-url>
cd pathsnap
python -m venv venv
venv\Scripts\activate        # Windows  (source venv/bin/activate on Linux/Mac)
pip install -r requirements.txt
```

Main dependencies: numpy, pandas, scipy, scikit-learn, lightgbm, optuna, folium, fastapi, uvicorn, streamlit, requests.

## Running

**Generate candidates and features**
```bash
python features/build_candidates.py --split train
python features/build_candidates.py --split student_test
```

**Train and compare models**
```bash
python models/train_all.py
python tuning/optimise.py
```

**Route reconstruction on one trip**
```bash
python hmm/p3_hmm.py data/raw/train/00000005
```

**API (Laptop 1)**
```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

**UI (Laptop 2)**
```bash
streamlit run ui/app.py -- --api http://<laptop1-ip>:8000
```
Both laptops must be on the same network. Allow port 8000 through the Windows firewall on Laptop 1.

## API

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | System health, model readiness, evaluator data availability |
| `/trips` | GET | List of available organizer-provided evaluator trip IDs only |
| `/predict/{trip_id}` | GET | Executes end-to-end inference without ground truth; returns route geometry, continuity, and post-prediction server-side evaluation metrics |

Every request is logged to stdout on Laptop 1 with timestamp, client IP, trip ID, response status, and latency. Ground-truth reference routes stay strictly on the server and are only inspected after inference is complete to compute accuracy metrics.

---

## P4 Implementation: Integration, Backend API & UI

### 1. P4 System Architecture

```
[ Laptop 2: Streamlit UI ]
       │
       │ HTTP (GET /health, GET /trips, GET /predict/{id})
       ▼
[ Laptop 1: FastAPI Backend ]
       │
       ├── Model Service (Pre-loads LightGBM baseline artifact: 19 features)
       ├── Trip Service (Validates evaluator directory & trip structure)
       │
       ├── Inference Pipeline (P1 + P2 + P3)
       │    ├── 1. features.loaders.load_trip(with_route=False) [P1]
       │    ├── 2. features.candidates.generate(trip)           [P1]
       │    ├── 3. features.features.build_features(trip, cand) [P1]
       │    ├── 4. model.predict_proba(feats[19_cols])[:, 1]    [P2]
       │    └── 5. hmm.decoder.decode_trip() Viterbi            [P3]
       │
       └── Server-Side Evaluation Service (Zero Data Leakage)
            ├── Evaluates against reference route ONLY after inference
            ├── Computes Top-1 Accuracy, Macro F1, Miss Rate, Continuity
            └── Strips all reference labels from API response
```

### 2. Environment Variables & Configuration

The backend and frontend can be configured via environment variables or a `.env` file at the repository root:

| Variable | Default | Purpose |
|---|---|---|
| `PATHSNAP_MODEL_PATH` | `models/lightgbm_baseline.joblib` | Path to trained model artifact |
| `PATHSNAP_EVALUATOR_DIR` | `data/raw/evaluator` | Directory where organizer evaluator trips are mounted |
| `PATHSNAP_HOST` | `0.0.0.0` | Backend bind host |
| `PATHSNAP_PORT` | `8000` | Backend bind port |
| `PATHSNAP_W_EMIT` | `1.0` | HMM emission weight factor |
| `PATHSNAP_LOG_LEVEL` | `INFO` | Backend log verbosity |
| `PATHSNAP_API_URL` | `http://localhost:8000` | Frontend target URL for Laptop 1 backend |

### 3. Two-Laptop Deployment Guide

Follow these steps to deploy PathSnap across two laptops during the evaluation:

#### Step 1: Connect Both Laptops to the Same Local Network
- Connect Laptop 1 (Backend) and Laptop 2 (UI) to the same Wi-Fi network or mobile hotspot.
- Alternatively, connect via an Ethernet cable or direct ad-hoc hotspot.

#### Step 2: Determine Laptop 1's LAN IP Address
On Laptop 1:
- **macOS / Linux**:
  ```bash
  ipconfig getifaddr en0   # or: hostname -I
  ```
- **Windows**:
  ```cmd
  ipconfig
  ```
  Look for the IPv4 Address (e.g. `192.168.1.105`).

#### Step 3: Configure Firewall on Laptop 1
Ensure incoming TCP connections on port `8000` are permitted:
- **Windows**:
  ```cmd
  netsh advfirewall firewall add rule name="PathSnap API" dir=in action=allow protocol=TCP localport=8000
  ```
- **macOS**: System Settings &rarr; Network &rarr; Firewall &rarr; Allow incoming connections for Python.
- **Linux**:
  ```bash
  sudo ufw allow 8000/tcp
  ```

#### Step 4: Mount Evaluator Data & Start Backend on Laptop 1
Place the 10 evaluator trip folders into `data/raw/evaluator/` (e.g., `data/raw/evaluator/00000101/`).

```bash
# Activate virtual environment
source venv/bin/activate    # or venv\Scripts\activate on Windows

# Launch FastAPI with Uvicorn
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

Verify backend startup:
- Model loaded message: `Model startup loading: SUCCESS (LGBMClassifier)`
- Trips mounted message: `Evaluator directory mounted: found 10 valid evaluator trips.`

#### Step 5: Test Connectivity from Laptop 2
From Laptop 2's browser or terminal:
```bash
curl http://<laptop1-ip>:8000/health
```
A JSON response with `"status": "healthy"` confirms network reachability.

#### Step 6: Launch Streamlit UI on Laptop 2
On Laptop 2:
```bash
# Set Laptop 1 IP and run Streamlit
export PATHSNAP_API_URL="http://<laptop1-ip>:8000"
streamlit run ui/app.py
```
*(On Windows cmd: `set PATHSNAP_API_URL=http://<laptop1-ip>:8000 && streamlit run ui/app.py`)*

Alternatively, enter `http://<laptop1-ip>:8000` directly into the sidebar text input in the running dashboard.

#### Troubleshooting Connection Problems:
1. **Connection Refused**: Verify Uvicorn is running with `--host 0.0.0.0` (not `127.0.0.1`).
2. **Request Timed Out**: Check firewall on Laptop 1 and verify both laptops are on the same subnet/IP range.
3. **Empty Trip List**: Ensure evaluator data folders on Laptop 1 contain `{id}.track`, `{id}.nodes`, and `{id}.arcs`.

---

### 4. Verification & Testing

PathSnap includes automated test suites covering unit logic, integration pipelines, API endpoints, and ground-truth leakage prevention:

```bash
# Run full pytest suite
python -m pytest -v
```

Tests executed:
- `test_api_health.py`: `/health` response verification, degraded state handling, model introspection.
- `test_api_trips.py`: `/trips` directory scanning, file validation, empty directory handling.
- `test_api_predict.py`: `/predict/{id}` path traversal protection, 404 handling, real P1/P2/P3 pipeline execution with LightGBM, and no-ground-truth evaluation handling.
- `test_security_leakage.py`: Strict assertion that reference routes, true arcs, and ground-truth labels are never exposed in API payloads.
- `test_services.py`: Individual verification of `ModelService`, `TripService`, `EvaluationService`, and `MapService`.
- `test_api_client.py`: Streamlit `PathSnapApiClient` HTTP handling and error parsing.

---

### 5. Genuine Screenshot Guide for Evaluation Report

To capture genuine screenshots for the technical report:
1. **Streamlit UI Dashboard**:
   - Start backend and UI.
   - Select an evaluator trip.
   - Click **Run Route Prediction**.
   - Capture the full browser view showing the interactive Folium map with GPS dots, predicted route polyline, start/end markers, and the KPI metric cards (Continuity, Inference Latency, Top-1 Accuracy, Macro F1).
2. **FastAPI OpenAPI / Swagger Response**:
   - Open `http://localhost:8000/docs` in the browser.
   - Execute `/health` and `/predict/{trip_id}`.
   - Capture the JSON response showing status, route coordinates, and evaluation summary.
3. **Backend Request Log Output**:
   - Capture the Laptop 1 terminal console showing Uvicorn logs with client IP, trip ID, and execution latency (e.g. `[PREDICTION REQUEST] IP=192.168.1.102 | TripID=00000001 | Status=200 | Latency=145.20ms`).

---

### 6. P4 Section for Technical Report

> **API, UI & Full-System Integration (P4 Summary)**:
> The PathSnap serving architecture decouples model inference from visualization across a two-laptop deployment. Laptop 1 hosts a high-performance FastAPI service coordinating the complete P1/P2/P3 pipeline: candidate generation via densified KD-trees ($R=50\text{ m}, K=10$), feature construction (19 kinematic and graph features), probability inference via a pre-loaded LightGBM classifier ($M=500, \text{lr}=0.05$), and route decoding via a Viterbi HMM accounting for topological arc connectivity and GPS time gaps ($\Delta t > 3\text{ s}$).
>
> Ground-truth reference routes are never provided to the inference pipeline (`with_route=False`). Post-inference evaluation independently computes observation-level Top-1 accuracy, Macro F1, and candidate miss rate on the server, returning only aggregated metrics. Laptop 2 runs a Streamlit dashboard that visualizes raw GPS tracks alongside reconstructed route polylines on interactive Folium cartography. The decoupled architecture provides sub-second prediction latency per trip while guaranteeing zero ground-truth data leakage.

---

### 7. Known Limitations & Dependencies

- **Evaluator Raw Dataset**: The 10 evaluator trips are held by the organizers and must be mounted into `data/raw/evaluator/` on Laptop 1 during the competition session.
- **Optuna Tuned Artifact**: While baseline LightGBM is trained and saved at `models/lightgbm_baseline.joblib`, additional Optuna hyperparameter trials can be dumped to `models/` and swapped seamlessly via `PATHSNAP_MODEL_PATH`.

## Metrics

Observation-level Top-1 Accuracy, Macro F1, Log Loss, Candidate Miss Rate, Route Continuity, Inference Time. Classifier-only and classifier + HMM results are both reported.

## Rules followed

- The evaluator-test data is never used for training, tuning or feature selection.
- No hardcoded predictions; every output goes through the full pipeline.
- Ground-truth labels never reach the model input or the UI.
- AI assistance is disclosed in the report as required by the event policy.

## Results

_To be filled in with actual numbers from `results/`._
