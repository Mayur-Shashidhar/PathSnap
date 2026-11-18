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

| Endpoint | Method | Returns |
|---|---|---|
| `/trips` | GET | list of available evaluation trip IDs |
| `/predict/{trip_id}` | GET | predicted arc per GPS point, route geometry, accuracy metrics |

Every request is logged to the console on Laptop 1. Ground-truth routes stay on the server and are only used to compute metrics after prediction.

## Metrics

Observation-level Top-1 Accuracy, Macro F1, Log Loss, Candidate Miss Rate, Route Continuity, Inference Time. Classifier-only and classifier + HMM results are both reported.

## Rules followed

- The evaluator-test data is never used for training, tuning or feature selection.
- No hardcoded predictions; every output goes through the full pipeline.
- Ground-truth labels never reach the model input or the UI.
- AI assistance is disclosed in the report as required by the event policy.

## Results

_To be filled in with actual numbers from `results/`._
