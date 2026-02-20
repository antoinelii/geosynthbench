# 🛰️ GeoSynthBench

Synthetic Geospatial Dataset for Structured Reasoning Evaluation

<p align="left"> <img src="https://img.shields.io/badge/python-3.12-blue.svg" /> <img src="https://img.shields.io/badge/uv-managed-blueviolet.svg" /> <img src="https://img.shields.io/badge/lint-ruff-informational.svg" /> <img src="https://img.shields.io/badge/tests-pytest-lightgrey.svg" /> <img src="https://img.shields.io/badge/license-MIT-black.svg" /> </p>

## Overview

GeoSynthBench is a controllable synthetic data generator for evaluating structured geospatial reasoning in multimodal models.
Rather than testing only perception, it probes:

- Spatial & topological reasoning
- Temporal change understanding
- Policy & rule-based inference
- Counterfactual reasoning
- Uncertainty handling

Each example includes rendered world state (t₀, optional t₁), a reasoning question, and deterministic ground-truth answers.

## Task Ladder

- Tasks scale in reasoning dep
- Perception — object detection & counting
- Topology — connectivity, intersections
- Temporal — change & disruption analysis
- Policy — zoning or rule violations
- Counterfactual / Uncertainty — hypothetical reasoning

The world state is generated symbolically first, then rendered — ensuring traceable reasoning and perfect labels.

## Overview

GeoSynthBench is a modular synthetic benchmark designed to evaluate **structured spatial reasoning** in satellite and aerial imagery.

Instead of focusing on perception-only tasks (captioning, detection, segmentation), this project targets:

- Elevation reasoning
- Terrain slope reasoning
- Hydrological proximity
- Road-network topology
- Graph-based isolation

All tasks are grounded in a deterministic world model and provide exact geometric or network-based oracle answers.

---

## Why This Project?

### Limitations of Current VLM Datasets

Most existing vision-language benchmarks:

- Emphasize object detection or captioning
- Lack structured geospatial reasoning
- Do not provide deterministic geometric ground truth
- Avoid graph/topology-based reasoning
- Rarely support controlled synthetic variation

GeoSynthBench addresses these gaps.

---

## Core Design

### 1. Structured World Model

Each sample is generated from a `WorldState` containing:

- Terrain (height field)
- Water bodies (polygons)
- Vegetation (polygons)
- Settlements
- Roads (graph structure)
- Buildings

All reasoning is computed from geometry — never from image pixels.

---

### 2. Synthetic Scene Generation

For each sample:

1. Generate `t0` world
2. Render RGB + semantic mask
3. Compute deterministic oracle
4. Generate natural-language reasoning prompt
5. Save everything in JSONL format

## Implemented Tasks

### E1 — Elevation Comparison

Which settlement is at higher elevation?

Oracle:

- Mean elevation from height field

---

### D1 — Distance to Water

Which settlement is closer to water?

Oracle:

- Minimum geometric distance to water polygon

---

### S1 — Slope Comparison

Which settlement is on steeper terrain?

Oracle:

- Terrain slope sampled at settlement center

---

### N1 — Network Isolation

Which settlement is most isolated in the road network?

Isolation metric:

- Mean shortest-path distance to all other settlements
- Edge weights = curved road polyline length

---

## Dataset Format

One JSON object per line:

'''json
{
"sample_id": "00012",
"task_code": "N1",
"modality": "single",
"inputs": {
"image": "data/N1/00012/rgb.png",
"mask": "data/N1/00012/mask.png"
},
"prompt": "...",
"answer": "C",
"oracle": {
"isolation_scores_m_by_label": { ... }
}
}
'''

## Repository Structure

'''
src/geosynthbench/
world/ # World model + terrain
gen/ # Procedural generation
render/ # RGB + semantic rendering
tasks/ # E1, D1, S1, N1
pipeline/ # Dataset writer
scripts/
generate\_\*.py
view_dataset.py
data/
E1/
D1/
S1/
N1/
'''

## Visualization

Run:

'''bash
uv run streamlit run scripts/view_dataset.py
'''

Features:

Dataset selector

Task filtering

Prompt + answer display

Oracle inspection

RGB + semantic mask visualization

Dataset summary statistics

## Design Philosophy

Deterministic oracle

Minimal over-engineering

Modular task pipeline

Geometry-first reasoning

Reproducible synthetic generation

## Future Work

Temporal change reasoning

Multi-hop spatial reasoning

Difficulty scaling

VLM fine-tuning experiments

Real-to-synthetic transfer
