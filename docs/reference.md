# GroundingDINO Project Reference

## Purpose
- Provide an end-to-end reference for the customized GroundingDINO deployment in `/home/kim/GroundingDINO`.
- Summarize how detection services, API layers, frontend, and tooling integrate so the material can be reused in academic documentation (e.g., undergraduate thesis).

## Repository Highlights
- `README.md`: Upstream documentation from IDEA-Research; keep for original model context and citations.
- `src/`: Application-specific Python packages (adapters, services, utilities).
- `api/`: FastAPI app, routers, request/response schemas, dependency wiring.
- `config/runtime.py`: Centralized runtime settings resolved from environment variables.
- `front/`: Streamlit UI variants plus Dockerfile and requirements.
- `inference/`: Backend Docker image assets, constraints, and ASGI entrypoint.
- `cli/`: Command-line utilities (currently detection runner).
- `tests/`: Pytest-based smoke checks (weights/gpu availability gate).
- `weights/`: Local model checkpoints (symlinked into containers).
- `docker-compose.yml`: Multi-service orchestration (FastAPI backend + Streamlit frontend).

## System Architecture
- **Model Adapters (`src/adapters/`)**  
  - `grounding_dino.py`: Wraps `groundingdino.util.inference` helpers; manages device selection, model loading, prediction, and annotation.
  - `omdet_turbo.py`: Hugging Face Transformers adapter for OmDet Turbo; handles prompt parsing, post-processing, and custom annotation drawing via OpenCV.
- **Domain Service Layer (`src/services/`)**  
  - `detection_service.py`: Core orchestration for detections (load image, run adapter, optionally annotate, persist results). Supports file uploads, directory batch search, and temporary storage management.
  - `manager.py`: Registry that resolves named detection services and alias mapping.
  - `factory.py`: Builds `DetectionService` instances using `RuntimeSettings`; optionally registers OmDet Turbo if configured.
- **Configuration (`config/runtime.py`)**  
  - Resolves paths for configs, weights, images, results, and search directory.
  - Binds default thresholds, device selection (prefers CUDA when available), annotation toggle, and OmDet parameters via environment variables (`GDINO_*`, `OMDET_*`, `DETECTION_DEFAULT_MODEL`).
- **API Layer (`api/`)**  
  - `api/app.py`: Creates FastAPI application, registers dependencies, mounts routers.
  - `routers/detect.py`: Exposes `POST /detect`, `POST /search`, and `GET /healthz`.
  - `schemas/detections.py`: Pydantic models for request/response payloads, including annotated image encoding.
  - `dependencies.py`: Injects a shared `DetectionServiceManager` into request scope (lazy initialization).
- **Client-Facing Interfaces**  
  - **Streamlit (`front/streamlit_app.py`)**: Korean-language UI for gallery search via backend API, with adjustable thresholds and model selection (GroundingDINO vs. OmDet Turbo experimental). Performs health checks and displays annotated imagery returned by the API.
  - **Streamlit Direct Prototype (`front/streamlit_direct.py`)**: Illustrates how to host model inference directly in Streamlit (currently uses placeholder logic).
  - **CLI (`cli/detect.py`)**: Runs detections from the command line, outputting JSON per image; useful for batch processing or scripting.
  - **Webcam Utility (`webcam.py`)**: Configures V4L2 devices for live capture sanity checks.

## Data Flow Summary
1. **Input Acquisition**: Images arrive via HTTP upload, filesystem search, or CLI-specified paths.
2. **Service Resolution**: `DetectionServiceManager` resolves the requested model (`grounding_dino`, alias `gdino`, or optionally `omdet_turbo`).
3. **Preprocessing**: Adapter loads the image into numpy/torch formats, applying necessary transforms (`groundingdino.util.transforms`, manual conversions, or `AutoProcessor`).
4. **Inference**: Adapter submits caption/text prompts to the model, generating bounding boxes, logits, and textual phrases.
5. **Post-processing**: `DetectionService` converts tensors to serializable detections; optional annotation draws boxes and scores onto the source image and writes them to `data/results`.
6. **Response**: API returns structured JSON (and base64 annotations when available); Streamlit/CLI consume this payload to render or persist outputs.

## Deployment & Execution
- **Dockerized Backend (`inference/Dockerfile`)**  
  - Based on `pytorch/pytorch:2.1.2-cuda12.1-cudnn8-devel`.  
  - Clones upstream repo for utilities, installs requirements from `inference/requirements.txt` under version constraints (`inference/constraints.txt`).  
  - Copies local API/service code plus data assets into `/opt/program`.  
  - Launches `uvicorn app:app --host 0.0.0.0 --port 8000`.  
  - Assumes GPU access (`gpus: all`) and an 8G shared memory allocation (set via compose).
- **Dockerized Frontend (`front/Dockerfile`)**  
  - Slim Python 3.10 base, installs Streamlit requirements, copies UI sources, and runs `streamlit run front/streamlit_app.py`.  
  - Defaults `BACKEND_URL` to `http://inference:8000` for intra-compose networking.
- **Top-Level Dockerfile**  
  - Provides an alternative build that pins CUDA 12.1 wheels, constrains numpy/transformers versions, installs GroundingDINO in editable mode, and verifies CUDA availability.
- **docker-compose**  
  - Spins up `inference` (FastAPI backend) and `frontend` (Streamlit).  
  - Mounts the repository into `/opt/program/GroundingDINO/` for live code reload/testing.  
  - Ensures GPU, ports (`8000`, `8501`), and environment variables are configured for joint operation.
- **Local Execution (without Docker)**  
  1. Ensure Python ≥3.10, GPU-capable PyTorch 2.1.2, and system packages (`ffmpeg`, `ninja`, OpenCV deps).  
  2. Install dependencies: `pip install -r inference/requirements.txt -c inference/constraints.txt` plus `pip install -e .` for `GroundingDINO`.  
  3. Place required weights in `weights/` (defaults: `weights/groundingdino_swint_ogc.pth`, optional `weights/omdet_turbo.pth`).  
  4. Export optional env vars (see below) to customize runtime behavior.  
  5. Launch API with `uvicorn inference.app:app --host 0.0.0.0 --port 8000`.  
  6. Run `streamlit run front/streamlit_app.py` (set `BACKEND_URL` if backend runs remote).

## Runtime Configuration
- **Core GroundingDINO Variables**
  - `GDINO_MODEL_CONFIG` (defaults to `groundingdino/config/GroundingDINO_SwinT_OGC.py`)
  - `GDINO_WEIGHTS_PATH` (defaults to `weights/groundingdino_swint_ogc.pth`)
  - `GDINO_DEVICE` (`cuda` if available, else `cpu`)
  - `GDINO_BOX_THRESHOLD`, `GDINO_TEXT_THRESHOLD`
  - `GDINO_IMAGES_DIR`, `GDINO_RESULTS_DIR`, `GDINO_SEARCH_DIR`
  - `GDINO_ANNOTATE_RESULTS` (`true`/`false`)
- **Detection Manager**
  - `DETECTION_DEFAULT_MODEL` (e.g., `grounding_dino`, `omdet_turbo`)
- **OmDet Turbo (optional)**
  - `OMDET_MODEL_ID` (default `omlab/omdet-turbo-swin-tiny-hf`)
  - `OMDET_WEIGHTS_PATH`
  - `OMDET_DEVICE`
  - `OMDET_CONFIDENCE_THRESHOLD`
  - `OMDET_CLASS_NAMES` (comma-separated list or path to newline-delimited file)

## Testing & Validation
- `tests/test_detection_service.py`: Confirms `create_detection_service` materializes when weights exist; skipped automatically if weights missing.
- `docker_test.py`: Smoke script executed in the top-level Docker image to validate torch/transformers/numpy versions and CUDA availability.
- Recommend adding functional tests for `/detect` and `/search` when expanding thesis work (e.g., using `fastapi.testclient` and sample fixtures).

## Data & Results Management
- Default gallery input directory: `data/gallery/` (sample images and demo outputs).
- Upload temp directory: `data/images/` (created automatically).
- Annotated results: `data/results/` (JPEG outputs appended with `_annotated`).
- Ensure these directories persist or mount to external storage when using containers to avoid data loss.

## Extensibility Notes
- Add new detection models by introducing another adapter implementing `ModelAdapterProtocol` and registering it via `factory.py`.
- Customize annotation styles by modifying either `groundingdino/util/inference.annotate` or adapter-specific `annotate` methods.
- Extend API by adding routers under `api/routers/` and including them in `api/app.py`.
- For large-scale batch jobs, consider expanding `DetectionService.detect_in_directory` to stream results or integrate message queues.

## References & Further Reading
- GroundingDINO Paper: *Grounding DINO: Marrying DINO with Grounded Pre-Training for Open-Set Object Detection* (arXiv:2303.05499).
- Grounded SAM and GroundingDINO 1.5 releases (see links in `README.md`).
- Hugging Face `OmDetTurboForObjectDetection` documentation for prompt schema and post-processing APIs.
- Streamlit documentation for advanced UI customization (`st.session_state`, caching, layout primitives).

## Suggested Thesis Angles
- Contrast between GroundingDINO (transformer-based open-set detection) and OmDet Turbo (prompt-adapted detection) in mixed media galleries.
- Evaluation of text prompt engineering strategies (punctuation-driven label splitting implemented in `OmDetTurboModelAdapter._prepare_text_labels`).
- System benchmarking: throughput and latency under GPU/CPU fallbacks (`GroundingDinoModelAdapter.resolve_device`).
- Human-in-the-loop annotation workflows enabled by base64-encoded results returned via API and visualized in Streamlit.

