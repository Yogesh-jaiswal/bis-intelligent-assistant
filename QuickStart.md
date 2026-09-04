# BIS Intelligent Assistant — Quick Start Guide

> [!IMPORTANT]
> **Smart India Hackathon (SIH 2026) Prototype Disclaimer**
> 
> This software is an experimental proof-of-concept / MVP developed under intense hackathon time constraints. It is intended for demonstration, functional evaluation, and feasibility assessment of deterministic RAG query pipelines over official Indian Standards.
>
> * **Not Production Hardened**: This API is **not** engineered for enterprise production security, multi-tenant isolation, or hardened authentication.
> * **Latency Considerations**: Response latency depends on the LLM backend (typically 15–45s per synthesized response when running on remote tunnels or consumer hardware). It is not optimized for real-time sub-second SLAs.

---

## 1. Prerequisites

* **Python**: 3.11 or 3.12
* **Docker & Docker Compose**: Installed and running
* **Git**: Installed
* **AI Model Endpoint**: Either local Ollama (`qwen3:8b`) with GPU acceleration OR hosted tunnel instance (via Kaggle or Google Colab)

---

## 2. Environment Configuration

### Step A: Clone the Repository
```bash
git clone https://github.com/Yogesh-jaiswal/bis-intelligent-assistant
cd bis-intelligent-assistant
```

### Step B: Create and Activate Python Virtual Environment
Always activate your virtual environment before installing packages or running scripts:

* **Create the virtual environment**:
  ```bash
  python -m venv .venv
  ```

* **Activate the virtual environment**:
  * **Windows (PowerShell)**:
    ```powershell
    .venv\Scripts\Activate.ps1
    ```
  * **Windows (Command Prompt / cmd.exe)**:
    ```cmd
    .venv\Scripts\activate.bat
    ```
  * **Linux / macOS (bash / zsh)**:
    ```bash
    source .venv/bin/activate
    ```

* **Verify active environment**:
  ```bash
  # Should point to python inside .venv
  python -c "import sys; print(sys.executable)"
  ```

### Step C: Install Python Dependencies
With `.venv` activated:
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Step D: Configure Environment Variables
Copy the template `.env.example` to create your local `.env`:
```bash
# Windows (PowerShell):
Copy-Item .env.example .env

# Linux / macOS:
cp .env.example .env
```

---

## 3. Provision the AI Model (Local GPU vs. Colab / Kaggle)

The assistant relies on Ollama running the instruction-tuned model `qwen3:8b`.

### Scenario A: Running Locally on Your PC (Requires Dedicated GPU)
If your workstation has a dedicated NVIDIA GPU (e.g. RTX 3060/4060 or higher with $\ge 8\text{ GB}$ VRAM):
1. Install and start [Ollama](https://ollama.ai):
   ```bash
   ollama serve
   ```
2. Pull the target model:
   ```bash
   ollama pull qwen3:8b
   ```
3. In your `.env`, set `MODEL_URL` to your localhost instance:
   ```ini
   # For local Python runs (python run.py):
   MODEL_URL=http://localhost:11434
   MODEL_NAME=qwen3:8b
   AI_PROVIDER=OLLAMA

   # If running the app inside Docker, use:
   # MODEL_URL=http://host.docker.internal:11434
   ```

### Scenario B: No Dedicated Local GPU? Use Google Colab or Kaggle (Recommended)
> [!WARNING]
> Running an 8-billion parameter LLM on CPU alone is **extremely slow** (often taking minutes per response) and may freeze your machine. If you do not have a dedicated GPU, use the provided free GPU notebooks:

1. Import one of the provided notebooks:
   * **Google Colab**: [`./Collab-BIS-Model.ipynb`](./Collab-BIS-Model.ipynb)
   * **Kaggle**: [`./Kaggle-BIS-Model.ipynb`](./Kaggle-BIS-Model.ipynb)
2. Select a GPU runtime (T4 on Colab or P100/T4 on Kaggle).
3. Run all cells in the notebook. The script will start Ollama and launch a Cloudflare tunnel.
4. Copy the public tunnel URL printed at the bottom (e.g. `https://<random-subdomain>.trycloudflare.com`).
5. Update `MODEL_URL` in your `.env`:
   ```ini
   MODEL_URL=https://<your-tunnel-subdomain>.trycloudflare.com
   MODEL_NAME=qwen3:8b
   AI_PROVIDER=OLLAMA
   ```

---

## 4. Initialize the Database

Before starting the application, the database schema must be migrated and the BIS dataset must be seeded.

### Step A: Start PostgreSQL

Start only the PostgreSQL service:

```bash
docker compose up -d db
```

Verify that the database container is running:

```bash
docker compose ps
```

### Step B: Build the Application Image

Build the application image before running database commands inside the container:

```bash
docker compose -f docker-compose.yml -f docker-compose.app.yml build
```

### Step C: Run Database Migrations

Run the Flask-Migrate migration command inside the application container:

```bash
docker compose -f docker-compose.yml -f docker-compose.app.yml run --rm app flask db upgrade
```

### Step D: Seed the BIS Dataset

Seed the catalog and standards data using the application container:

```bash
docker compose -f docker-compose.yml -f docker-compose.app.yml run --rm app flask seed-dataset
```

> [!IMPORTANT]
> Migrations and dataset seeding must be completed successfully before using the application or running the full E2E system tests.

---

## 5. Start the Application

You can start the application either fully containerized via Docker or locally as a development server.

### Method A: Docker Compose (Full Stack)
To build and launch both the database and the containerized application image at once:

```bash
docker compose -f docker-compose.yml -f docker-compose.app.yml up -d --build
```

* The app will build [`./Dockerfile.app`](./Dockerfile.app) and create the image `bis-assistant-app:latest`.
* Tesseract OCR is included by default. To skip OCR installation during build, use:
  ```bash
  docker compose -f docker-compose.yml -f docker-compose.app.yml build --build-arg INSTALL_OCR=false
  docker compose -f docker-compose.yml -f docker-compose.app.yml up -d
  ```
* Verify health:
  ```bash
  curl http://127.0.0.1:5000/v1/health
  ```

To stop all containers:
```bash
docker compose -f docker-compose.yml -f docker-compose.app.yml down
```

---

### Method B: Local Development Server
With the database running (`docker compose up -d`), launch the Flask development server:

```bash
python run.py
```
The API server listens on `http://127.0.0.1:5000`.

---

## 6. Testing the API

### Testing with Insomnia or Postman
Pre-configured API collections are provided in [`./api_collections/`](./api_collections/):
* **Insomnia**: Import [`./api_collections/Insomnia API Collections`](./api_collections/Insomnia%20API%20Collections)
* **Postman / Thunder Client**: Import [`./api_collections/Generic API Collections`](./api_collections/Generic%20API%20Collections)

Follow the step-by-step import instructions in the [**API Collections Guide**](./api_collections/README.md).

### Automated Feature Tests (Fast & Deterministic)
Tests execute against fast mock repositories and deterministic providers—**the app container is not required, only PostgreSQL**:
```bash
pytest tests/features/ -v
```
Detailed testing documentation is available in [`./tests.md`](./tests.md).

### End-to-End System Tests
To validate the full pipeline (Ollama + PostgreSQL + API server) against live multi-turn conversations:
```bash
python tests/e2e/run_e2e.py
```
For more information on E2E scenarios and limits, see [`./tests/e2e/README.md`](./tests/e2e/README.md).

---

## 7. Example API Query

Send a query to the Assistant:

```bash
curl -X POST http://127.0.0.1:5000/v1/query \
  -H "Content-Type: application/json" \
  -d '{
    "message": {
      "content": "Which BIS standard applies to PVC cables?",
      "language": "en"
    }
  }'
```

Expected JSON response envelope:
```json
{
  "success": true,
  "data": {
    "conversation_id": "conv_01K...",
    "message_type": "answer",
    "message": "Polyvinyl chloride insulated cables are covered under IS 694:2010...",
    "citations": [
      {
        "citation_id": 1,
        "source_title": "IS 694:2010",
        "source_url": "https://standards.bis.gov.in/is694",
        "page_number": 1
      }
    ],
    "data": [
      {
        "card_type": "standard_details",
        "title": "IS 694:2010",
        "fields": {
          "standard_number": "IS 694:2010",
          "title": "PVC Insulated Cables",
          "status": "Active"
        }
      }
    ]
  },
  "error": null
}
```
