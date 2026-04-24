# AI-Scan ( AI Supply Chain Security Scanner )

A Security Scanner which scans AI/ML project pipelines for supply chain vulnerabilities i.e., dependency CVEs, typosquatting, unsafe model loading, prompt injection, leaked credentials, and misconfigurations. Point it at a GitHub repo or local project and it runs four scanners at once.

**Live demo:** [ai-scanner-z9yr.onrender.com](https://ai-scanner-z9yr.onrender.com)

## What it does

Four scanners run at the same time against your project:

- **Dependency auditor** — checks packages against the OSV vulnerability database for known CVEs. Also catches typosquatting (packages with names suspiciously close to popular ones).
- **Model scanner** — flags unsafe deserialization calls like `torch.load()` and `pickle.load()` that can execute hidden code when loading model files. Also checks HuggingFace models for trust signals.
- **Prompt injection tester** — sends 36 attack payloads to a live LLM endpoint across 6 categories (jailbreaks, system prompt extraction, data exfiltration, etc.) and reports which ones succeed.
- **Config analyzer** — finds hardcoded API keys, leaked secrets in Jupyter notebook outputs, DEBUG mode left on, permissive CORS, and unprotected inference endpoints.

## Setup

You need Python 3.10+, Node.js 18+, and Git.

```bash
git clone https://github.com/CalebPerez1/ai-scanner.git
cd ai-scanner

# backend
python3 -m venv venv
source venv/bin/activate
pip install -r backend/requirements.txt

# frontend
cd frontend && npm install && cd ..
```

## Usage

Run both backend and frontend:

```bash
./run.sh
```

Or separately:

```bash
# terminal 1
source venv/bin/activate
PYTHONPATH=. uvicorn backend.main:app --reload

# terminal 2
cd frontend && npm start
```

Dashboard opens at `http://localhost:3000`.

### CLI

```bash
PYTHONPATH=. python -m backend.cli ./my-project
PYTHONPATH=. python -m backend.cli https://github.com/org/repo
PYTHONPATH=. python -m backend.cli ./my-project --llm-endpoint http://localhost:8080/v1/chat/completions
PYTHONPATH=. python -m backend.cli ./my-project --output json --output-file report.json
```

### Docker

```bash
docker compose up --build
```

Runs everything on `http://localhost:8000`.

## Tests

```bash
source venv/bin/activate
PYTHONPATH=. pytest -v
```

198 tests across all modules.

## Built with

Python, FastAPI, React, asyncio, Pydantic, OSV.dev API, HuggingFace Hub API
