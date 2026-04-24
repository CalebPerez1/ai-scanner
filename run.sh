#!/bin/bash
cd "$(dirname "$0")"
source venv/bin/activate
PYTHONPATH=. uvicorn backend.main:app --reload &
cd frontend && npm start &
wait
