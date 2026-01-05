#!/bin/bash
cd server
export PYTHONIOENCODING=utf-8
./venv/Scripts/python.exe -m uvicorn main:app --reload --port 5000 --host 0.0.0.0
