#!/bin/bash
gunicorn legal_agent:app -w 4 -k uvicorn.workers.UvicornWorker -b 0.0.0.0:$PORT