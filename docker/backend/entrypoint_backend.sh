#!/bin/sh

uvicorn handwriting_recognition_service.asgi:application --host 0.0.0.0 --port 8000 --workers 4
