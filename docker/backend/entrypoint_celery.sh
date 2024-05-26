#!/bin/sh

celery -A handwriting_recognition_service worker --loglevel=info --concurrency 1 -E -P threads
