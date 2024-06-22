#!/bin/sh

celery -A handwriting_recognition_service worker --loglevel=info --concurrency 2 -E -P threads
