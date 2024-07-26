# syntax=docker/dockerfile:1.7-labs
FROM ubuntu:22.04 AS dependencies

WORKDIR /usr/src/app

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

RUN apt-get update && \
    apt-get install -y libgl1-mesa-glx \
                       ffmpeg libsm6 libxext6 \
                       python3.10 python3-pip \
                       libpq-dev python3-dev \
                       poppler-utils \
                       libmagic1 \
                       git \
                       fontconfig


FROM dependencies AS python-dependencies

RUN python3 -m pip install --upgrade pip
RUN pip install python-magic

ADD backend/sage/ sage/
RUN pip install --no-cache-dir ./sage[errant]

ADD backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt


FROM python-dependencies AS final

ADD --exclude=backend/requirements.txt \
    --exclude=backend/sage/ \
    backend/ .

RUN python3 manage.py collectstatic
RUN python3 manage.py makemigrations 