FROM ubuntu:22.04

WORKDIR /usr/src/app

COPY backend/ .

#ADD docker/backend/entrypoint_backend.sh .
#ADD docker/backend/entrypoint_celery.sh .

RUN chmod +x entrypoint_backend.sh
RUN chmod +x entrypoint_celery.sh
RUN chmod +x setup/config_db.sh

# installing dependencies for opencv
RUN apt-get update
RUN apt install libgl1-mesa-glx -y
RUN apt-get install ffmpeg libsm6 libxext6 -y

# installing dependencies for python-magic
RUN apt-get install libmagic1 -y

# installing dependencies for psycopg2
RUN apt-get install libpq-dev python3-dev -y

RUN apt-get install poppler-utils -y

# installing python3.10
RUN apt-get -y install python3.10 python3-pip

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# installing python dependencies
RUN python3 -m pip install --upgrade pip
RUN apt update -y
RUN apt install git -y
RUN pip install ./sage[errant]
RUN pip install -r requirements.txt
RUN pip install python-magic

#RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121


# setup django project
RUN python3 manage.py collectstatic

RUN python3 manage.py makemigrations

RUN apt-get install -y fontconfig 