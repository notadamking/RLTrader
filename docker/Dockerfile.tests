FROM python:3.6.8-jessie

ADD ./requirements.base.txt /code/
ADD ./requirements.no-gpu.txt /code/
ADD ./requirements.tests.txt /code/requirements.txt

WORKDIR /code

RUN apt-get update \
    && apt-get install -y build-essential mpich libpq-dev \
    && pip install --progress-bar off --requirement requirements.txt