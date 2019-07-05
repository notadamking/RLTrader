FROM python:3.6.8-jessie

ADD ./requirements.base.txt /code/
ADD ./requirements.txt /code/

WORKDIR /code

RUN apt-get update \
    && apt-get install -y build-essential mpich libpq-dev \
    && pip install -r requirements.txt