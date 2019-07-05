FROM python:3.6.8-jessie

ADD ./requirements.base.txt /code/
ADD ./requirements.no-gpu.txt /code/requirements.txt

WORKDIR /code

RUN apt-get update \
    && apt-get install -y build-essential mpich libpq-dev

# should merge to top RUN to avoid extra layers - for debug only :/
RUN pip install -r requirements.txt