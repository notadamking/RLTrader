FROM postgres:11-alpine

ARG ID=1000
ARG GI=1000

ENV POSTGRES_USER=rl_trader
ENV POSTGRES_PASSWORD=rl_trader
ENV POSTGRES_DB='rl_trader'
ENV PGDATA=/var/lib/postgresql/data/trader-data

RUN adduser -D -u $ID rl_trader