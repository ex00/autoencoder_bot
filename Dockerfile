FROM python:3.7-alpine
LABEL maintainer="https://github.com/ex00/"

COPY requirements.txt /

RUN pip install -r /requirements.txt

COPY ./bot/src /bot
