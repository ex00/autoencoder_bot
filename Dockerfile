FROM python:3.7
LABEL maintainer="https://github.com/ex00/"

COPY requirements.txt /

RUN pip install -r /requirements.txt

COPY ./bot/src /bot
COPY ./autoencoder/saved /models
COPY ./autoencoder/src /autoencoder/src
ENV PYTHONPATH=${PYTHONPATH}:/autoencoder
