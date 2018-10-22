FROM python:3

RUN mkdir -p /worker
WORKDIR /worker

COPY ./install.sh /worker/install.sh
RUN bash install.sh
COPY . /worker

EXPOSE 50051

#CMD ["python", "server.py"]
