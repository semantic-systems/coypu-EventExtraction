FROM python:3.8-slim

RUN \
    apt-get update && \
    apt-get -y upgrade && \
    apt-get -y dist-upgrade && \
    apt-get -y autoremove && \
    apt-get -y install \
        git \
        curl \
    && \
    apt-get -y clean

COPY requirements.txt /src/
RUN pip install -r /src/requirements.txt
RUN mkdir -p /data
COPY . /src/
WORKDIR src

EXPOSE 5000/tcp

ENTRYPOINT ["python"]
CMD ["/src/script.py"]
