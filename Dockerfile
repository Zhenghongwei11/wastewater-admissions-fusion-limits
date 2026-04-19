FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
      git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /repo

COPY requirements.txt /repo/requirements.txt
RUN python -m pip install --upgrade pip && python -m pip install -r /repo/requirements.txt

COPY . /repo

ENTRYPOINT ["/bin/bash"]

