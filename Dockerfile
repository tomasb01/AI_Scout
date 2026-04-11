FROM python:3.12-slim

WORKDIR /app

COPY pyproject.toml .
COPY aiscout/ aiscout/

RUN pip install --no-cache-dir .

RUN mkdir -p /app/reports /app/config

ENTRYPOINT ["aiscout"]
CMD ["scan", "--help"]
