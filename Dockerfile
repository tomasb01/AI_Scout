FROM python:3.12-slim

# Install git (required for GitPython clone operations)
RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy project files
COPY pyproject.toml .
COPY aiscout/ aiscout/

# Install dependencies
RUN pip install --no-cache-dir .

# Create directories
RUN mkdir -p /app/reports /app/config

# Expose web UI port
EXPOSE 8080

# Default: start Web UI
CMD ["aiscout", "web", "--port", "8080"]
