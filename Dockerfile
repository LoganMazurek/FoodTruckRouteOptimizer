# Food Truck Route Optimizer container image
# Serves the Flask app via Waitress (same server used in the supervisor setup).
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    FLASK_ENV=production

WORKDIR /app

# Install Python dependencies first to leverage Docker layer caching.
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY . /app

# Per-session graph/route state and the OSM cache live under these paths.
# They are bind-mounted in docker-compose so data survives image rebuilds.
RUN mkdir -p /app/temp /app/data/cache

EXPOSE 5001

# Mirrors the production command from deploy.sh (waitress on port 5001).
CMD ["python", "-m", "waitress", "--host", "0.0.0.0", "--port", "5001", "app:app"]
