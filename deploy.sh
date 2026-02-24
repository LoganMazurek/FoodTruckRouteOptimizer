#!/bin/bash
# Deployment script for DigitalOcean Droplet

set -e

echo "🚀 Deploying Food Truck Route Optimizer..."

DOMAIN="nroute.loganmazurek.com"
CERTBOT_EMAIL="${CERTBOT_EMAIL:-}"

# Update system
sudo apt-get update
sudo apt-get upgrade -y

# Install dependencies
sudo apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    git \
    nginx \
    supervisor \
    docker.io

# Add ubuntu user to docker group
sudo usermod -aG docker ubuntu || true

# Start and enable Docker
sudo systemctl start docker
sudo systemctl enable docker

# Clone repo (or update if exists)
if [ ! -d "/home/ubuntu/FoodTruckRouteOptimizer" ]; then
    cd /home/ubuntu
    git clone https://github.com/LoganMazurek/FoodTruckRouteOptimizer.git
else
    cd /home/ubuntu/FoodTruckRouteOptimizer
    # Stop Flask before updating to ensure clean state
    sudo supervisorctl stop flask || true
    # Stash any local changes and pull latest
    git stash
    git pull origin main
    # Clean Python cache
    find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
    find . -type f -name "*.pyc" -delete 2>/dev/null || true
fi

cd /home/ubuntu/FoodTruckRouteOptimizer

# Create necessary directories if they don't exist
mkdir -p data/illinois
mkdir -p deploy/osrm
mkdir -p temp

# Set proper permissions
chown -R ubuntu:ubuntu /home/ubuntu/FoodTruckRouteOptimizer
chmod -R 755 /home/ubuntu/FoodTruckRouteOptimizer/temp

OSRM_PROFILE_DIR="/home/ubuntu/FoodTruckRouteOptimizer/deploy/osrm"
OSRM_PROFILE_BASE="$OSRM_PROFILE_DIR/car.lua"
OSRM_PROFILE_CUSTOM="$OSRM_PROFILE_DIR/car.custom.lua"
OSRM_PBF_PATH="/home/ubuntu/FoodTruckRouteOptimizer/data/illinois/illinois-260201.osm.pbf"
OSRM_DATA_PATH="/home/ubuntu/FoodTruckRouteOptimizer/data/illinois/illinois-260201.osrm"

# Pull OSRM Docker image
echo "📦 Pulling OSRM Docker image..."
sudo docker pull osrm/osrm-backend

# Create a custom profile from the container's default if needed
if [ ! -f "$OSRM_PROFILE_CUSTOM" ] || [ "${OSRM_REBUILD}" = "1" ]; then
    echo "🧩 Building custom OSRM profile..."
    sudo docker run --rm \
        -v "$OSRM_PROFILE_DIR:/profiles" \
        osrm/osrm-backend \
        bash -lc "cat /opt/car.lua > /profiles/car.lua"

    python3 "$OSRM_PROFILE_DIR/patch_car_profile.py" \
        "$OSRM_PROFILE_BASE" \
        "$OSRM_PROFILE_CUSTOM"
fi

# Stop and remove old OSRM containers if they exist
sudo docker stop osrm-illinois osrm-wisconsin 2>/dev/null || true
sudo docker rm osrm-illinois osrm-wisconsin 2>/dev/null || true

# Build CH data if requested or missing
if [ -f "$OSRM_PBF_PATH" ] && ( [ "${OSRM_REBUILD}" = "1" ] || [ ! -f "$OSRM_DATA_PATH" ] ); then
    echo "🗺️  Building OSRM data (CH)..."
    sudo docker run --rm -t \
        -v /home/ubuntu/FoodTruckRouteOptimizer/data/illinois:/data \
        -v "$OSRM_PROFILE_DIR:/profiles" \
        osrm/osrm-backend \
        osrm-extract -p /profiles/car.custom.lua /data/illinois-260201.osm.pbf

    sudo docker run --rm -t \
        -v /home/ubuntu/FoodTruckRouteOptimizer/data/illinois:/data \
        osrm/osrm-backend \
        osrm-partition /data/illinois-260201.osrm

    sudo docker run --rm -t \
        -v /home/ubuntu/FoodTruckRouteOptimizer/data/illinois:/data \
        osrm/osrm-backend \
        osrm-customize /data/illinois-260201.osrm
fi

# Start Illinois OSRM container
if [ -f "data/illinois/illinois-260201.osrm" ]; then
    echo "🚀 Starting Illinois OSRM server on port 5000..."
    sudo docker run -d \
        --name osrm-illinois \
        --restart always \
        -p 5000:5000 \
        --memory=1g \
        -v /home/ubuntu/FoodTruckRouteOptimizer/data/illinois:/data \
        osrm/osrm-backend osrm-routed --algorithm mld /data/illinois-260201.osrm
else
    echo "⚠️  Illinois OSRM data not found - upload to data/illinois/"
    echo "    All routing will use public OSRM until Illinois data is uploaded"
fi

# Create Python virtual environment
if [ ! -f "venv/bin/pip" ]; then
    echo "📦 Creating Python virtual environment..."
    rm -rf venv
    python3 -m venv venv
fi

# Install Python dependencies in venv
echo "📦 Installing Python dependencies..."
venv/bin/pip install -r requirements.txt

# Create supervisor config for Flask
sudo tee /etc/supervisor/conf.d/flask.conf > /dev/null <<EOF
[program:flask]
command=/home/ubuntu/FoodTruckRouteOptimizer/venv/bin/python -m waitress --host 127.0.0.1 --port 5001 app:app
directory=/home/ubuntu/FoodTruckRouteOptimizer
autostart=true
autorestart=true
user=ubuntu
environment=SECRET_KEY="$(openssl rand -hex 32)",FLASK_ENV="production"
stdout_logfile=/var/log/flask.log
stderr_logfile=/var/log/flask_error.log
EOF

# Configure Nginx as reverse proxy
# Check if SSL certificate exists - if yes, preserve SSL config
if [ -f "/etc/letsencrypt/live/$DOMAIN/fullchain.pem" ]; then
    echo "🔐 SSL certificate found - configuring HTTPS..."
    sudo tee /etc/nginx/sites-available/default > /dev/null <<EOF
server {
    listen 80 default_server;
    listen [::]:80 default_server;
    server_name $DOMAIN;
    
    # Redirect HTTP to HTTPS
    return 301 https://\$server_name\$request_uri;
}

server {
    listen 443 ssl http2 default_server;
    listen [::]:443 ssl http2 default_server;
    server_name $DOMAIN;

    # SSL configuration
    ssl_certificate /etc/letsencrypt/live/$DOMAIN/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/$DOMAIN/privkey.pem;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_prefer_server_ciphers on;
    ssl_ciphers 'ECDHE-ECDSA-AES128-GCM-SHA256:ECDHE-RSA-AES128-GCM-SHA256:ECDHE-ECDSA-AES256-GCM-SHA384:ECDHE-RSA-AES256-GCM-SHA384';

    location / {
        proxy_pass http://127.0.0.1:5001;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
        proxy_set_header X-Forwarded-Host \$server_name;
    }
}
EOF
else
    echo "⚠️  No SSL certificate found - configuring HTTP only..."
    sudo tee /etc/nginx/sites-available/default > /dev/null <<EOF
server {
    listen 80 default_server;
    listen [::]:80 default_server;

    server_name _;

    location / {
        proxy_pass http://127.0.0.1:5001;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
    }
}
EOF
fi

# Attempt automated SSL setup if email is provided
if [ -n "$CERTBOT_EMAIL" ] && [ ! -f "/etc/letsencrypt/live/$DOMAIN/fullchain.pem" ]; then
    echo "🔐 Attempting automated SSL setup for $DOMAIN..."
    sudo apt-get install -y certbot python3-certbot-nginx
    sudo systemctl reload nginx || true

    if ! sudo certbot --nginx -d "$DOMAIN" \
        --non-interactive \
        --agree-tos \
        --email "$CERTBOT_EMAIL" \
        --redirect; then
        echo "⚠️  Certbot failed. Ensure DNS points to this server and try again."
    fi
fi

# Restart services
echo "🔄 Restarting services..."
sudo systemctl restart nginx
sudo supervisorctl reread
sudo supervisorctl update

if sudo supervisorctl status flask | grep -q RUNNING; then
    sudo supervisorctl restart flask
else
    sudo supervisorctl start flask
fi

# Wait a moment for services to start
sleep 3

echo "✅ Deployment complete!"
echo "Visit your server's IP address to access the app"
echo ""
echo "Health check available at: http://localhost:5001/health"
echo ""
echo "OSRM containers running:"
sudo docker ps --filter name=osrm
echo ""
echo "Routing:"
echo "  Illinois: Local OSRM (port 5000)"
echo "  Other states: Public OSRM"
echo ""
echo "View logs:"
echo "  Flask: sudo tail -f /var/log/flask.log"
echo "  Flask errors: sudo tail -f /var/log/flask_error.log"
echo "  Illinois OSRM: sudo docker logs osrm-illinois"
echo ""
echo "Check deployment health:"
echo "  bash /home/ubuntu/FoodTruckRouteOptimizer/check_deployment.sh"
