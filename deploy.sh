#!/bin/bash
# Deployment script for DigitalOcean Droplet

set -e

echo "🚀 Deploying Food Truck Route Optimizer..."

# Update system
sudo apt-get update
sudo apt-get upgrade -y

# Install dependencies
sudo apt-get install -y \
    python3 \
    python3-pip \
    git \
    nginx \
    supervisor

# Clone repo (or update if exists)
if [ ! -d "/home/ubuntu/FoodTruckRouteOptimizer" ]; then
    cd /home/ubuntu
    git clone https://github.com/LoganMazurek/FoodTruckRouteOptimizer.git
else
    cd /home/ubuntu/FoodTruckRouteOptimizer
    git pull origin main
fi

cd /home/ubuntu/FoodTruckRouteOptimizer

# Create data directory if it doesn't exist
mkdir -p data

# Download OSRM data if not present (this may take 5-15 minutes)
if [ ! -f "data/illinois-260201.osrm" ]; then
    echo "📥 Downloading OSRM Illinois data (this may take a few minutes)..."
    
    # Install required tools
    sudo apt-get install -y wget
    
    # Download the OSM data
    wget -O data/illinois-260201.osm.pbf https://download.geofabrik.de/north-america/us/illinois-latest.osm.pbf
    
    # Build OSRM data
    echo "⚙️  Building OSRM database (this may take 5+ minutes)..."
    sudo apt-get install -y osrm-backend
    
    cd data
    osrm-extract -p /usr/share/osrm/profiles/car.lua illinois-260201.osm.pbf
    osrm-contract illinois-260201.osrm
    cd ..
    
    echo "✅ OSRM data ready!"
else
    echo "✅ OSRM data already present"
fi

# Install Python dependencies
pip3 install -r requirements.txt

# Create supervisor config for OSRM
sudo tee /etc/supervisor/conf.d/osrm.conf > /dev/null <<EOF
[program:osrm]
command=/usr/bin/osrm-routed --algorithm=MLD /home/ubuntu/FoodTruckRouteOptimizer/data/illinois-260201.osrm
autostart=true
autorestart=true
user=ubuntu
stdout_logfile=/var/log/osrm.log
stderr_logfile=/var/log/osrm_error.log
EOF

# Create supervisor config for Flask
sudo tee /etc/supervisor/conf.d/flask.conf > /dev/null <<EOF
[program:flask]
command=/usr/bin/python3 -m waitress --host 127.0.0.1 --port 5001 app:app
directory=/home/ubuntu/FoodTruckRouteOptimizer
autostart=true
autorestart=true
user=ubuntu
stdout_logfile=/var/log/flask.log
stderr_logfile=/var/log/flask_error.log
EOF

# Configure Nginx as reverse proxy
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

# Restart services
sudo systemctl restart nginx
sudo supervisorctl reread
sudo supervisorctl update
sudo supervisorctl start osrm flask

echo "✅ Deployment complete!"
echo "Visit your server's IP address to access the app"
echo "View logs with: sudo tail -f /var/log/flask.log"
