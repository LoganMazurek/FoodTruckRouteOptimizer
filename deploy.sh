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
    git pull origin main
fi

cd /home/ubuntu/FoodTruckRouteOptimizer

# Create data directories if they don't exist
mkdir -p data/illinois
mkdir -p data/wisconsin

# Check for OSRM data (should be uploaded separately from local build)
if [ -f "data/illinois/illinois-260201.osrm" ]; then
    echo "✅ Illinois OSRM data present"
else
    echo "⚠️  Illinois OSRM data not found - upload to data/illinois/"
fi

if [ -f "data/wisconsin/wisconsin-260218.osrm" ]; then
    echo "✅ Wisconsin OSRM data present"
else
    echo "⚠️  Wisconsin OSRM data not found - upload to data/wisconsin/"
fi

# Pull OSRM Docker image
echo "📦 Pulling OSRM Docker image..."
sudo docker pull osrm/osrm-backend

# Stop and remove old OSRM containers if they exist
sudo docker stop osrm-illinois osrm-wisconsin 2>/dev/null || true
sudo docker rm osrm-illinois osrm-wisconsin 2>/dev/null || true

# Start Illinois OSRM container
if [ -f "data/illinois/illinois-260201.osrm" ]; then
    echo "🚀 Starting Illinois OSRM server on port 5000..."
    sudo docker run -d \
        --name osrm-illinois \
        --restart always \
        -p 5000:5000 \
        -v /home/ubuntu/FoodTruckRouteOptimizer/data/illinois:/data \
        osrm/osrm-backend osrm-routed --algorithm mld /data/illinois-260201.osrm
fi

# Start Wisconsin OSRM container
if [ -f "data/wisconsin/wisconsin-260218.osrm" ]; then
    echo "🚀 Starting Wisconsin OSRM server on port 5002..."
    sudo docker run -d \
        --name osrm-wisconsin \
        --restart always \
        -p 5002:5000 \
        -v /home/ubuntu/FoodTruckRouteOptimizer/data/wisconsin:/data \
        osrm/osrm-backend osrm-routed --algorithm mld /data/wisconsin-260218.osrm
fi

# Create Python virtual environment
if [ ! -d "venv" ]; then
    echo "📦 Creating Python virtual environment..."
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
sudo supervisorctl start flask

echo "✅ Deployment complete!"
echo "Visit your server's IP address to access the app"
echo ""
echo "OSRM containers running:"
sudo docker ps --filter name=osrm
echo ""
echo "View logs:"
echo "  Flask: sudo tail -f /var/log/flask.log"
echo "  Illinois OSRM: sudo docker logs osrm-illinois"
echo "  Wisconsin OSRM: sudo docker logs osrm-wisconsin"
