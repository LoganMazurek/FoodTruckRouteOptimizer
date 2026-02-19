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

# Create data directories if they don't exist
mkdir -p data/illinois
mkdir -p data/wisconsin

# Check for OSRM backend installation
if ! command -v osrm-routed &> /dev/null; then
    echo "📦 Installing OSRM backend..."
    sudo apt-get install -y osrm-backend
fi

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

# Install Python dependencies
pip3 install -r requirements.txt

# Create supervisor config for Illinois OSRM
sudo tee /etc/supervisor/conf.d/osrm-illinois.conf > /dev/null <<EOF
[program:osrm-illinois]
command=/usr/bin/osrm-routed --algorithm=MLD --port 5000 /home/ubuntu/FoodTruckRouteOptimizer/data/illinois/illinois-260201.osrm
autostart=true
autorestart=true
user=ubuntu
stdout_logfile=/var/log/osrm-illinois.log
stderr_logfile=/var/log/osrm-illinois_error.log
EOF

# Create supervisor config for Wisconsin OSRM
sudo tee /etc/supervisor/conf.d/osrm-wisconsin.conf > /dev/null <<EOF
[program:osrm-wisconsin]
command=/usr/bin/osrm-routed --algorithm=MLD --port 5002 /home/ubuntu/FoodTruckRouteOptimizer/data/wisconsin/wisconsin-260218.osrm
autostart=true
autorestart=true
user=ubuntu
stdout_logfile=/var/log/osrm-wisconsin.log
stderr_logfile=/var/log/osrm-wisconsin_error.log
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
sudo supervisorctl start osrm-illinois osrm-wisconsin flask

echo "✅ Deployment complete!"
echo "Visit your server's IP address to access the app"
echo "OSRM Illinois running on port 5000"
echo "OSRM Wisconsin running on port 5002"
echo "View logs with: sudo tail -f /var/log/flask.log"
