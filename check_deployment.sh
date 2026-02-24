#!/bin/bash
# Script to check deployment health and diagnose issues

echo "🔍 Food Truck Route Optimizer - Deployment Health Check"
echo "=========================================="
echo ""

# Check if Flask is running
echo "📋 Checking Flask service status..."
sudo supervisorctl status flask
echo ""

# Check recent Flask logs
echo "📜 Recent Flask logs (last 20 lines)..."
sudo tail -n 20 /var/log/flask.log
echo ""

echo "❌ Recent Flask errors (last 20 lines)..."
sudo tail -n 20 /var/log/flask_error.log
echo ""

# Check if temp directory exists and is writable
echo "📁 Checking temp directory..."
if [ -d "/home/ubuntu/FoodTruckRouteOptimizer/temp" ]; then
    echo "✅ temp directory exists"
    ls -la /home/ubuntu/FoodTruckRouteOptimizer/temp
    echo ""
    if [ -w "/home/ubuntu/FoodTruckRouteOptimizer/temp" ]; then
        echo "✅ temp directory is writable"
    else
        echo "❌ temp directory is NOT writable"
        echo "   Fix with: sudo chmod -R 755 /home/ubuntu/FoodTruckRouteOptimizer/temp"
    fi
else
    echo "❌ temp directory does NOT exist"
    echo "   Fix with: mkdir -p /home/ubuntu/FoodTruckRouteOptimizer/temp"
fi
echo ""

# Check nginx status
echo "🌐 Checking nginx status..."
sudo systemctl status nginx --no-pager | head -n 10
echo ""

# Test health endpoint
echo "🏥 Testing health endpoint..."
curl -s http://localhost:5001/health | python3 -m json.tool || echo "❌ Health endpoint not responding"
echo ""

# Check OSRM containers
echo "🚗 Checking OSRM containers..."
sudo docker ps --filter name=osrm
echo ""

# Check Python dependencies
echo "📦 Checking Python environment..."
if [ -f "/home/ubuntu/FoodTruckRouteOptimizer/venv/bin/python" ]; then
    echo "✅ Virtual environment exists"
    /home/ubuntu/FoodTruckRouteOptimizer/venv/bin/pip list | grep -E "(Flask|networkx|geopy|requests|overpy|waitress)"
else
    echo "❌ Virtual environment NOT found"
fi
echo ""

# Check disk space
echo "💾 Disk space..."
df -h /home/ubuntu
echo ""

# Check memory
echo "🧠 Memory usage..."
free -h
echo ""

echo "=========================================="
echo "✅ Health check complete!"
echo ""
echo "To view live logs, run:"
echo "  sudo tail -f /var/log/flask.log"
echo "  sudo tail -f /var/log/flask_error.log"
