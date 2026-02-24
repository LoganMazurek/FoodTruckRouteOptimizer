#!/bin/bash
# Quick fix script for stale Python cache and outdated code

echo "🔧 Quick Fix for Template Errors"
echo "================================="

cd /home/ubuntu/FoodTruckRouteOptimizer

echo "1. Stopping Flask..."
sudo supervisorctl stop flask

echo "2. Clearing Python cache..."
find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
find . -type f -name "*.pyc" -delete 2>/dev/null || true

echo "3. Pulling latest code..."
git stash
git pull origin main

echo "4. Checking templates..."
ls -la templates/

echo "5. Restarting Flask..."
sudo supervisorctl start flask

sleep 2

echo "6. Checking Flask status..."
sudo supervisorctl status flask

echo ""
echo "✅ Quick fix complete!"
echo ""
echo "Test the app now. If still broken, run full deploy:"
echo "  bash deploy.sh"
