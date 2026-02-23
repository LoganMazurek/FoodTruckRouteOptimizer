#!/bin/bash
# Diagnostic script for droplet deployment issues

echo "=== CHECKING SERVICES ==="
echo ""

echo "Flask status:"
sudo supervisorctl status flask
echo ""

echo "Nginx status:"
sudo systemctl status nginx --no-pager
echo ""

echo "Docker containers:"
sudo docker ps -a
echo ""

echo "=== CHECKING PORTS ==="
echo ""

echo "Listening ports:"
sudo netstat -tlnp | grep -E ':(80|5000|5001) '
echo ""

echo "=== TESTING FLASK LOCALLY ==="
echo ""

echo "Curl Flask on port 5001:"
curl -I http://127.0.0.1:5001 2>&1 | head -5
echo ""

echo "=== CHECKING NGINX CONFIG ==="
echo ""

echo "Nginx config test:"
sudo nginx -t
echo ""

echo "Active nginx sites:"
ls -la /etc/nginx/sites-enabled/
echo ""

echo "=== CHECKING FIREWALL ==="
echo ""

echo "UFW status:"
sudo ufw status
echo ""

echo "iptables rules:"
sudo iptables -L -n | grep -E '(80|5000|5001)'
echo ""

echo "=== CHECKING LOGS ==="
echo ""

echo "Last 10 lines of Flask log:"
sudo tail -10 /var/log/flask.log
echo ""

echo "Last 10 lines of Nginx error log:"
sudo tail -10 /var/log/nginx/error.log
echo ""

echo "=== NETWORK INFO ==="
echo ""

echo "Public IP address:"
curl -s ifconfig.me
echo ""
echo ""

echo "DNS resolution for nroute.loganmazurek.com:"
nslookup nroute.loganmazurek.com
echo ""
