# Deployment & Troubleshooting Guide

## Quick Redeploy

After pushing changes to GitHub, SSH into your server and run:

```bash
cd /home/ubuntu/FoodTruckRouteOptimizer
git pull origin main
bash deploy.sh
```

**Note:** The deploy script now preserves SSL certificates. Your HTTPS configuration will remain intact after redeployment.

## SSL Certificate Setup (First Time Only)

If you get `NET::ERR_CERT_COMMON_NAME_INVALID`, you need to set up SSL:

```bash
# Install certbot
sudo apt-get install -y certbot python3-certbot-nginx

# Get SSL certificate (replace email with your email)
sudo certbot --nginx -d nroute.loganmazurek.com --non-interactive --agree-tos --email your.email@example.com

# Test auto-renewal
sudo certbot renew --dry-run
```

**Important:** The deploy script now preserves SSL certificates! Once you run certbot once, redeploying won't break HTTPS.

### Certbot Auto-Renewal

Certbot automatically sets up a systemd timer for certificate renewal. Check it with:

```bash
# Check renewal timer status
sudo systemctl status certbot.timer

# Test renewal (dry run)
sudo certbot renew --dry-run

# List certificates
sudo certbot certificates
```

Certificates auto-renew 30 days before expiration.

## Troubleshooting

### 1. Check Deployment Health

```bash
bash /home/ubuntu/FoodTruckRouteOptimizer/check_deployment.sh
```

### 2. Check Flask Logs

```bash
# View recent logs
sudo tail -n 50 /var/log/flask.log
sudo tail -n 50 /var/log/flask_error.log

# Follow logs in real-time
sudo tail -f /var/log/flask.log
```

### 3. Test Health Endpoint

```bash
curl http://localhost:5001/health | python3 -m json.tool
```

Expected response:
```json
{
  "status": "healthy",
  "python_version": "3.x.x",
  "flask_running": true,
  "temp_dir_exists": true,
  "temp_dir_writable": true
}
```

### 4. Common Issues

#### "Internal Server Error" on ZIP Code Input

**Possible causes:**
- Temp directory doesn't exist or isn't writable
- Nominatim geocoding API is rate-limited
- Missing Python dependencies

**Solutions:**
```bash
# Ensure temp directory exists and is writable
mkdir -p /home/ubuntu/FoodTruckRouteOptimizer/temp
chmod -R 755 /home/ubuntu/FoodTruckRouteOptimizer/temp
chown -R ubuntu:ubuntu /home/ubuntu/FoodTruckRouteOptimizer

# Check Flask logs for specific error
sudo tail -n 100 /var/log/flask_error.log

# Reinstall Python dependencies
cd /home/ubuntu/FoodTruckRouteOptimizer
venv/bin/pip install -r requirements.txt

# Restart Flask
sudo supervisorctl restart flask
```

#### Flask Service Won't Start

```bash
# Check supervisor status
sudo supervisorctl status flask

# Try starting manually
sudo supervisorctl start flask

# If it fails, check for Python errors
cd /home/ubuntu/FoodTruckRouteOptimizer
venv/bin/python app.py
```

#### SSL Certificate Issues

If HTTPS is not working or you see `NET::ERR_CERT_COMMON_NAME_INVALID`:

```bash
# Check if certificate exists
sudo ls -la /etc/letsencrypt/live/nroute.loganmazurek.com/

# View certificate details
sudo certbot certificates

# If certificate exists, redeploy to update nginx config
cd /home/ubuntu/FoodTruckRouteOptimizer
bash deploy.sh

# If certificate doesn't exist, set it up
bash setup_ssl.sh  # (edit email in script first)
# OR manually:
sudo certbot --nginx -d nroute.loganmazurek.com

# Check nginx configuration
sudo nginx -t
sudo systemctl reload nginx
```

**Common SSL Issues:**
- **Certificate expired**: Run `sudo certbot renew --force-renewal`
- **Nginx not using SSL**: Check `/etc/nginx/sites-available/default` has SSL config
- **Certbot not auto-renewing**: Check `sudo systemctl status certbot.timer`

### 5. Manual Service Control

```bash
# Restart all services
sudo supervisorctl restart flask
sudo systemctl restart nginx

# Stop services
sudo supervisorctl stop flask
sudo systemctl stop nginx

# View supervisor config
cat /etc/supervisor/conf.d/flask.conf

# View nginx config
cat /etc/nginx/sites-available/default
```

### 6. Environment Variables

The Flask app uses these environment variables (set in supervisor config):
- `SECRET_KEY`: Random key for Flask sessions (auto-generated in deploy.sh)
- `FLASK_ENV`: Set to "production"

To change them:
```bash
sudo nano /etc/supervisor/conf.d/flask.conf
sudo supervisorctl reread
sudo supervisorctl update
sudo supervisorctl restart flask
```

## Performance Monitoring

```bash
# Check memory usage
free -h

# Check disk space
df -h

# Check CPU usage
top

# Monitor Flask in real-time
sudo tail -f /var/log/flask.log /var/log/flask_error.log
```

## Database/Storage

The app stores temporary graph data in:
```
/home/ubuntu/FoodTruckRouteOptimizer/temp/
```

To clear old cached data:
```bash
rm -f /home/ubuntu/FoodTruckRouteOptimizer/temp/*_graph.pkl
rm -f /home/ubuntu/FoodTruckRouteOptimizer/temp/*_route.pkl
```

## Useful Commands Reference

```bash
# Full redeploy
cd /home/ubuntu/FoodTruckRouteOptimizer && git pull && bash deploy.sh

# Quick restart (no rebuild)
sudo supervisorctl restart flask

# View all logs
sudo tail -f /var/log/flask*.log

# Health check
bash check_deployment.sh

# Test app locally
cd /home/ubuntu/FoodTruckRouteOptimizer
venv/bin/python app.py
```
