#!/bin/bash
# Set up SSL certificate for nroute.loganmazurek.com

set -e

DOMAIN="nroute.loganmazurek.com"
EMAIL="your.email@example.com"  # Change this!

echo "🔐 SSL Certificate Setup for $DOMAIN"
echo "====================================="

# Check if certificate already exists
if [ -f "/etc/letsencrypt/live/$DOMAIN/fullchain.pem" ]; then
    echo "✅ SSL certificate already exists!"
    echo ""
    sudo certbot certificates
    echo ""
    echo "Certificate info displayed above."
    echo ""
    echo "To renew: sudo certbot renew"
    echo "To force renew: sudo certbot renew --force-renewal"
    exit 0
fi

# Check if email is still the placeholder
if [ "$EMAIL" = "your.email@example.com" ]; then
    echo "❌ Please edit this script and set your email address!"
    echo "   Open: nano setup_ssl.sh"
    echo "   Change: EMAIL=\"your.email@example.com\""
    echo "   To:     EMAIL=\"youremail@domain.com\""
    exit 1
fi

# Install certbot if not installed
if ! command -v certbot &> /dev/null; then
    echo "📦 Installing certbot..."
    sudo apt-get update
    sudo apt-get install -y certbot python3-certbot-nginx
fi

# Get certificate
echo "🔐 Requesting SSL certificate for $DOMAIN..."
echo "    Email: $EMAIL"
echo ""

sudo certbot --nginx -d "$DOMAIN" \
    --non-interactive \
    --agree-tos \
    --email "$EMAIL" \
    --redirect

echo ""
echo "✅ SSL certificate installed!"
echo ""
echo "Test auto-renewal:"
echo "  sudo certbot renew --dry-run"
echo ""
echo "View certificates:"
echo "  sudo certbot certificates"
echo ""
echo "Now redeploy to update nginx config:"
echo "  cd /home/ubuntu/FoodTruckRouteOptimizer"
echo "  bash deploy.sh"
