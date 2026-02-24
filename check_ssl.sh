#!/bin/bash
# Check SSL certificate status

echo "🔐 SSL Certificate Status Check"
echo "================================"
echo ""

DOMAIN="nroute.loganmazurek.com"

# Check if certificate exists
if [ -f "/etc/letsencrypt/live/$DOMAIN/fullchain.pem" ]; then
    echo "✅ SSL Certificate exists for $DOMAIN"
    echo ""
    
    # Show certificate details
    sudo certbot certificates
    echo ""
    
    # Check expiration
    echo "📅 Certificate expiration:"
    sudo openssl x509 -in "/etc/letsencrypt/live/$DOMAIN/fullchain.pem" -noout -enddate
    echo ""
    
    # Check nginx SSL config
    if grep -q "ssl_certificate.*$DOMAIN" /etc/nginx/sites-available/default; then
        echo "✅ Nginx is configured for SSL"
    else
        echo "❌ Nginx SSL config missing - run: bash deploy.sh"
    fi
    echo ""
    
    # Check certbot timer
    echo "🔄 Auto-renewal status:"
    sudo systemctl status certbot.timer --no-pager | head -n 5
    echo ""
    
    # Test nginx config
    echo "🧪 Testing nginx configuration..."
    if sudo nginx -t 2>&1 | grep -q "successful"; then
        echo "✅ Nginx config is valid"
    else
        echo "❌ Nginx config has errors:"
        sudo nginx -t
    fi
    
else
    echo "❌ No SSL certificate found for $DOMAIN"
    echo ""
    echo "To set up SSL:"
    echo "  1. Edit setup_ssl.sh and set your email"
    echo "  2. Run: bash setup_ssl.sh"
    echo "  OR manually:"
    echo "  sudo certbot --nginx -d $DOMAIN"
    echo ""
    echo "After getting certificate, redeploy:"
    echo "  bash deploy.sh"
fi

echo ""
echo "================================"
echo ""
echo "Useful commands:"
echo "  Test renewal:  sudo certbot renew --dry-run"
echo "  Force renewal: sudo certbot renew --force-renewal"
echo "  View certs:    sudo certbot certificates"
echo "  Check nginx:   sudo nginx -t"
