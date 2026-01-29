#!/bin/sh
set -e

CERTS_DIR="/etc/nginx/certs"
ACME_DIR="/var/cache/nginx/acme"
DHPARAM_FILE="${CERTS_DIR}/dhparam.pem"
SELF_SIGNED_CERT="${CERTS_DIR}/selfsigned.crt"
SELF_SIGNED_KEY="${CERTS_DIR}/selfsigned.key"

# Defaults
USE_LETSENCRYPT="${USE_LETSENCRYPT:-false}"
LETSENCRYPT_ENV="${LETSENCRYPT_ENV:-staging}"

# ACME servers
ACME_STAGING="https://acme-staging-v02.api.letsencrypt.org/directory"
ACME_PRODUCTION="https://acme-v02.api.letsencrypt.org/directory"

# Determine server name (DOMAIN takes precedence over IP)
if [ -n "$DOMAIN" ]; then
    SERVER_NAME="$DOMAIN"
elif [ -n "$IP" ]; then
    SERVER_NAME="$IP"
else
    echo "ERROR: Either DOMAIN or IP must be set"
    exit 1
fi

# Determine ACME server
if [ "$LETSENCRYPT_ENV" = "production" ]; then
    ACME_SERVER="$ACME_PRODUCTION"
else
    ACME_SERVER="$ACME_STAGING"
fi

# Validate ACME_EMAIL if Let's Encrypt is enabled
if [ "$USE_LETSENCRYPT" = "true" ] && [ -z "$ACME_EMAIL" ]; then
    echo "ERROR: ACME_EMAIL is required when USE_LETSENCRYPT=true"
    exit 1
fi

echo "==> Configuration:"
echo "    Server name: $SERVER_NAME"
echo "    Use Let's Encrypt: $USE_LETSENCRYPT"
echo "    Let's Encrypt env: $LETSENCRYPT_ENV"

# Generate DH parameters if not exists
if [ ! -f "$DHPARAM_FILE" ]; then
    echo "==> Generating DH parameters (2048 bit)..."
    openssl dhparam -out "$DHPARAM_FILE" 2048
    echo "==> DH parameters generated"
fi

# Prepare SSL certificate configuration
if [ "$USE_LETSENCRYPT" = "true" ]; then
    echo "==> Using Let's Encrypt certificates"

    # ACME certificate directive
    SSL_CERTIFICATE_CONFIG="acme_certificate letsencrypt ${SERVER_NAME};
        ssl_certificate \$acme_certificate;
        ssl_certificate_key \$acme_certificate_key;"
else
    echo "==> Using self-signed certificates"

    # Generate self-signed certificate if not exists or SERVER_NAME changed
    CERT_CN=""
    if [ -f "$SELF_SIGNED_CERT" ]; then
        CERT_CN=$(openssl x509 -in "$SELF_SIGNED_CERT" -noout -subject 2>/dev/null | sed -n 's/.*CN *= *\([^,]*\).*/\1/p')
    fi

    if [ ! -f "$SELF_SIGNED_CERT" ] || [ "$CERT_CN" != "$SERVER_NAME" ]; then
        echo "==> Generating self-signed certificate for $SERVER_NAME..."

        # Check if IP or domain
        if echo "$SERVER_NAME" | grep -qE '^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$'; then
            # IP address
            SAN="IP:${SERVER_NAME}"
        else
            # Domain name
            SAN="DNS:${SERVER_NAME}"
        fi

        openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
            -keyout "$SELF_SIGNED_KEY" \
            -out "$SELF_SIGNED_CERT" \
            -subj "/CN=${SERVER_NAME}" \
            -addext "subjectAltName=${SAN}"

        echo "==> Self-signed certificate generated"
    fi

    SSL_CERTIFICATE_CONFIG="ssl_certificate ${SELF_SIGNED_CERT};
        ssl_certificate_key ${SELF_SIGNED_KEY};"
fi

# Export variables for envsubst
export SERVER_NAME
export ACME_SERVER
export ACME_EMAIL
export SSL_CERTIFICATE_CONFIG

# Generate nginx config from template
echo "==> Generating nginx configuration..."
envsubst '${SERVER_NAME} ${ACME_SERVER} ${ACME_EMAIL} ${SSL_CERTIFICATE_CONFIG}' \
    < /etc/nginx/nginx.conf.template \
    > /etc/nginx/nginx.conf

# Validate nginx config
echo "==> Validating nginx configuration..."
nginx -t

echo "==> Starting nginx..."
exec "$@"
