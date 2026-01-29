#!/bin/sh
set -e

# Path to the .htpasswd file (relative to script location)
HTPASSWD_FILE="$(cd "$(dirname "$0")/../nginx" && pwd)/.htpasswd"

usage() {
    echo "Usage: $0 <username> [password]"
    echo ""
    echo "Add or update a user in .htpasswd file using Docker"
    echo ""
    echo "Arguments:"
    echo "  username    Username to add"
    echo "  password    Password (optional, will prompt if not provided)"
    echo ""
    echo "Examples:"
    echo "  $0 admin                  # Will prompt for password"
    echo "  $0 admin mysecretpass     # Set password directly"
    exit 1
}

if [ -z "$1" ]; then
    usage
fi

USERNAME="$1"
PASSWORD="$2"

# Check if docker is available
if ! command -v docker >/dev/null 2>&1; then
    echo "ERROR: docker not found."
    echo "This script requires Docker to run the htpasswd utility."
    exit 1
fi

# Ensure the file exists so we can mount it
if [ ! -f "$HTPASSWD_FILE" ]; then
    touch "$HTPASSWD_FILE"
    echo "Created $HTPASSWD_FILE"
fi

echo "Using Docker to run htpasswd..."

# Add/update user
if [ -n "$PASSWORD" ]; then
    docker run --rm \
        -v "$HTPASSWD_FILE:/auth/.htpasswd" \
        httpd:alpine \
        htpasswd -bB /auth/.htpasswd "$USERNAME" "$PASSWORD"
else
    docker run --rm -it \
        -v "$HTPASSWD_FILE:/auth/.htpasswd" \
        httpd:alpine \
        htpasswd -B /auth/.htpasswd "$USERNAME"
fi

echo ""
echo "User '$USERNAME' added/updated in $HTPASSWD_FILE"
