#!/bin/sh
set -e

HTPASSWD_FILE="$(dirname "$0")/../nginx/.htpasswd"

usage() {
    echo "Usage: $0 <username> [password]"
    echo ""
    echo "Add or update a user in .htpasswd file"
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

# Check if htpasswd is available
if ! command -v htpasswd >/dev/null 2>&1; then
    echo "ERROR: htpasswd not found"
    echo ""
    echo "Install it with:"
    echo "  macOS:   brew install httpd"
    echo "  Debian:  apt install apache2-utils"
    echo "  Alpine:  apk add apache2-utils"
    echo "  RHEL:    dnf install httpd-tools"
    exit 1
fi

# Create file if not exists
if [ ! -f "$HTPASSWD_FILE" ]; then
    touch "$HTPASSWD_FILE"
    echo "Created $HTPASSWD_FILE"
fi

# Add/update user
if [ -n "$PASSWORD" ]; then
    # Password provided as argument
    htpasswd -bB "$HTPASSWD_FILE" "$USERNAME" "$PASSWORD"
else
    # Prompt for password
    htpasswd -B "$HTPASSWD_FILE" "$USERNAME"
fi

echo ""
echo "User '$USERNAME' added/updated in $HTPASSWD_FILE"
