# Stage 1: Install nginx-module-acme from mainline repo
FROM alpine:latest AS builder

RUN apk add --no-cache wget \
    && wget -O /etc/apk/keys/nginx_signing.rsa.pub https://nginx.org/keys/nginx_signing.rsa.pub \
    && ALPINE_VERSION=$(cut -d. -f1,2 /etc/alpine-release) \
    && echo "https://nginx.org/packages/mainline/alpine/v${ALPINE_VERSION}/main" >> /etc/apk/repositories \
    && apk add --no-cache nginx nginx-module-acme

# Stage 2: Final image
FROM nginx:alpine

# Copy ACME module from builder
COPY --from=builder /usr/lib/nginx/modules/ngx_http_acme_module.so /usr/lib/nginx/modules/

# Install runtime dependencies
RUN apk add --no-cache \
    openssl \
    apache2-utils \
    gettext

# Create directories
RUN mkdir -p /var/cache/nginx/acme /etc/nginx/certs

# Copy config template and entrypoint
COPY nginx.conf.template /etc/nginx/nginx.conf.template
COPY docker-entrypoint.sh /docker-entrypoint.sh
RUN chmod +x /docker-entrypoint.sh

EXPOSE 80 443

ENTRYPOINT ["/docker-entrypoint.sh"]
CMD ["nginx", "-g", "daemon off;"]
