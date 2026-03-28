#!/bin/bash
# ── InnovativeAIs Chatbot — EC2 Deployment Script ────────────────────────────
# Run this on your EC2 instance (Ubuntu 22.04)
# Usage: bash deploy.sh

set -e

echo "======================================"
echo " InnovativeAIs Chatbot EC2 Deployer"
echo "======================================"

# ── Step 1: Update system ─────────────────────────────────────────────────────
echo "[1/6] Updating system..."
sudo apt-get update -y
sudo apt-get install -y docker.io docker-compose nginx certbot python3-certbot-nginx

# ── Step 2: Start Docker ──────────────────────────────────────────────────────
echo "[2/6] Starting Docker..."
sudo systemctl start docker
sudo systemctl enable docker
sudo usermod -aG docker $USER

# ── Step 3: Create app directory ──────────────────────────────────────────────
echo "[3/6] Setting up app directory..."
mkdir -p ~/chatbot
cd ~/chatbot

# ── Step 4: Create .env file ──────────────────────────────────────────────────
echo "[4/6] Creating environment file..."
cat > .env << 'ENVEOF'
OPENAI_API_KEY=YOUR_OPENAI_KEY_HERE
SESSIONS_TABLE=rag-sessions
AWS_DEFAULT_REGION=ap-south-1
AWS_ACCESS_KEY_ID=YOUR_AWS_ACCESS_KEY
AWS_SECRET_ACCESS_KEY=YOUR_AWS_SECRET_KEY
ENVEOF

echo ">> IMPORTANT: Edit .env file with your actual keys!"
echo ">> Run: nano ~/chatbot/.env"

# ── Step 5: Create docker-compose.yml ────────────────────────────────────────
echo "[5/6] Creating docker-compose.yml..."
cat > docker-compose.yml << 'COMPOSEEOF'
version: '3.8'
services:
  chatbot:
    build: .
    ports:
      - "8000:8000"
    env_file:
      - .env
    restart: always
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
COMPOSEEOF

# ── Step 6: Create Nginx config ───────────────────────────────────────────────
echo "[6/6] Creating Nginx config..."
sudo bash -c 'cat > /etc/nginx/sites-available/chatbot << NGINXEOF
server {
    listen 80;
    server_name _;

    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_read_timeout 120s;
    }
}
NGINXEOF'

sudo ln -sf /etc/nginx/sites-available/chatbot /etc/nginx/sites-enabled/
sudo rm -f /etc/nginx/sites-enabled/default
sudo nginx -t && sudo systemctl restart nginx

echo ""
echo "======================================"
echo " Setup Complete!"
echo "======================================"
echo ""
echo "Next steps:"
echo "1. Copy your files: scp main.py Dockerfile requirements.txt ubuntu@YOUR_EC2_IP:~/chatbot/"
echo "2. Edit env file:   nano ~/chatbot/.env"
echo "3. Build & run:     cd ~/chatbot && docker-compose up -d --build"
echo "4. Test:            curl http://YOUR_EC2_IP/health"
echo ""
echo "API will be available at: http://YOUR_EC2_IP/chat"
echo ""