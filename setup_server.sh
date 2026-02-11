#!/bin/bash
# ============================================
# SkinSight API — GCP VM Setup Script
# Run this ONCE after SSH-ing into your VM
# ============================================

set -e  # Exit on any error

echo "🔧 Step 1: Updating system packages..."
sudo apt update && sudo apt upgrade -y

echo "🐍 Step 2: Installing Python 3.11 & pip..."
sudo apt install -y python3.11 python3.11-venv python3.11-dev python3-pip git

echo "📦 Step 3: Cloning repository..."
cd ~
git clone https://github.com/gurusanjay2322/SkinSight-API.git
cd SkinSight-API

echo "🏗️ Step 4: Creating virtual environment..."
python3.11 -m venv venv
source venv/bin/activate

echo "📥 Step 5: Installing dependencies (this takes ~5 min)..."
pip install --upgrade pip
pip install -r requirements.txt

echo "🔑 Step 6: Creating .env file..."
cat > .env << 'EOF'
OPENAI_API_KEY=YOUR_OPENAI_KEY_HERE
OPENWEATHERMAP_API_KEY=YOUR_WEATHER_KEY_HERE
EOF

echo ""
echo "⚠️  IMPORTANT: Edit .env with your actual API keys!"
echo "   Run: nano .env"
echo ""

echo "✅ Step 7: Testing the app..."
python run.py &
sleep 10
curl -s http://localhost:5000/ && echo ""
kill %1 2>/dev/null

echo ""
echo "🎉 Setup complete! Now run the app with:"
echo "   cd ~/SkinSight-API"
echo "   source venv/bin/activate"
echo "   gunicorn --bind=0.0.0.0:5000 --timeout 120 --workers 2 run:app"
echo ""
echo "Or to run in background (survives SSH disconnect):"
echo "   nohup gunicorn --bind=0.0.0.0:5000 --timeout 120 --workers 2 run:app > server.log 2>&1 &"
