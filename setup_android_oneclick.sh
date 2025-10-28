#!/data/data/com.termux/files/usr/bin/bash
set -e

echo "🌍 Setting up Lytollis Chaos Lab (Android One-Click)..."

# Update pip & create venv
python3 -m venv .venv
source .venv/bin/activate

pip install --upgrade pip wheel setuptools

# Install scientific stack (Termux-friendly)
pip install numpy==1.26.4 scipy matplotlib networkx

# Make output directories
mkdir -p paper/figs

# Run all domains automatically
bash run_all.sh || true

# Export plots to shared Storage (visible in Gallery / Files)
mkdir -p /sdcard/Lytollis-chaos-law
cp -r paper/figs /sdcard/Lytollis-chaos-law/

echo "✅ COMPLETE!"
echo "📁 View plots with: Files → Internal Storage → Lytollis-chaos-law/figs"
echo "🔥 Done. No further steps."
