#!/usr/bin/env bash
echo "🌍 Lytollis Chaos Lab — Android One-Click Runner"

# Ensure packages
pkg update -y
pkg upgrade -y
pkg install -y python git

# Create venv
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install numpy scipy matplotlib

echo "📁 Running full ta-URT domain simulations..."
bash run_all.sh

echo "✅ DONE"
echo "📂 Results saved to: paper/figs"
