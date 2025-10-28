#!/data/data/com.termux/files/usr/bin/bash
set -e
source .venv/bin/activate

echo "🚀 Running Full Lytollis Chaos Law Cross-Domain Demo..."

python3 domains/chaos/run.py
python3 domains/plasma/run.py
python3 domains/cosmology/run.py
python3 domains/robotics/run.py
python3 domains/powergrid/run.py
python3 domains/epidemic/run.py
python3 domains/neuro/run.py

echo "✅ All simulations complete."
echo "📁 Plots saved to: paper/figs and /sdcard/Lytollis-chaos-law/figs"
