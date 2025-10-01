#!/bin/bash

echo "🚀 Setting up the social_momentum project..."

# Navigate to the project folder
cd ~/Documents/social_momentum || { echo "❌ Error: Directory not found!"; exit 1; }

# Create the main project structure
echo "📂 Creating directories..."
mkdir -p src/{controllers,perception,simulation,utils} tests docs

# Create empty Python files
echo "📜 Creating Python files..."
touch src/controllers/{planner.py,robot_controller.py}
touch src/perception/{sensors.py,human_tracker.py}
touch src/simulation/environment.py
touch src/utils/math_utils.py
touch src/main.py
touch requirements.txt README.md

# Set up virtual environment
echo "🐍 Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Install necessary Python libraries
echo "📦 Installing dependencies..."
pip install --upgrade pip
pip install numpy matplotlib

echo "✅ Setup complete! Run 'source venv/bin/activate' to activate your virtual environment."