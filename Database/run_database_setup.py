#!/usr/bin/env python3
"""
Run the complete database setup
"""

import subprocess
import sys
import os

def run_setup():
    print("🏏 Cricket Score Prediction - Database Setup")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not os.path.exists('../data/team_lookup.csv'):
        print("❌ Error: Please run this script from the database/ directory")
        print("   Make sure you're in the database/ folder and the data/ folder exists in the parent directory")
        sys.exit(1)
    
    # Install requirements
    print("📦 Installing Python requirements...")
    try:
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'], check=True)
        print("✅ Requirements installed")
    except subprocess.CalledProcessError:
        print("❌ Error installing requirements")
        sys.exit(1)
    
    # Run database setup
    print("\n🗄️ Setting up database...")
    try:
        subprocess.run([sys.executable, 'setup_database.py'], check=True)
        print("✅ Database setup complete")
    except subprocess.CalledProcessError:
        print("❌ Error setting up database")
        sys.exit(1)
    
    print("\n🎉 Setup complete!")
    print("🚀 You can now run the Flask API server with: python app.py")

if __name__ == "__main__":
    run_setup()
