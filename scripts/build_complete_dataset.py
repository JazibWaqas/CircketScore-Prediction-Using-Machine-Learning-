#!/usr/bin/env python3
"""
Build Complete Comprehensive Dataset
Master script that builds all individual datasets and combines them into final dataset
"""

import os
import sys
import subprocess
import time

def run_script(script_name, description):
    """Run a Python script and handle errors"""
    print(f"\n{'='*60}")
    print(f"RUNNING: {description}")
    print(f"Script: {script_name}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run([sys.executable, script_name], 
                              capture_output=True, text=True, timeout=3600)
        
        if result.returncode == 0:
            print(f"✅ SUCCESS: {description}")
            print(result.stdout)
            return True
        else:
            print(f"❌ ERROR: {description}")
            print(f"Error: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"⏰ TIMEOUT: {description} (took longer than 1 hour)")
        return False
    except Exception as e:
        print(f"❌ EXCEPTION: {description}")
        print(f"Exception: {e}")
        return False

def main():
    """Main function to build complete dataset"""
    print("🏏 BUILDING COMPLETE CRICKET PREDICTION DATASET")
    print("=" * 60)
    print("This will create a comprehensive dataset with:")
    print("1. Individual player impact metrics")
    print("2. Venue and weather conditions")
    print("3. Team composition and chemistry")
    print("4. Combined comprehensive features")
    print("=" * 60)
    
    # Create output directory
    os.makedirs("processed_data", exist_ok=True)
    
    # List of scripts to run in order
    scripts = [
        ("scripts/build_player_impact_dataset.py", "Player Impact Dataset"),
        ("scripts/build_venue_conditions_dataset.py", "Venue Conditions Dataset"),
        ("scripts/build_team_composition_dataset.py", "Team Composition Dataset"),
        ("scripts/combine_final_dataset.py", "Final Comprehensive Dataset")
    ]
    
    # Track success/failure
    results = {}
    
    # Run each script
    for script, description in scripts:
        start_time = time.time()
        success = run_script(script, description)
        end_time = time.time()
        
        results[description] = {
            'success': success,
            'time': end_time - start_time
        }
        
        if not success:
            print(f"\n❌ FAILED: {description}")
            print("Stopping execution due to failure.")
            break
    
    # Summary
    print(f"\n{'='*60}")
    print("BUILD SUMMARY")
    print(f"{'='*60}")
    
    for description, result in results.items():
        status = "✅ SUCCESS" if result['success'] else "❌ FAILED"
        time_taken = f"{result['time']:.1f}s"
        print(f"{status} - {description} ({time_taken})")
    
    # Check if all succeeded
    all_success = all(result['success'] for result in results.values())
    
    if all_success:
        print(f"\n🎉 ALL DATASETS BUILT SUCCESSFULLY!")
        print(f"📁 Check the 'processed_data' folder for:")
        print(f"   - player_impact_dataset.csv")
        print(f"   - venue_conditions_dataset.csv")
        print(f"   - team_composition_dataset.csv")
        print(f"   - final_comprehensive_dataset.csv")
        print(f"\n🚀 Your comprehensive dataset is ready for XGBoost training!")
    else:
        print(f"\n❌ SOME DATASETS FAILED TO BUILD")
        print(f"Check the error messages above for details.")
        print(f"You may need to run individual scripts manually.")

if __name__ == "__main__":
    main()
