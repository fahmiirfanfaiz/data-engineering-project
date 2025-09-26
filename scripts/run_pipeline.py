#!/usr/bin/env python3
"""
COVID-19 Jakarta Dataset Analysis Pipeline Runner
Provides various execution options for the modular pipeline
"""

import argparse
import sys
import os
from pathlib import Path

# Add parent directory to path for imports
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

from main import COVIDDataPipeline, run_pipeline_phase

def main():
    parser = argparse.ArgumentParser(description='COVID-19 Jakarta Dataset Analysis Pipeline')
    
    # Add argument groups
    execution_group = parser.add_mutually_exclusive_group()
    execution_group.add_argument('--full', action='store_true', 
                                help='Run complete pipeline (default)')
    execution_group.add_argument('--phase', choices=['loading', 'cleaning', 'integration', 'reduction', 'visualization'],
                                help='Run specific phase only')
    
    # Pipeline options
    parser.add_argument('--no-viz', action='store_true',
                       help='Skip visualization creation')
    parser.add_argument('--no-report', action='store_true', 
                       help='Skip report generation')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    
    # Parse arguments
    args = parser.parse_args()
    
    print("=" * 80)
    print("COVID-19 Jakarta Dataset Analysis Pipeline")
    print("=" * 80)
    
    try:
        if args.phase:
            print(f"\nRunning {args.phase} phase only...")
            result = run_pipeline_phase(args.phase)
            if result:
                print(f"✓ {args.phase.title()} phase completed successfully!")
            else:
                print(f"✗ {args.phase.title()} phase failed!")
                sys.exit(1)
        else:
            print("\nRunning complete pipeline...")
            pipeline = COVIDDataPipeline(verbose=args.verbose)
            success = pipeline.run_full_pipeline(
                include_visualization=not args.no_viz,
                save_report=not args.no_report
            )
            
            if success:
                print("\n🎉 PIPELINE COMPLETED SUCCESSFULLY!")
                print("✓ All data processing phases executed")
                if not args.no_viz:
                    print("✓ Independent visualizations created")
                print("✓ Modular structure ready for use")
            else:
                print("\n❌ PIPELINE COMPLETED WITH ERRORS")
                print("⚠ Check logs for details")
                sys.exit(1)
                
    except KeyboardInterrupt:
        print("\n\n⚠ Pipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {str(e)}")
        sys.exit(1)
    
    print("=" * 80)

if __name__ == "__main__":
    main()