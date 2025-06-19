# pixal/cli.py
import traceback
import sys
import argparse
from pathlib import Path
#from pixal import train, detect
from pixal.modules.config_loader import load_config
from pixal.preprocessing import runner as preprocessing_runner



def main():
    run_training = None  # placeholder to supress TensorFlow output
    detect = None       # placeholder to supress Tensorflow output
    try:
        parser = argparse.ArgumentParser(prog="pixal", description="Pixel-based Anomaly Detection CLI")
        subparsers = parser.add_subparsers(dest="command", required=True)

        # Validate
        detect_cmd = subparsers.add_parser("validate", help="Run validation (preprocess + detect) on new images")
        detect_cmd.add_argument("--input","-i", required=True, help="Folder with test images")
        detect_cmd.add_argument("--quiet", "-q", help="Quiet output", action="store_true")

        # Detect
        detect_cmd = subparsers.add_parser("detect", help="Run anomaly detection on new images")
        #detect_cmd.add_argument("--input","-i", required=True, help="Folder with test images")
        detect_cmd.add_argument("--quiet", "-q", help="Quiet output", action="store_true")
        
        args = parser.parse_args()
        

        if args.command == "validate":
            if detect is None:
                from pixal.validate import runner as validation_runner
            validation_runner.run_validation(args.input,quiet=args.quiet)
        elif args.command == "detect":
            if detect is None:
                from pixal.validate import runner as validation_runner
            validation_runner.run_detection(quiet=args.quiet)
    except Exception as e:
        exc_type, exc_value, exc_tb = sys.exc_info()
        tb = traceback.extract_tb(exc_tb)[-1]  # Get the last traceback entry
        filename = tb.filename
        line_number = tb.lineno
        print(f"❌ PIXAL-LITE CLI crashed: {e} (File: {filename}, Line: {line_number})")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        exc_type, exc_value, exc_tb = sys.exc_info()
        tb = traceback.extract_tb(exc_tb)[-1]  # Get the last traceback entry
        filename = tb.filename
        line_number = tb.lineno
        print(f"❌ PIXAL-LITE CLI crashed: {e} (File: {filename}, Line: {line_number})")
