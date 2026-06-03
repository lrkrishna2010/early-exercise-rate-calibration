"""End-to-end runner: LSM validation, Heston-American, curve + Hull-White, figures."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analysis
if __name__ == "__main__":
    analysis.main()
