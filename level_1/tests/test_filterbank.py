import numpy as np
import sys
from pathlib import Path

# Setup Project Root path to access src folder
root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))

from src.filterbank import filter_bank, i_filter_bank

def verify_filterbank_precision():
    """Verify MDCT reconstruction precision across different frame types."""
    N = 2048
    hop = 1024
    
    # 1. Create a random stereo signal (3072 samples for 2 frames)
    np.random.seed(42)
    x = np.random.rand(hop + N, 2) 
    
    # 2. Define test parameters
    test_types = ["OLS", "LSS", "LPS", "ESH"]
    win_type = "KBD"

    print(f"{'Frame Type':<12} | {'Max Error':<10}")
    print("-" * 25)

    for f_type in test_types:
        # Define Frame 0 and Frame 1 with 50% overlap (hop=1024)
        f0_T = x[0:2048, :]    
        f1_T = x[1024:3072, :] 

        # Forward Transform (Analysis)
        f0_F = filter_bank(f0_T, "OLS", "KBD")
        f1_F = filter_bank(f1_T, "LSS", "KBD")

        # Inverse Transform (Synthesis)
        t0 = i_filter_bank(f0_F, "OLS", "KBD")
        t1 = i_filter_bank(f1_F, "LSS", "KBD")

        # Perform Overlap-Add (OLA) in the overlapping region
        reconstructed = t0[1024:2048, :] + t1[0:1024, :]

        # Compare with original signal
        original = x[1024:2048, :]
        max_err = np.max(np.abs(original - reconstructed))
        
        print(f"{f_type:<12} | {max_err:.2e}")

if __name__ == "__main__":
    verify_filterbank_precision()