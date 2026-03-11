import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path

# Set project root for level_2 access
root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))

from src.tns import tns, i_tns

def run_full_tns_test():
    """Verify TNS analysis and synthesis across all AAC frame types."""
    # Create output directory for plots
    out_folder = root_dir / "output"
    out_folder.mkdir(exist_ok=True)
    
    # List of frame types defined in the AAC standard
    test_types = ["OLS", "LSS", "LPS", "ESH"]
    
    print(f"{'Frame Type':<10} | {'Max Error':<12}")
    print("-" * 25)

    for f_type in test_types:
        # Define MDCT dimensions based on frame type
        if f_type == "ESH":
            shape = (128, 8) # 8 subframes of 128 samples each
        else:
            shape = (1024, 1) # Standard Long frames (OLS, LSS, LPS)
            
        # Generate random MDCT coefficients for testing
        np.random.seed(42) 
        X_orig = np.random.randn(*shape) * 10.0
        
        # 1. Forward TNS (Encoding/Filtering)
        X_tns, coeffs = tns(X_orig, f_type)
        
        # 2. Inverse TNS (Decoding/Reconstruction)
        X_rec = i_tns(X_tns, f_type, coeffs)
        
        # 3. Calculate Maximum Reconstruction Error
        error = np.max(np.abs(X_orig - X_rec))
            
        print(f"{f_type:<10} | {error:.2e}")
        
        # --- Visualization for Integrity Check ---
        plt.figure(figsize=(12, 4))
        plt.suptitle(f"TNS Integrity Check: {f_type}")
        
        # Plot 1: Compare original MDCT vs filtered TNS coefficients
        plt.subplot(1, 2, 1)
        plt.plot(X_orig.flatten(), label='Original MDCT', alpha=0.6)
        plt.plot(X_tns.flatten(), label='TNS Filtered', alpha=0.6)
        plt.xlabel("Frequency Bin")
        plt.ylabel("Amplitude")
        plt.legend()
        
        # Plot 2: Reconstruction error signal
        plt.subplot(1, 2, 2)
        plt.plot((X_orig - X_rec).flatten(), color='red', lw=0.5)
        plt.title(f"Reconstruction Error (Max: {error:.2e})")
        plt.xlabel("Frequency Bin")
        plt.ylabel("Error")
        
        plt.tight_layout()
        # Save plot to output folder
        plt.savefig(out_folder / f"full_tns_check_{f_type}.png")
        plt.close() 

if __name__ == "__main__":
    run_full_tns_test()