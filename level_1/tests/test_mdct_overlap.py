import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path

# Set up project root path to access src folder
root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))

from src.filterbank import mdct, imdct, _sin_window

def test_mdct_overlap():
    """Verify Time Domain Aliasing Cancellation (TDAC) in the overlap region."""
    N = 2048
    hop = 1024
    
    # 1. Create a continuous random signal
    np.random.seed(42)
    total_signal = np.random.rand(N + hop) 
    
    # 2. Prepare analysis/synthesis window
    w = _sin_window(N)
    
    # 3. Process Frame 0: Apply window -> MDCT -> IMDCT -> Apply window
    x0 = total_signal[0:N]
    y0 = imdct(mdct(x0 * w)) * w 
    
    # 4. Process Frame 1: Offset by hop size
    x1 = total_signal[hop:hop+N]
    y1 = imdct(mdct(x1 * w)) * w 
    
    # 5. Extract components in the overlap region (samples 1024 to 2047)
    overlap_original = total_signal[hop:N]
    part_from_y0 = y0[hop:N] # Second half of Frame 0
    part_from_y1 = y1[0:hop] # First half of Frame 1
    
    # 6. Final reconstruction via Overlap-Add (OLA)
    reconstructed = (part_from_y0 + part_from_y1)
    
    # --- Plotting Results ---
    plt.figure(figsize=(12, 7))
    
    # Plot individual contributions (these contain time-domain aliasing)
    plt.plot(part_from_y0, label='Contribution from Frame 0 (with Aliasing)', color='red', linestyle='--', alpha=0.5)
    plt.plot(part_from_y1, label='Contribution from Frame 1 (with Aliasing)', color='orange', linestyle='--', alpha=0.5)
    
    # Plot the sum of aliased signals
    plt.plot(reconstructed, label='Final Reconstructed Signal (After OLA)', color='blue', linewidth=2)
    
    # Plot the original signal for reference
    plt.plot(overlap_original, label='Original Reference Signal', color='black', linestyle=':', alpha=0.8)

    plt.title("Time Domain Aliasing Cancellation (TDAC) via Overlap-Add\n" + 
              "Observe how the red and orange aliased signals sum to the blue reconstructed signal")
    plt.xlabel("Sample index in overlap region (0 to 1023)")
    plt.ylabel("Amplitude")
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    
    # Error measurement
    max_err = np.max(np.abs(overlap_original - reconstructed))
    print(f"Max Error in overlap region: {max_err:.2e}")
    
    # Save visualization to output directory
    output_dir = root_dir / "output"
    save_path = output_dir / "mdct_frame_overlap_test.png"
    plt.savefig(save_path, dpi=300)
    print(f"Plot saved to: {save_path}")
    
    plt.show()

if __name__ == "__main__":
    test_mdct_overlap()