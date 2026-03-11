import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path

# Setup Project Root path to access src folder
root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))

# Import required functions from local filterbank module
from src.filterbank import mdct, imdct, _sin_window

def verify_with_plots():
    """Visualize MDCT/IMDCT reconstruction for a single isolated frame."""
    N = 2048
    
    # 1. Create a random reference signal
    np.random.seed(42)
    x = np.random.rand(N)
    
    # 2. Apply Sine window as per standard requirements
    w = _sin_window(N)
    x_windowed = x * w
    
    # 3. Perform Forward MDCT (Frequency Analysis)
    X = mdct(x_windowed)
    
    # 4. Perform Inverse MDCT (Time Synthesis)
    x_hat = imdct(X)
    
    # --- Console Output ---
    print("-" * 40)
    print("MDCT/IMDCT VISUAL INTEGRITY CHECK")
    print("-" * 40)
    print(f"Signal Max Amplitude: {np.max(np.abs(x_windowed)):.4f}")
    print(f"Reconstructed Max Amplitude: {np.max(np.abs(x_hat)):.4f}")
    
    # Check error in the central region where aliasing is minimal
    center_start, center_end = N // 4, 3 * N // 4
    max_err_center = np.max(np.abs(x_windowed[center_start:center_end] - x_hat[center_start:center_end]))
    print(f"Max Error in central region (N/4 to 3N/4): {max_err_center:.2e}")
    print("-" * 40)

    # --- Plotting for Visual Confirmation ---
    plt.figure(figsize=(12, 6))
    plt.plot(x_windowed, label='Original Windowed Signal', color='blue', alpha=0.6, linewidth=2)
    plt.plot(x_hat, label='IMDCT Result (Single Frame with Aliasing)', color='red', linestyle='--', alpha=0.8)
    
    # Highlight regions where time-domain aliasing is most prominent
    plt.axvspan(0, N//4, color='gray', alpha=0.2, label='Aliasing Region')
    plt.axvspan(3*N//4, N, color='gray', alpha=0.2)
    
    plt.title("MDCT/IMDCT Single Frame Analysis\n" + 
              "(Observe how aliasing distorts the edges while the center remains accurate)")
    plt.xlabel("Sample index (n)")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Define output directory path
    output_dir = root_dir / "output"
    
    # Save the plot for documentation
    save_path = output_dir / "mdct_one_frame_test.png"
    plt.savefig(save_path, dpi=300)
    print(f"Plot saved to: {save_path}")
    
    plt.show()

if __name__ == "__main__":
    verify_with_plots()