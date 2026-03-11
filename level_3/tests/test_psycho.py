import numpy as np
import sys
import os
from pathlib import Path
import matplotlib.pyplot as plt

# Setup paths for module access
test_file_path = Path(__file__).resolve()
project_root = test_file_path.parent.parent
sys.path.insert(0, str(project_root))

# Imports from project files
from src.psycho import psycho, spreading_function
from src.tns import load_band_tables
import src.quantizer as q_mod 
import src.psycho as p_mod

# Inject quantizer functions into the psycho module
# This is necessary because psycho.py calls these functions internally
p_mod.compute_fft_spectrum = q_mod.compute_fft_spectrum
p_mod.extract_esh_subframes = q_mod.extract_esh_subframes
p_mod.spectral_predictability = q_mod.spectral_predictability
p_mod.band_energy_and_weighted_predictability = q_mod.band_energy_and_weighted_predictability
p_mod.apply_spreading_and_normalization = q_mod.apply_spreading_and_normalization
p_mod.compute_energy_threshold = q_mod.compute_energy_threshold
p_mod.compute_noise_partition = q_mod.compute_noise_partition

def init_spreading_matrices():
    """Initialize spreading matrices using the custom spreading_function."""
    original_cwd = os.getcwd()
    os.chdir(project_root)
    bj_long, bj_short = load_band_tables("TableB219.mat")
    
    def get_S(bj):
        n = bj.shape[0]
        S = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                S[i, j] = spreading_function(i, j, bj)
        return S

    print("[*] Calculating spreading matrices...")
    S_l, S_s = get_S(bj_long), get_S(bj_short)
    os.chdir(original_cwd)
    return S_l, S_s

def run_psycho_test():
    """Verify the Psychoacoustic model using Tone vs Noise signals."""
    print("="*60)
    print("PSYCHOACOUSTIC MODEL INTEGRITY TEST")
    print("="*60)

    # Initialization
    S_long, S_short = init_spreading_matrices()
    N = 2048
    fs = 44100
    t = np.arange(N) / fs

    # Signal Generation
    # Case A: 1kHz Tone (Highly predictable/tonal)
    tone_curr = 0.5 * np.sin(2 * np.pi * 1000 * t)
    
    # Case B: White Noise (Random/unpredictable)
    np.random.seed(42)
    noise_curr = np.random.normal(0, 0.02, N)
    noise_prev1 = np.random.normal(0, 0.02, N) # Different sample
    noise_prev2 = np.random.normal(0, 0.02, N) # Different sample

    original_cwd = os.getcwd()
    os.chdir(project_root)
    
    print("[*] Running psycho analysis...")
    # For tone, we send identical frames to demonstrate perfect predictability
    smr_tone = psycho(tone_curr, "OLS", tone_curr, tone_curr, S_long, S_short)
    
    # For noise, we send different samples to demonstrate low predictability
    smr_noise = psycho(noise_curr, "OLS", noise_prev1, noise_prev2, S_long, S_short)
    
    os.chdir(original_cwd)

    # Results & Visualization
    print(f"\n[*] Tone Max SMR: {np.max(smr_tone):.2f}")
    print(f"[*] Noise Max SMR: {np.max(smr_noise):.2f}")

    # Plot SMR in dB for better visualization of the dynamic range
    plt.figure(figsize=(12, 8))
    
    # Tone subplot
    plt.subplot(2, 1, 1)
    plt.plot(10 * np.log10(smr_tone + 1e-15), label='Tone SMR (dB)', color='blue')
    plt.title("SMR in dB for 1kHz Tone (High Predictability)")
    plt.ylabel("SMR (dB)")
    plt.grid(True, alpha=0.3)
    plt.legend()

    # Noise subplot
    plt.subplot(2, 1, 2)
    plt.plot(10 * np.log10(smr_noise + 1e-15), label='Noise SMR (dB)', color='red')
    plt.title("SMR in dB for White Noise (Low Predictability)")
    plt.xlabel("Scale Factor Band Index")
    plt.ylabel("SMR (dB)")
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.tight_layout()
    output_dir = project_root / "output"
    output_dir.mkdir(exist_ok=True)
    plt.savefig(output_dir / "psycho_test_results.png")
    
    print(f"\n[SUCCESS] Peak detected at band: {np.argmax(smr_tone)}")
    print(f"[*] Plot saved in {output_dir}/psycho_test_results.png")
    plt.show()

if __name__ == "__main__":
    run_psycho_test()