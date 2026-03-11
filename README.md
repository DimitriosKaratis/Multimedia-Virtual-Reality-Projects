# MPEG-4 AAC Audio Codec Implementation

An implementation of a high-fidelity audio compression system based on the **MPEG-4 AAC (Advanced Audio Coding)** international standard. This project was developed as part of the **Multimedia Systems** course (2025-2026) at the Aristotle University of Thessaloniki.

## 🤖 Project Overview

The system is a lossy audio codec designed to achieve significant data reduction while maintaining high subjective audio quality by utilizing psychoacoustic modeling. The implementation follows a modular three-level architecture:

* **Level 1 (Filterbank & SSC):** Time-to-frequency transformation using MDCT and transient detection via Sequence Segmentation Control.
* **Level 2 (Temporal Noise Shaping):** Implementation of TNS to control the temporal structure of quantization noise, effectively managing transients and pitch.
* **Level 3 (Psychoacoustics & Quantization):** Integration of a psychoacoustic model for masking threshold calculation, non-uniform quantization, and Huffman entropy coding.

## 🛠️ Tech Stack

* **Language:** Python 3.10+
* **Scientific Libraries:**
    * `NumPy`: Numerical computations and matrix operations.
    * `SciPy`: Signal processing (filters, FFT) and Yule-Walker solvers.
    * `Matplotlib`: Data visualization and spectral analysis.
    * `SoundFile`: Audio I/O for `.wav` files.

## 🏗️ Architecture & Features

### 1. Filterbank & Sequence Segmentation (SSC)
* **MDCT/IMDCT:** Employs Modified Discrete Cosine Transform with 50% overlap to achieve **Perfect Reconstruction** in the absence of quantization.
* **Adaptive Windowing:** Supports Sine and Kaiser-Bessel Derived (KBD) windows with dynamic switching between Long, Start, Stop, and Short sequences.
* **Transient Detection:** High-pass filtering and energy analysis to trigger short windows (ESH) to prevent pre-echo artifacts.

### 2. Temporal Noise Shaping (TNS)
* **LPC Analysis:** Uses 4th-order Linear Predictive Coding on the normalized MDCT spectrum to shape noise in the time domain.
* **Spectral Whitening:** Implements spectral normalization based on scalefactor bands for effective TNS coefficient calculation.

### 3. Psychoacoustic Model & Quantization
* **Masking Thresholds:** Models human auditory perception using spreading functions and spectral predictability to distinguish between tone and noise maskers.
* **Iterative Optimization:** A two-loop quantization strategy that adjusts Global Gain and scalefactors to keep quantization noise below the calculated masking threshold.
* **Entropy Coding:** Compliant Huffman coding with Escape sequence support for differential scalefactors up to ±60.

## 🚀 Execution Guide

### Requirements
Install dependencies using the provided requirements file:
```bash
pip install -r requirements.txt
```

### Main Levels
To execute the full processing chain for a specific level:
```bash
python src/main_level_1.py  # Run Filterbank tests
python src/main_level_2.py  # Run TNS-enabled processing
python src/main_level_3.py  # Run full AAC compression/decompression
```

### Testing & Verification
Verify individual module integrity (e.g., MDCT accuracy or TNS stability):
```bash
python tests/test_filterbank.py
python tests/test_psycho.py
python tests/test_quantizer.py
```

## 📊 Performance Results (Level 3)

The following metrics represent the final performance and compression efficiency of the AAC codec implementation:

| Metric | Result |
| :--- | :--- |
| **Compression Ratio** | 7.09:1 |
| **Compressed Bitrate** | 216.7 kbps |
| **SNR (Psychoacoustic)** | 8.66 dB (optimized for transparency) |
| **Reconstruction SNR** | 163.06 dB (Level 1 & 2 transparency) |

*The SNR of 8.66 dB in Level 3 reflects controlled quantization noise placed below the masking threshold to achieve psychoacoustic transparency.*
