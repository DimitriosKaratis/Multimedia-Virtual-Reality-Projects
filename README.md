# MPEG-4 AAC Audio Codec Implementation

An implementation of a high-fidelity audio compression system based on the **MPEG-4 AAC (Advanced Audio Coding)** international standard. This project was developed as part of the **Multimedia Systems** course (2025-2026) at the Aristotle University of Thessaloniki.

## 🤖 Project Overview

The system is a lossy audio codec designed to achieve significant data reduction while maintaining high subjective audio quality by utilizing psychoacoustic modeling. The implementation follows a modular three-level architecture:

* [cite_start]**Level 1 (Filterbank & SSC):** Time-to-frequency transformation using MDCT and transient detection via Sequence Segmentation Control[cite: 16, 44].
* [cite_start]**Level 2 (Temporal Noise Shaping):** Implementation of TNS to control the temporal structure of quantization noise, effectively managing transients and pitch[cite: 17, 181].
* [cite_start]**Level 3 (Psychoacoustics & Quantization):** Integration of a psychoacoustic model for masking threshold calculation, non-uniform quantization, and Huffman entropy coding[cite: 18, 306].

## 🛠️ Tech Stack

* [cite_start]**Language:** Python 3.10+ [cite: 24]
* [cite_start]**Scientific Libraries:** * `NumPy`: Numerical computations and matrix operations[cite: 24, 36].
    * [cite_start]`SciPy`: Signal processing (filters, FFT) and Yule-Walker solvers[cite: 24, 36, 190].
    * [cite_start]`Matplotlib`: Data visualization and spectral analysis[cite: 36].
    * [cite_start]`SoundFile`: Audio I/O for `.wav` files[cite: 36, 926].

## 🏗️ Architecture & Features

### 1. Filterbank & Sequence Segmentation (SSC)
* [cite_start]**MDCT/IMDCT:** Employs Modified Discrete Cosine Transform with 50% overlap to achieve **Perfect Reconstruction** in the absence of quantization[cite: 21, 53, 107].
* [cite_start]**Adaptive Windowing:** Supports Sine and Kaiser-Bessel Derived (KBD) windows with dynamic switching between Long, Start, Stop, and Short sequences[cite: 54, 56, 121].
* [cite_start]**Transient Detection:** High-pass filtering and energy analysis to trigger short windows (ESH) to prevent pre-echo artifacts[cite: 47, 62].

### 2. Temporal Noise Shaping (TNS)
* [cite_start]**LPC Analysis:** Uses 4th-order Linear Predictive Coding on the normalized MDCT spectrum to shape noise in the time domain[cite: 188, 190].
* [cite_start]**Spectral Whitening:** Implements spectral normalization based on scalefactor bands for effective TNS coefficient calculation[cite: 186, 300].

### 3. Psychoacoustic Model & Quantization
* [cite_start]**Masking Thresholds:** Models human auditory perception using spreading functions and spectral predictability to distinguish between tone and noise maskers[cite: 311, 312, 316].
* [cite_start]**Iterative Optimization:** A two-loop quantization strategy that adjusts Global Gain and scalefactors to keep quantization noise below the calculated masking threshold[cite: 318, 319].
* [cite_start]**Entropy Coding:** Compliant Huffman coding with Escape sequence support for differential scalefactors up to ±60[cite: 404, 405].

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
| **Compression Ratio** | [cite_start]7.09:1 [cite: 416, 474] |
| **Compressed Bitrate** | [cite_start]216.7 kbps [cite: 416, 476] |
| **SNR (Psychoacoustic)** | [cite_start]8.66 dB (optimized for transparency) [cite: 416, 463] |
| **Reconstruction SNR** | [cite_start]163.06 dB (Level 1 & 2 transparency) [cite: 171, 281, 302] |

[cite_start]*The SNR of 8.66 dB in Level 3 reflects controlled quantization noise placed below the masking threshold to achieve psychoacoustic transparency[cite: 463, 483].*
