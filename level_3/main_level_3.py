import os
import numpy as np
import soundfile as sf
import scipy.io as sio
import matplotlib.pyplot as plt
import time
from src.huff_utils import *
from src.quantizer import *
from src.ssc import SSC
from src.tns import tns, i_tns, load_band_tables
from src.filterbank import filter_bank, i_filter_bank
from src.psycho import psycho, spreading_function


def visualize_psycho_quant(frame_idx, X_orig, X_hat, T_b, table, ch_name):
    """
    Generates a verification plot showing: 
    Original MDCT, Reconstructed MDCT, Masking Threshold, and Quantization Error.
    """
    plt.figure(figsize=(14, 8))
    
    # Helper to convert to db safely
    def to_db(x): return 20 * np.log10(np.abs(x) + 1e-15)
    
    # Flatten everything for consistent indexing
    X_orig_f = X_orig.flatten()
    X_hat_f = X_hat.flatten()
    
    # Calculate the Quantization Error (Noise)
    error_f = X_orig_f - X_hat_f
    
    # Plotting MDCT Coefficients
    plt.plot(to_db(X_orig_f), label='Original MDCT', color='blue', lw=1, alpha=0.3)
    plt.plot(to_db(X_hat_f), label='Quantized/Dequantized MDCT', color='cyan', lw=0.8, alpha=0.5)
    
    # Plotting Quantization Error (The red line you wanted to compare)
    plt.plot(to_db(error_f), label='Quantization Error (Noise)', color='red', lw=1, alpha=0.8)
    
    # Plotting Masking Thresholds T(b)
    w_low = table[:, 1].astype(int)
    w_high = table[:, 2].astype(int)
    num_subframes = 8 if X_orig.ndim == 2 else 1
    bins_per_frame = 128 if num_subframes == 8 else 1024
    
    for s in range(num_subframes):
        offset = s * bins_per_frame
        for b in range(len(T_b) // num_subframes):
            t_idx = s * (len(T_b) // num_subframes) + b
            threshold_db = 10 * np.log10(T_b.flatten()[t_idx] + 1e-15)
            
            plt.hlines(threshold_db, offset + w_low[b], offset + w_high[b], 
                       colors='green', linestyles='-', lw=3, alpha=0.9,
                       label='Masking Threshold T(b)' if (s == 0 and b == 0) else "")

    plt.title(f"Psychoacoustic Verification - Frame {frame_idx} ({ch_name})")
    plt.xlabel("Frequency Line (k)")
    plt.ylabel("Magnitude (dB)")
    plt.ylim([-180, 60]) # Standard range for audio analysis
    plt.legend(loc='upper right', framealpha=0.9)
    plt.grid(True, which='both', linestyle=':', alpha=0.4)
    
    plt.tight_layout()
    plt.savefig(f"output/verification_frame_{frame_idx}.png", dpi=300)
    plt.close()

def plot_waveform_comparison(orig_wav, rec_wav):
    """Compares the waveforms of the original and reconstructed audio files."""
    x, fs = sf.read(orig_wav)
    y, _ = sf.read(rec_wav)
    
    if len(x.shape) > 1: x = x[:, 0]
    if len(y.shape) > 1: y = y[:, 0]
    
    min_len = min(len(x), len(y))
    x, y = x[:min_len], y[:min_len]
    error = x - y
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    ax1.plot(x, label='Original', color='blue', alpha=0.6)
    ax1.plot(y, label='Reconstructed', color='red', linestyle='--', alpha=0.5)
    ax1.set_title("Waveform Comparison (Time Domain)")
    ax1.legend(); ax1.grid(True, alpha=0.3)
    
    ax2.plot(error, label='Reconstruction Error (Noise)', color='black', lw=0.5)
    ax2.set_title("Difference Signal (Original - Reconstructed)")
    ax2.set_xlabel("Samples")
    ax2.legend(); ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("output/waveform_comparison.png", dpi=300)
    print(f"[*] [VISUALIZATION] Waveform comparison saved to output/waveform_comparison.png")


def aac_quantizer(frame_F, frame_type, SMR):
    """
    AAC quantizer for one channel: derives quantization symbols (S), scalefactor differences (sfc),
    and global gain (G) using the masking constraints from the psychoacoustic model.
    """
    bj_long, bj_short = load_band_tables("TableB219.mat")
    MQ = 8191 
    
    if frame_type == "ESH":
        bj = bj_short
        num_bands = bj.shape[0]
        Tb = compute_mdct_thresholds(frame_F, SMR, bj_short)
        
        max_vals = np.max(np.abs(frame_F), axis=0) # frame_F shape (128, 8) -> max_vals shape (8,)
        max_vals = np.maximum(max_vals, 1e-15)
        G_start = np.rint((16/3) * np.log2(max_vals**0.75 / MQ)) # Array (8,)
        
        S_total = np.zeros((8, 128))
        sfc = np.zeros((8, num_bands-1))
        
        # Find alpha values of each band for each subframe, and the global gain for each subframe
        # We use the same Tb for all subframes, as they share the same SMR and band structure.
        S_total, sfc, G = optimize_scalefactors(S_total, sfc, G_start, frame_F.T, num_bands, bj, 8, Tb)

    else: 
        bj = bj_long
        num_bands = bj.shape[0]
        Tb = compute_mdct_thresholds(frame_F, SMR, bj_long)
        
        max_val = np.max(np.abs(frame_F))
        max_val = np.maximum(max_val, 1e-15)
        G_start = np.rint((16/3) * np.log2(max_val**0.75 / MQ))
        
        f_F_2d = frame_F[None, :] 
        Tb_2d = Tb[None, :]
        G_array = np.array([G_start])
        
        S_total_2d = np.zeros((1, 1024))
        sfc_2d = np.zeros((1, num_bands-1))

        S_total_res, sfc_res, G_res = optimize_scalefactors(S_total_2d, sfc_2d, G_array, f_F_2d, num_bands, bj, 1, Tb_2d) 
        
        S_total = S_total_res[0]
        sfc = sfc_res[0]
        G = G_res[0]

    return S_total, sfc, G

def i_aac_quantizer(S, sfc, G, frame_type):
    """
    Inverse AAC quantizer.

    Reconstructs MDCT coefficients from quantized symbols (S),
    scalefactor differences (sfc), and global gain (G).

    For long frames, reconstruction is performed band-by-band.
    For ESH frames, reconstruction is applied independently
    to each of the 8 short windows.

    Returns the dequantized MDCT spectrum X_hat.
    """

    bj_long, bj_short = load_band_tables("TableB219.mat")
    
    if frame_type == "ESH":
        bj = bj_short
        num_bands = bj.shape[0]
        X_hat = np.zeros((8, 128))
        for s in range(8):
            alpha_previous_band = G[s] 
            for b in range(num_bands):
                w_low, w_high = int(bj[b, 1]), int(bj[b, 2])
                
                if b == 0:
                    alpha_b = G[s]
                else:
                    alpha_b = sfc[s, b-1] + alpha_previous_band
                
                S_slice = S[s, w_low:w_high+1]
                X_hat[s, w_low:w_high+1] = i_quantizer(S_slice, alpha_b)
                
                alpha_previous_band = alpha_b
    else:
        bj = bj_long
        num_bands = bj.shape[0]
        X_hat = np.zeros(1024)
        alpha_previous_band = G
        for b in range(num_bands):
            w_low, w_high = int(bj[b, 1]), int(bj[b, 2])
            
            if b == 0:
                alpha_b = G
            else:
                alpha_b = sfc[b-1] + alpha_previous_band

            S_slice = S[w_low:w_high+1]
            X_hat[w_low:w_high+1] = i_quantizer(S_slice, alpha_b)

            alpha_previous_band = alpha_b

    return X_hat

def aac_coder_3(filename_in, filename_aac_coded, S_long, S_short):
    """
    AAC Level 3 encoder.

    Performs the complete encoding pipeline:
    - Frame segmentation with 50% overlap
    - Window type decision (SSC)
    - Psychoacoustic analysis (SMR estimation)
    - MDCT analysis filterbank
    - Temporal Noise Shaping (TNS)
    - Quantization (global gain + scalefactors)
    - Huffman encoding

    Returns the structured bitstream (aac_seq_3) and
    stores it in MATLAB .mat format.
    """
    audio, _ = sf.read(filename_in)
    if audio.ndim == 1: audio = np.column_stack([audio, audio])
    huff_LUT_list = load_LUT("src/huffCodebooks.mat")
    
    zeros_pad = np.zeros((2048, 2))

    # PADDED AUDIO ----- >Zeros(2048), Audio, Zeros(2048),
    # to ensure we can always access t-1 and t-2 frames for the psychoacoustic model even at the edges of the signal
    audio_padded = np.vstack([zeros_pad, audio, np.zeros((2048, 2))])
    

    # 50% overlap analysis:
    # current frame starts at idx
    # previous frames are offset by hop = 1024 samples
    hop = 1024
    
    
    start_offset = 2048 # due to the padding
    num_frames = (len(audio) + 1024) // hop 
    
    aac_seq_3 = []
    prev_type = "OLS"
    WIN_TYPE = "SIN"

    for i in range(num_frames):
        idx = start_offset + i * hop
        
        curr = audio_padded[idx : idx + 2048, :]
        nxt  = audio_padded[idx + hop : idx + hop + 2048, :]

        # --- Guarantee shapes for src.ssc.SSC ---
        if curr.shape != (2048, 2):
            tmp = np.zeros((2048, 2), dtype=audio_padded.dtype)
            tmp[:curr.shape[0], :curr.shape[1]] = curr
            curr = tmp

        if nxt.shape != (2048, 2):
            tmp = np.zeros((2048, 2), dtype=audio_padded.dtype)
            tmp[:nxt.shape[0], :nxt.shape[1]] = nxt
            nxt = tmp

        f_type = SSC(curr, nxt, prev_type)


        # Psychoacoustic Model Frames, frames overlap by 50% -> t-1 and t-2 are offset by 1024 samples from the current frame
        # t-1: Starts 1024 samples BEFORE the current frame
        p1 = audio_padded[idx - 1024 : idx + 1024, :] 
        
        # t-2: Starts 2048 samples BEFORE the current frame (or 1024 before p1)
        p2 = audio_padded[idx - 2048 : idx, :]       
        
        # Dimension check (for edges)
        if p1.shape[0] != 2048: p1 = np.zeros((2048, 2))
        if p2.shape[0] != 2048: p2 = np.zeros((2048, 2))
        
        # Psychoacoustic model per channel
        SMR_l = psycho(curr[:, 0], f_type, p1[:, 0], p2[:, 0], S_long, S_short)
        SMR_r = psycho(curr[:, 1], f_type, p1[:, 1], p2[:, 1], S_long, S_short)
        
        # Filterbank (MDCT)
        f_F = filter_bank(curr, f_type, WIN_TYPE) 
        mdct_clean_l = f_F[..., 0]
        mdct_clean_r = f_F[..., 1]

        # TNS per channel
        f_F_tns_l, tns_l = tns(mdct_clean_l, f_type)
        f_F_tns_r, tns_r = tns(mdct_clean_r, f_type)

        # Quantization per channel
        S_l, sfc_l, G_l = aac_quantizer(f_F_tns_l, f_type, SMR_l)
        S_r, sfc_r, G_r = aac_quantizer(f_F_tns_r, f_type, SMR_r)

        bj_long, bj_short = load_band_tables("TableB219.mat")
        bj = bj_short if f_type == "ESH" else bj_long
        T_l = compute_mdct_thresholds(f_F_tns_l, SMR_l, bj)
        T_r = compute_mdct_thresholds(f_F_tns_r, SMR_r, bj)

        frame_data = {
            "frame_type": f_type, "win_type": WIN_TYPE,
            "chl": {"tns_coeffs": tns_l, "T": T_l, "G": G_l, "SMR": SMR_l},
            "chr": {"tns_coeffs": tns_r, "T": T_r, "G": G_r, "SMR": SMR_r}
        }

        frame_data["chl"]["orig_mdct"] = f_F[..., 0]
        frame_data["chr"]["orig_mdct"] = f_F[..., 1]

        # Huffman Encoding
        for ch_key, S_ch, sfc_ch in [("chl", S_l, sfc_l), ("chr", S_r, sfc_r)]:
            S_flat = S_ch.flatten().astype(int)
            stream_huff, cb = encode_huff(S_flat, huff_LUT_list)
            
            sfc_flat = sfc_ch.flatten().astype(int)
          
            sfc_huff = huff_LUT_code_ESC(huff_LUT_list[11], sfc_flat)
            cb_sfc = 11
            
            real_bits_per_band = []
            for b_idx in range(len(bj)):
                w_low, w_high = int(bj[b_idx, 1]), int(bj[b_idx, 2])
                if f_type == "ESH":
                    band_coeffs = S_ch[:, w_low:w_high+1].flatten().astype(int)
                else:
                    band_coeffs = S_ch[w_low:w_high+1].astype(int)
                
                if band_coeffs.size > 0 and cb > 0:
                    b_stream, _ = encode_huff(band_coeffs, huff_LUT_list, force_codebook=cb)
                    real_bits_per_band.append(len(b_stream))
                else:
                    real_bits_per_band.append(0)

            frame_data[ch_key].update({
                "stream": stream_huff, "sfc": sfc_huff,
                "codebook": cb, "sfc_codebook": cb_sfc,
                "real_bits_per_band": real_bits_per_band
            })
    
        aac_seq_3.append(frame_data)
        prev_type = f_type

    sio.savemat(filename_aac_coded, {"aac_seq_3": aac_seq_3})
    return aac_seq_3

def i_aac_coder_3(aac_seq_3, filename_out):
    """
    AAC Level 3 decoder.

    Reconstructs the time-domain signal by:
    - Huffman decoding of MDCT symbols and scalefactors
    - Inverse quantization
    - Inverse TNS filtering
    - Inverse MDCT (synthesis filterbank)
    - Overlap-add reconstruction

    Writes the decoded waveform to disk and
    returns the reconstructed signal.
    """
    huff_LUT_list = load_LUT("src/huffCodebooks.mat")
    
    bj_long, bj_short = load_band_tables("TableB219.mat")
    num_bands_long = bj_long.shape[0]  
    num_bands_short = bj_short.shape[0]

    hop = 1024
    if isinstance(aac_seq_3, np.ndarray): aac_seq_3 = aac_seq_3.flatten()
    
    num_frames = len(aac_seq_3)
    out_audio = np.zeros((num_frames * hop + 1024, 2))
    
    for i, data in enumerate(aac_seq_3):
        f_type = str(data["frame_type"]) if not isinstance(data["frame_type"], str) else data["frame_type"]
        win_type = str(data["win_type"]) if not isinstance(data["win_type"], str) else data["win_type"]
        
        f_F_recon = []
        for ch_key in ["chl", "chr"]:
            ch_data = data[ch_key]
            
            # Huffman Decoding (MDCT)
            cb_idx = int(ch_data["codebook"])
            if cb_idx == 0:
                s_decoded = np.zeros(1024) if f_type != "ESH" else np.zeros((8, 128))
            else:
                s_decoded = decode_huff(str(ch_data["stream"]), huff_LUT_list[cb_idx])

            # Huffman Decoding (Scalefactors)
            cb_sfc_idx = int(ch_data["sfc_codebook"])
            sfc_decoded = decode_huff(str(ch_data["sfc"]), huff_LUT_list[cb_sfc_idx])
            
            if f_type == "ESH":
                expected_len = 8 * (num_bands_short - 1)
                sfc_decoded = sfc_decoded[:expected_len]
                sfc_decoded = np.array(sfc_decoded).reshape((8, num_bands_short - 1)) 
                s_decoded = np.array(s_decoded).reshape((8, 128))
            else:
                expected_len = num_bands_long - 1
                sfc_decoded = sfc_decoded[:expected_len]
                sfc_decoded = np.array(sfc_decoded).reshape(num_bands_long - 1)  
                s_decoded = np.array(s_decoded).reshape(1024)
            
            # Inverse Quantization and iTNS
            f_F_q = i_aac_quantizer(s_decoded, sfc_decoded, ch_data["G"], f_type)
            f_F_ch = i_tns(f_F_q, f_type, ch_data["tns_coeffs"])
            f_F_recon.append(f_F_ch)

        # IMDCT and Synthesis Filterbank
        if f_type == "ESH":
            f_F_stereo = np.stack(f_F_recon, axis=-1)  
        else:
            f_F_stereo = np.stack(f_F_recon, axis=-1) 
            
        f_T = i_filter_bank(f_F_stereo, f_type, win_type)
        out_audio[i*hop : i*hop+2048, :] += f_T
    
    audio, _ = sf.read("LicorDeCalandraca.wav")
    original_length = audio.shape[0]

    start_idx = 0 
    
    end_idx = start_idx + original_length
    if end_idx > out_audio.shape[0]:
        final_audio = out_audio[start_idx:, :]
    else:
        final_audio = out_audio[start_idx : end_idx, :]
    sf.write(filename_out, final_audio, 48000)
    return final_audio

def demo_aac_3(filename_in, filename_out, filename_aac_coded, S_long, S_short):
    """
    End-to-end demonstration of AAC Level 3 codec.

    Executes encoding and decoding,
    computes SNR and bitrate,
    generates analysis plots,
    and prints a structured performance summary.
    """
    print("Encoding Process Started ...")
    print()
    start_enc = time.perf_counter() 

    aac_seq_3 = aac_coder_3(filename_in, filename_aac_coded, S_long, S_short)

    end_enc = time.perf_counter()    
    encoding_duration = end_enc - start_enc
    print(f"Encoding finished in: {encoding_duration:.4f} seconds")
    print("-"*55)
    print()

    print("Decoding Process Started ...")
    print()
    start_dec = time.perf_counter() 

    decoded = i_aac_coder_3(aac_seq_3, filename_out)

    end_dec = time.perf_counter()
    decoding_duration = end_dec - start_dec
    print(f"Decoding finished in: {decoding_duration:.4f} seconds")
    print("-"*55)
    print()

    original, fs = sf.read(filename_in)
    L = min(len(original), len(decoded))
    
    # SNR
    noise = original[:L] - decoded[:L]
    signal_power = np.mean(original[:L]**2)
    noise_power = np.mean(noise**2)
    snr = 10 * np.log10(signal_power / noise_power)

    
    # Bitrate Calculation
    total_bits = 0
    for frame in aac_seq_3:

        # Frame type (2 bits)
        total_bits += 2
        
        # Window type (1 bit)
        total_bits += 1
        for ch in ["chl", "chr"]:
            # Global gain (8 bits each)
            total_bits += 8 * np.atleast_1d(frame[ch]["G"]).size

            # Codebook indices (4 + 4 bits) - one for MDCT stream and one for scalefactors
            # stream codebook + sfc codebook
            total_bits += 4 + 4  

            # TNS coefficients (4 bits each, quantized)
            tns_data = frame[ch]["tns_coeffs"]
            if isinstance(tns_data, list):
                for coeffs in tns_data:
                    total_bits += coeffs.size * 4
            else:
                total_bits += tns_data.size * 4

            # Huffman streams
            total_bits += len(frame[ch]["stream"])
            total_bits += len(frame[ch]["sfc"])
            
    # Total duration in seconds
    duration = L / fs
    
    bitrate = total_bits / duration
    
    # Uncompressed bitrate: 48000 * 16 * 2 = 1536000 bps
    uncompressed_bitrate = fs * 16 * 2 
    compression = uncompressed_bitrate / bitrate
    
    print("\n" + "="*55)
    print(" " * 18 + "AAC LEVEL 3 SUMMARY")
    print("="*55)

    print(f"{'Original Bits:':<28}{int(L * 16 * 2):>15,} bits")
    print(f"{'Compressed Bits:':<28}{total_bits:>15,} bits")

    print("-"*55)

    print(f"{'Original Bitrate:':<28}{uncompressed_bitrate/1000:>15.1f} kbps")
    print(f"{'Compressed Bitrate:':<28}{bitrate/1000:>15.1f} kbps")
    print(f"{'Compression Ratio:':<28}{compression:>15.2f} : 1")

    print("-"*55)

    print(f"{'Signal-to-Noise Ratio:':<28}{snr:>15.2f} dB")
    print(f"{'Noise Mean:':<28}{np.mean(noise):>15.8f}")
    print(f"{'Noise Variance:':<28}{np.var(noise):>15.6f}")

    print("="*55 + "\n")

    summary_text = (
        "\n" + "="*55 + "\n"
        + " " * 18 + "AAC LEVEL 3 SUMMARY\n"
        + "="*55 + "\n"
        + f"{'Original Bits:':<28}{int(L * 16 * 2):>15,} bits\n"
        + f"{'Compressed Bits:':<28}{total_bits:>15,} bits\n"
        + "-"*55 + "\n"
        + f"{'Original Bitrate:':<28}{uncompressed_bitrate/1000:>15.1f} kbps\n"
        + f"{'Compressed Bitrate:':<28}{bitrate/1000:>15.1f} kbps\n"
        + f"{'Compression Ratio:':<28}{compression:>15.2f} : 1\n"
        + "-"*55 + "\n"
        + f"{'Signal-to-Noise Ratio:':<28}{snr:>15.2f} dB\n"
        + f"{'Noise Mean:':<28}{np.mean(noise):>15.8f}\n"
        + f"{'Noise Variance:':<28}{np.var(noise):>15.6f}\n"
        + "="*55 + "\n"
    )

    # Save to file
    with open("output/aac_level3_summary.txt", "w") as f:
        f.write(summary_text)


    # Plot threshold verification for a specific frame
    idx = 15 if len(aac_seq_3) > 15 else len(aac_seq_3) // 2
    frame = aac_seq_3[idx]
    f_type = frame["frame_type"]

    bj_long, bj_short = load_band_tables("TableB219.mat")
    bj = bj_short if f_type == "ESH" else bj_long

    f_F_orig = frame["chl"]["orig_mdct"]

    huff_LUT = load_LUT("src/huffCodebooks.mat")
    s_q = decode_huff(str(frame["chl"]["stream"]), huff_LUT[frame["chl"]["codebook"]])
    sfc_q = decode_huff(str(frame["chl"]["sfc"]), huff_LUT[frame["chl"]["sfc_codebook"]])

    if f_type == "ESH":
        s_q = np.array(s_q).reshape((8, 128))
        sfc_q = np.array(sfc_q).reshape((8, 41))

    f_F_q = i_aac_quantizer(s_q, sfc_q, frame["chl"]["G"], f_type)
    
    # Plot bits vs SMR for the same frame
    frame = aac_seq_3[idx]
    bj = bj_short if frame["frame_type"] == "ESH" else bj_long
    
    # UPDATED VISUALIZATION CALLS
    visualize_psycho_quant(idx, f_F_orig, f_F_q, frame["chl"]["T"], bj, "Left Channel")
    plot_waveform_comparison(filename_in, filename_out)

    print(f"Generating bit allocation analysis for Frame {idx}...")

    return snr, bitrate, compression

if __name__ == "__main__":
    os.makedirs("output", exist_ok=True)
    bj_long, bj_short = load_band_tables("TableB219.mat")

    S_long = np.zeros((bj_long.shape[0], bj_long.shape[0]))
    for i in range(bj_long.shape[0]):
        for j in range(bj_long.shape[0]):
            S_long[i, j] = spreading_function(i, j, bj_long)

    S_short = np.zeros((bj_short.shape[0], bj_short.shape[0]))
    for i in range(bj_short.shape[0]):
        for j in range(bj_short.shape[0]):
            S_short[i, j] = spreading_function(i, j, bj_short)

    path_in = "LicorDeCalandraca.wav"
    path_out = "./output/out_level_3.wav"
    path_aac_coded = "./output/aac_coded_level3.mat"
    
    demo_aac_3(path_in, path_out, path_aac_coded, S_long, S_short)




