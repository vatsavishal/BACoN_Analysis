##working on real data instead of simulated one
## modified the code to use derivative of data instead of using raw data.
## all operations are performed on derivative of data which is then integrated back to original pulse height
## generates a deconv_peakHeights..txt which contains [rowNo][indices][heights] in the "deconvolved peaks" NOT the original raw wf peaks

#this code takes input isolated_SPE_wfAvg-ch.txt for its template signal

############## going to incorporate analysing all channels simultaneously

### comment / uncomment i want to work with derivative or not 


### Modified on 13th June to include area around peaks too to see a core lation betweene height and area 
### correction to the code on 19thAug' Added an area calc in deconv wf too so that only take peaks which have a certain peak height AND area ( will help in removing erreneous spikes that arise from FFT-IFFT)

### Modified on 25th Sept, re did conv/deconv with linear rfft instead of circular FFT , made renormalizing easier and better

#### Modified on Oct7th for matched filter to better get amplitude of peaks(maintains linearity in decnved peaks) and corrected for time offset 
### Modified on Oct 13th, get better estimate of number of bnumber of peaks by getting FWHM of template Noise-based thresholds (sigma via MAD) adapt per file/event conditions without overfitting to the biggest pulse.
#### Width constraints informed by template FWHM prevent spurious spiky detections while letting real close peaks through
#### see line 136-137

### Modified on Oct22 for NO deconv on PMT, ch12 doesnt get smoothing and deconv, just basic peak finding.

# Unified processing:
# - ch 0..11 : Wiener deconvolution + matched-filter detection, heights from deconv_wf
# - ch 12    : NO deconvolution, NO smoothing, simple threshold-based peak finding

## Modified Oct 31 to include plateau also, and to split plateau (flat top peaks) in corresponding 2-3 peaks
###Modified Nov2 to include peaks that MF missed but deconv detected.



import uproot
import numpy as np
import time
import gi
gi.require_version("Gtk", "3.0")

from scipy import stats as st
from scipy.signal import wiener, savgol_filter, find_peaks
from scipy.signal.windows import tukey

import matplotlib.pyplot as plt

plt.rcParams.update({
    'font.size': 28,
    'axes.titlesize': 28,
    'axes.labelsize': 28,
    'xtick.labelsize': 28,
    'ytick.labelsize': 28,
    'legend.fontsize': 28,
})

# ---------- run config ----------
data_path  = '/mnt/Data2/BaconRun5Data/rootData'
date_data  = '11_07_2025'

start_file = 150
max_no_of_files = 200            # processes files [start_file, max_no_of_files)
start_channel = 0              # iterate channels 0..12
end_channel   = 13

# output timestamp shift (samples) for deconvolved channels
offset = 290

# analysis window (indices)
analysis_lo = 100
analysis_hi = 7000

# PMT (channel 12) threshold (ADC units after inversion & baseline subtract)
VTH_PMT = 60.0
PMT_EVENT_ELIM_THRESHOLD = None  # e.g. 16000 to skip pathological events; None = disabled for cosmics

# ---------- helpers ----------
def compute_baseline(wf, mode=True, wf_range_bsl=(0, 500)):
    if mode:
        return st.mode(wf[wf_range_bsl[0]:wf_range_bsl[1]], keepdims=False).mode.astype(np.float32)
    return np.float32(np.mean(wf[wf_range_bsl[0]:wf_range_bsl[1]]))

def subtract_baseline(wfs, mode=True, wf_range_bsl=(0, 500), mean_bsl=False):
    if wfs.ndim == 1:
        b = compute_baseline(wfs, mode=mode, wf_range_bsl=wf_range_bsl)
    else:
        if mean_bsl:
            b = np.float32(np.mean([compute_baseline(w, mode=mode, wf_range_bsl=wf_range_bsl) for w in wfs]))
        else:
            b = np.array([compute_baseline(w, mode=mode, wf_range_bsl=wf_range_bsl)
                          for w in wfs], dtype=np.float32).reshape(-1, 1)
    return wfs - b

def template_fwhm_samples(template):            ###necessary if using width too in find_peaks 
    t = np.asarray(template, dtype=np.float32)
    if t.size == 0:
        return 10
    tmax = float(np.max(t))
    if tmax <= 0.0:
        return 10
    hm = 0.5 * tmax
    idx = np.where(t >= hm)[0]
    if idx.size == 0:
        return 10
    return int(idx[-1] - idx[0] + 1)

# ---------- main ----------
start_time = time.perf_counter()

channel = start_channel
while channel < end_channel:

    print("current channel in analysis:", channel)

    # ------ PMT branch: channel 12 (no deconvolution, no smoothing) ------
    if channel == 12:
        outfile = f'peaks_ch12-{start_file}-{max_no_of_files}files-{date_data}.txt'
        for file_num in range(start_file, max_no_of_files):
            filename = f"{data_path}/run-{date_data}-file_{file_num}.root"
            print("Analyzing (PMT):", filename)

            infile  = uproot.open(filename)
            RawTree = infile['RawTree']

            # load traces (float32)
            adc_ch = np.array(RawTree[f'chan{channel}/rdigi'].array(), dtype=np.float32)

            # invert PMT: value_to_subtract - adc
            value_to_subtract = np.float32((1 << 14) - 1)  # 16383
            adc_ch = value_to_subtract - adc_ch

            # baseline subtract (per waveform)
            wfs = subtract_baseline(adc_ch, mode=True, wf_range_bsl=(0, 500), mean_bsl=False)

            out_buf = []
            BUF_FLUSH_N = 1000

            with open(outfile, 'a') as f:
                for i in range(wfs.shape[0]):
                    wf = wfs[i]

                    if PMT_EVENT_ELIM_THRESHOLD is not None and np.any(np.abs(wf) > PMT_EVENT_ELIM_THRESHOLD):
                        continue

                    # simple threshold peak finding (no distance, no prominence)
                    peaks, props = find_peaks(wf, height=VTH_PMT)
                    if peaks.size == 0:
                        continue

                    # enforce analysis window
                    keep = (peaks > analysis_lo) & (peaks < analysis_hi)
                    if not np.any(keep):
                        continue
                    peaks = peaks[keep]
                    heights = props["peak_heights"][keep]

                    # write
                    str_idx = ','.join(map(str, peaks.tolist()))
                    str_hgt = ','.join(map(lambda v: f"{v:.6g}", heights.tolist()))
                    out_buf.append(f"[{i}]\t[{str_idx}]\t[{str_hgt}]\n")

                    if len(out_buf) >= BUF_FLUSH_N:
                        f.writelines(out_buf); out_buf.clear()

                if out_buf:
                    f.writelines(out_buf); out_buf.clear()

        print(f"data written to {outfile}")
        channel += 1
        continue  # move to next channel

    # ------ Deconvolution branch: channels 0..11 ------
    if channel == 11:
        infile_template = f'isolated_SPE_wfsAvg10_06_2025-11.txt'
    elif channel == 10:
        infile_template = f'isolated_SPE_wfsAvg10_06_2025-10.txt'
    elif channel == 9:
        infile_template = f'isolated_SPE_wfsAvg10_06_2025-9.txt'
    else:
        infile_template = f'isolated_SPE_wfsAvg-nontriggerSiPM.txt'

    # load & prep template (float32)
    isolated_spe_wf_averaged = np.loadtxt(infile_template).astype(np.float32)

    # smooth template for SiPMs; (we're not in ch12 branch)
    template_clean = savgol_filter(isolated_spe_wf_averaged, 65, 5).astype(np.float32)
    template_time = template_clean
    n = 7500
    m = len(template_time)
    t_peak = int(np.argmax(template_time))

    # rFFT size for linear convolution (next pow2 >= n+m-1)
    N = 1 << (n + m - 1 - 1).bit_length()

    # template FFTs
    T = np.fft.rfft(np.pad(template_time, (0, N - m)).astype(np.float32))
    H = T.copy()

    # Wiener inverse (fixed; no per-event scaling)
    lam = np.float32(1e-2 * np.max(np.abs(T)**2))  # tune 1e-3..3e-2
    Hinv = np.conj(T) / (np.abs(T)**2 + lam)

    # matched-filter normalization (time-domain energy)
    E = np.float32(np.sum(template_time**2) + 1e-12)

    # linear-conv crop; compensate internal template peak (removes constant lag)
    start = (m - 1) - t_peak
    end   = start + n

    # taper & pad buffer (reused per event)
    tukey_alpha = 0.1
    tukey_win = tukey(n, tukey_alpha).astype(np.float32)
    pad_buf = np.zeros(N, dtype=np.float32)

    # template FWHM (kept for reference / if you re-enable width gates later)
    fwhm = template_fwhm_samples(template_time)
    # min_width = max(3, int(0.3 * fwhm))
    # wlen = int(3 * fwhm) if fwhm > 0 else 31

    # output file for this channel
    outfile = f'deconv_peakHeights-{start_file}-{max_no_of_files}files-{date_data}-{channel}.txt'

    file_num = start_file
    while file_num < max_no_of_files:

        filename = f"{data_path}/run-{date_data}-file_{file_num}.root"
        print("Analyzing (SiPM):", filename)

        infile  = uproot.open(filename)
        RawTree = infile['RawTree']

        # load traces (float32)
        adc_ch = np.array(RawTree[f'chan{channel}/rdigi'].array(), dtype=np.float32)

        # invert for triggered SiPMs (9,10,11) to be positive
        if channel in (9, 10, 11):
            value_to_subtract = np.float32((1 << 14) - 1)
            adc_ch = value_to_subtract - adc_ch

        # baseline subtract (per waveform)
        subt_wfs = subtract_baseline(adc_ch, mode=True, wf_range_bsl=(0, 500), mean_bsl=False)

        separation = 5  # samples

        out_buf = []
        BUF_FLUSH_N = 1000

        with open(outfile, 'a') as f:
            for i in range(subt_wfs.shape[0]):
                obs_signal = subt_wfs[i]

                # light smoothing of observed signal (as in your pipeline)
                obs_signal_clean = savgol_filter(obs_signal, 50, 3).astype(np.float32)
                dy_dt = obs_signal_clean

                # taper+pad without allocations
                pad_buf.fill(0.0)
                np.multiply(dy_dt[:n], tukey_win, out=pad_buf[:n])

                # rFFT
                X = np.fft.rfft(pad_buf)

                # deconvolution (heights)
                S_hat_fft = X * Hinv
                s_hat_full = np.fft.irfft(S_hat_fft, n=N)
                deconv_wf = s_hat_full[start:end].astype(np.float32)

                # keep both filters 
                deconv_wf = wiener(deconv_wf, mysize=50, noise=None).astype(np.float32)
                deconv_wf = savgol_filter(deconv_wf, 50, 3).astype(np.float32)

                # matched filter for detection/timing
                mf_full = np.fft.irfft(X * np.conj(H), n=N) / E
                mf = mf_full[start:end].astype(np.float32)

                # robust sigma from MF (MAD via percentiles)
                p50 = np.percentile(mf, 50)
                mad = np.percentile(np.abs(mf - p50), 50) + 1e-12
                sigma = np.float32(1.4826 * mad)

                # a bit looser since we also have fallback path
                height_thr = np.float32(5.5) * sigma

                # ---- 1) MF detection with plateau awareness ----
                mf_peaks, mf_props = find_peaks(
                    mf,
                    height=height_thr,
                    # prominence=(prom_thr, None), 
                    # distance=separation, 
                    # width=(min_width, None), 
                    # wlen=local_wlen, 
                    # rel_height=0.5
                    plateau_size=(1, None)
                )

                # keep inside window (no pre trigger signals)
                mf_peaks = np.array([p for p in mf_peaks if (analysis_lo < p < analysis_hi)], dtype=int)

                # plateau splitting
                final_from_mf = []
                if mf_peaks.size > 0 and "plateau_sizes" in mf_props:
                    plateau_sizes = mf_props["plateau_sizes"]
                    left_edges    = mf_props["left_edges"]
                    right_edges   = mf_props["right_edges"]

                    PLATEAU_SPLIT_MIN = 8   # ≥4 samples → try split
                    EXPECTED_DELTA    = 3   # typical deconv delta width
                    MAX_SUBPEAKS      = 3

                    for j, pk in enumerate(mf_peaks):
                        if plateau_sizes[j] <= 1:
                            final_from_mf.append(int(pk))
                            continue

                        size_j = int(plateau_sizes[j])
                        L = int(left_edges[j]); R = int(right_edges[j])

                        if size_j < PLATEAU_SPLIT_MIN:
                            mid = (L + R) // 2
                            if analysis_lo < mid < analysis_hi:
                                final_from_mf.append(mid)
                            continue

                        width = R - L + 1
                        est_n = int(round(width / float(EXPECTED_DELTA)))
                        est_n = max(2, min(est_n, MAX_SUBPEAKS))
                        step = width / float(est_n)

                        for k in range(est_n):
                            pos = int(round(L + (k + 0.5) * step))
                            pos = max(L, min(R, pos))
                            if analysis_lo < pos < analysis_hi:
                                final_from_mf.append(pos)
                else:
                    final_from_mf = mf_peaks.tolist()

                # dedupe+sort MF set
                final_from_mf = sorted(set(final_from_mf))

                # ---- 2) Fallback detection on deconv_wf (height-only... some peaks MF misses but deconv gets it) ----
                p50_dec = np.percentile(deconv_wf, 50)
                mad_dec = np.percentile(np.abs(deconv_wf - p50_dec), 50) + 1e-12
                sigma_dec = np.float32(1.4826 * mad_dec)
                dec_height_thr = np.float32(5.0) * sigma_dec   # slightly looser than MF

                dec_peaks, _ = find_peaks(deconv_wf, height=dec_height_thr)
                dec_peaks = np.array([p for p in dec_peaks if (analysis_lo < p < analysis_hi)], dtype=int)

                # ---- 3) Merge: keep deconv-only peaks not too close to MF peaks ----
                MERGE_DIST = 6  # samples time
                final_peaks = list(final_from_mf)

                for p in dec_peaks:
                    if any(abs(p - q) <= MERGE_DIST for q in final_from_mf):
                        continue
                    final_peaks.append(int(p))

                # final clean-up
                final_peaks = sorted(set(final_peaks))
                if not final_peaks:
                    continue
                final_peaks = np.array(final_peaks, dtype=int)

                # heights from deconvolved waveform at final locations
                peak_heights_all = deconv_wf[final_peaks]

                # timestamp shift for deconvolved channels — **use final_peaks**
                corrected_peaks = (final_peaks + offset).astype(np.int32)

                # write buffered
                str1 = ','.join(map(str, corrected_peaks.tolist()))
                str2 = ','.join(map(lambda v: f"{v:.6g}", peak_heights_all.tolist()))
                out_buf.append(f"[{i}]\t[{str1}]\t[{str2}]\n")

                #### Optional plotting for a couple of events
                # if i == 28 or i == 29:
                #     x = np.arange(len(deconv_wf))
                #     x_shifted = x + offset
                #     plt.figure(figsize=(10,6))
                #     plt.plot(obs_signal, label=f"Observed (baseline-sub), channel: {channel} event:{i}")
                #     plt.plot(x_shifted, mf, label="Matched-filter")
                #     plt.plot(x_shifted, deconv_wf, label="deconv_wf")
                #     plt.plot(corrected_peaks, deconv_wf[final_peaks], 'x', label="Peaks")
                #     plt.legend(); plt.yscale("log"); plt.grid(True); plt.show()

                if len(out_buf) >= BUF_FLUSH_N:
                    f.writelines(out_buf)
                    out_buf.clear()

            if out_buf:
                f.writelines(out_buf)
                out_buf.clear()

        file_num += 1

    print(f"data written to {outfile}")
    channel += 1

end_time = time.perf_counter()
print(f"Elapsed time: {(end_time-start_time):.2f} seconds")
