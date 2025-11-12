############# read the generated txt file (deconv_peakHeights...txt) and make a histogram of pulse heights and timing from it
#### from the peak height will decipher SPE/DPE/TPE     it is heights in deconvolved data not original raw data



# 1. Cuts: eliminate events with >8 peaks in any of ch9/10/11
# 2. Cuts: eliminate events if any of ch9/10/11 miss all peaks
# 2. Use triangluar cuts on trigger SiPM, make sure each trigger SiPM sees at least 20% of total light and at most 60% of total light seen in all trigger SiPM combined
# 3. Proper PE height calibration for each channel
# 4. PE timing histogram construction using calibrated 1PE range
# 5. Summation of PE values across trigger channels (9,10,11)

## modified on 14thJune to include area also around each peak (area is 50 samples before peak and 250 samples after peak)

## modified on Sept 26th, now with the new deconvolution, the deconved peaks match the actual peak heights.

############# read the generated txt file (deconv_peakHeights...txt) and make a histogram of pulse heights and timing from it
#### from the peak height will decipher SPE/DPE/TPE     it is heights in deconvolved data not original raw data

# 1. Cuts: eliminate events with >8 peaks in any of ch9/10/11
# 2. Cuts: eliminate events if any of ch9/10/11 miss all peaks
# 2. Use triangular cuts on trigger SiPM: each must see 20–60% of total PE (NOT peak count)
# 3. Proper PE height calibration for each channel (continuous)
# 4. PE timing histogram construction using calibrated 1PE range
# 5. Summation of PE values across trigger channels (9,10,11)

## modified on 14thJune to include area also around each peak (area is 50 samples before peak and 250 samples after peak)
## modified on Sept 26th, now with the new deconvolution, the deconved peaks match the actual peak heights.
## modified on Oct 15th: replaced polynomial calibration with continuous window-anchored PCHIP mapping (0 ADC -> 0 PE)
## modified on Oct 15th: triangular cut now uses TOTAL PE per channel (continuous-calibrated), not peak counts.

### made triangle/ternary cut optional

############# read the generated txt file (deconv_peakHeights...txt) and make a histogram of pulse heights and timing from it
#### from the peak height will decipher SPE/DPE/TPE     it is heights in deconvolved data not original raw data

# 1. Cuts: eliminate events with >8 peaks in any of ch9/10/11
# 2. Cuts: eliminate events if any of ch9/10/11 miss all peaks
# 2. (Optional) triangular cuts on trigger SiPM: each must see 20–60% of total PE (NOT peak count)
# 3. Proper PE height calibration for each channel (continuous)
# 4. PE timing histogram construction using calibrated 1PE range
# 5. Summation of PE values across trigger channels (9,10,11)

## modified on 14thJune: include area around each peak (50 pre, 250 post samples)
## modified on Sept 26th: new deconvolution, deconved peaks match actual heights
## modified on Oct 15th: continuous window-anchored PCHIP mapping (0 ADC -> 0 PE)
## modified on Oct 16th: event-level triangle-cut (applies to all channels)
## modified on Oct 16th: ROOT outputs use doubles + TH1D with explicit binning
## modified on Oct 16th: APPLY_TRIANGLE_CUT flag to toggle triangular cut

import sys
import glob
import uproot
import numpy as np
import ast, re
import time
import math

import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import matplotlib.colors as colors
import itertools
import ROOT
import array

from scipy.interpolate import PchipInterpolator  # continuous monotone calibration

import gi
gi.require_version("Gtk", "3.0")

# =========================
# TOGGLE: triangular cut
APPLY_TRIANGLE_CUT = True   # set to False to disable the 20–61% PE-fraction cut
# =========================

### plotting parameters:
plt.rcParams.update({
    'font.size': 28,
    'axes.titlesize': 28,
    'axes.labelsize': 28,
    'xtick.labelsize': 28,
    'ytick.labelsize': 28,
    'legend.fontsize': 28,
})

start_time = time.perf_counter()

start_files = 150
max_no_files = 200
run_date = '11_07_2025'

# file structure params
ROWS_PER_FILE = 21555
NUM_FILES = max_no_files - start_files

NOISE_THRES = 0.021 ### some noisy peaks creep in

# Colors
color_cycle = itertools.cycle(plt.cm.tab20.colors)

# Result containers
summed_height_histo = []
summed_height_histo_calibrated = []
summed_timing_SPE_histo = []
trigger_pe_sum = []
trigger_raw_sum = []        ##uncalibrated trigger sum storage

all_height_histo_per_channel = []
all_timing_histo_per_channel = []
all_height_histo_per_channel_calibrated = []

# ======== ADC windows per channel for PE anchors (edit ranges as needed) ========
calibration_windows = {
    9:  {"SPE": (0.012, 0.0532), "DPE": (0.0533, 0.0978), "TPE": (0.0979, 0.1455)},
    10: {"SPE": (0.02, 0.04),    "DPE": (0.0521, 0.080),  "TPE": (0.0881, 0.1246)},
    11: {"SPE": (0.02, 0.05),    "DPE": (0.0531, 0.0985), "TPE": (0.0986, 0.1347)},
    # other channels: leave empty dict (identity mapping 0+)
}

def _centers_sorted(win_dict):
    labels = ["SPE","DPE","TPE","QPE","5PE","6PE","7PE","8PE"]
    xs, ys = [], []
    for i, lab in enumerate(labels, start=1):
        if lab in win_dict:
            lo, hi = win_dict[lab]
            xs.append(0.5*(lo+hi))
            ys.append(float(i))
    if not xs:
        return np.array([], float), np.array([], float)
    order = np.argsort(xs)
    return np.array(xs, float)[order], np.array(ys, float)[order]

def build_continuous_calibrator_for_channel(win_dict, use_pchip=True):
    xc, yc = _centers_sorted(win_dict)
    if xc.size == 0:
        return lambda h: max(0.0, float(h))
    xs = np.concatenate([[0.0], xc])
    ys = np.concatenate([[0.0], yc])
    eps = 1e-12
    for i in range(1, len(xs)):
        if xs[i] <= xs[i-1]:
            xs[i] = xs[i-1] + eps
    if use_pchip and len(xs) >= 3:
        interp = PchipInterpolator(xs, ys, extrapolate=True)
        def f(h):
            v = float(interp(h))
            return 0.0 if v < 0 else v
        return f
    else:
        def f(h):
            h = float(h)
            if h <= xs[0]:
                m = (ys[1]-ys[0])/(xs[1]-xs[0]) if len(xs) > 1 else 1.0
                return max(0.0, ys[0] + m*(h - xs[0]))
            if h >= xs[-1]:
                m = (ys[-1]-ys[-2])/(xs[-1]-xs[-2]) if len(xs) > 1 else 1.0
                return max(0.0, ys[-1] + m*(h - xs[-1]))
            return float(np.interp(h, xs, ys))
        return f

# Build per-channel ADC->PE calibrators
adc_to_pe_fn = {}
for ch in range(12):
    win = calibration_windows.get(ch, {})
    adc_to_pe_fn[ch] = build_continuous_calibrator_for_channel(win, use_pchip=True)

# ---- Global counters (once per event) ----
total_events_iterated = 0
skipped_missing_triggers = 0
skipped_gt45_peaks = 0
skipped_zero_total_pe = 0
skipped_channel_zero_pe = 0
rejected_tri_cut = 0
accepted_tri_cut = 0

# --------------- Load per-channel peak lists from txt (with GLOBAL event ids) #### has a noise cut off too, remove it if see fit, ---------------
channel_data = {}
first_filename = None
for ch in range(12):
    filename = f'deconv_peakHeights-{start_files}-{max_no_files}files-{run_date}-{ch}.txt'
    if first_filename is None:
        first_filename = filename
    with open(filename, 'r') as f:
        lines = f.readlines()
    channel_data[ch] = []
    for i, line in enumerate(lines):
        local_row = int(line.split('\t')[0].strip('[]'))
        global_evt_id = i  # unique across concatenated files
        times   = ast.literal_eval(line.split('\t')[1].strip())
        heights = ast.literal_eval(re.sub(r'\]+$', ']', line.split('\t')[2].strip()))
        # --- Noise filter: remove small peaks (< 0.028) ---
        filtered_pairs = [(t, h) for t, h in zip(times, heights) if h >= NOISE_THRES]
        if not filtered_pairs:
            continue  # skip if no valid peaks remain after filtering

        times, heights = zip(*filtered_pairs)
        channel_data[ch].append((global_evt_id, local_row, list(times), list(heights)))
    print(f"Loaded {len(channel_data[ch])} lines for ch {ch} from {filename}")

# Transpose to group by GLOBAL event id
all_events = {}
for ch in range(12):
    for global_evt_id, local_row, times, heights in channel_data[ch]:
        if global_evt_id not in all_events:
            all_events[global_evt_id] = {}
        all_events[global_evt_id][ch] = (times, heights)

print("Unique global events seen:", len(all_events))

## ========== PRECOMPUTE EVENT-LEVEL ACCEPT/REJECT USING TRIGGER CHANNELS ==========
tern_f9, tern_f10, tern_f11, tern_accept = [], [], [], []
event_accept = {}   # global_evt_id -> bool
event_total_pe = {} # global_evt_id -> total PE across 9/10/11 (for trigger_pe_sum)
event_total_raw = {}

for evt in sorted(all_events.keys()):
    total_events_iterated += 1

    evt_data = all_events[evt]

    # Missing triggers?
    if not all(tc in evt_data for tc in [9, 10, 11]):
        skipped_missing_triggers += 1
        event_accept[evt] = False
        continue

    # # >45 peaks sanity cap?
    # if any(len(evt_data[tc][1]) > 45 for tc in [9, 10, 11]):
    #     skipped_gt45_peaks += 1
    #     event_accept[evt] = False
    #     continue

    # PE sums per trigger channel (continuous calibrated)
    pe_sums = {}
    for tc in [9, 10, 11]:
        fn = adc_to_pe_fn[tc]
        pe_sums[tc] = sum(fn(h) for h in evt_data[tc][1])

    total_pe = sum(pe_sums.values())

    total_raw = 0.0
    for tc in [9, 10, 11]:
        total_raw += sum(evt_data[tc][1])

    if total_pe <= 0:
        skipped_zero_total_pe += 1
        event_accept[evt] = False
        continue

    if any(pe_sums[tc] <= 0 for tc in [9, 10, 11]):
        skipped_channel_zero_pe += 1
        event_accept[evt] = False
        continue

    # Fractions
    f9  = pe_sums[9]  / total_pe
    f10 = pe_sums[10] / total_pe
    f11 = pe_sums[11] / total_pe

    # Decide acceptance (toggleable)
    if APPLY_TRIANGLE_CUT:
        accept = (0.2 <= f9 <= 0.61) and (0.2 <= f10 <= 0.61) and (0.2 <= f11 <= 0.61)
    else:
        accept = True  # bypass the triangular requirement

    # record for ternary plot either way
    tern_f9.append(f9); tern_f10.append(f10); tern_f11.append(f11); tern_accept.append(accept)
    event_accept[evt] = accept
    event_total_pe[evt] = total_pe
    event_total_raw[evt] = total_raw 

    if APPLY_TRIANGLE_CUT:
        if accept: accepted_tri_cut += 1
        else:      rejected_tri_cut += 1

# ========= Helpers for safer ROOT hist binning (avoid 1–2-bin collapse) =========
def _make_range(vals, pad_frac=0.02, min_span=1e-7):
    if not vals:
        return (0.0, 1.0)
    lo = float(min(vals)); hi = float(max(vals))
    if hi - lo < min_span:
        mid = 0.5*(hi+lo)
        span = max(min_span, abs(mid)*pad_frac + min_span)
        return mid - span, mid + span
    span = hi - lo
    pad = span * pad_frac
    return lo - pad, hi + pad

def _nbins_for(vals, target_bins=1000, max_bins=4000, min_bins=100):
    if not vals:
        return 100
    return max(min_bins, min(max_bins, int(target_bins)))

## ===================== Process all events for ALL CHANNELS =====================
for ch in range(12):
    raw_heights = []
    calibrated_heights = []
    timing_spe = []

    adc_to_pe = adc_to_pe_fn[ch]

    for evt in sorted(all_events.keys()):
        # Skip whole event for ALL channels if it failed the triangle cut (only when enabled)
        if APPLY_TRIANGLE_CUT and not event_accept.get(evt, False):
            continue

        # Process this event for this channel
        if ch in all_events[evt]:
            times, heights = all_events[evt][ch]
            raw_heights.extend(heights)

            cvals = [adc_to_pe(h) for h in heights]
            calibrated_heights.extend(cvals)

            # SPE timing gate (continuous band)
            timing_spe.extend([
                t for t, pe in zip(times, cvals)
                if (0.5 <= pe < 1.5)
            ])

        # Only once per event (but we’re inside ch loop): add trigger sum on ch==0
        if ch == 0:
            total_pe = event_total_pe.get(evt, 0.0)
            total_raw = event_total_raw.get(evt, 0.0)
            trigger_pe_sum.append(total_pe)
            trigger_raw_sum.append(total_raw)

    # Store for plotting/ROOT
    all_height_histo_per_channel.append(np.array(raw_heights))
    all_height_histo_per_channel_calibrated.append(np.array(calibrated_heights))
    all_timing_histo_per_channel.append(np.array(timing_spe))

    summed_height_histo.extend(raw_heights)
    summed_height_histo_calibrated.extend(calibrated_heights)
    summed_timing_SPE_histo.extend(timing_spe)

    # =================== ROOT OUTPUTS (DOUBLE + TH1D with explicit bins) ===================

    # ---- Uncalibrated: TTree (double) + TH1D ----
    root_file_h = ROOT.TFile(f"heightHisto_{start_files}-{max_no_files}files-{run_date}-{ch}.root", "RECREATE")

    tree_h = ROOT.TTree("tree", "Pulse Height (uncalibrated)")
    pulse_height = array.array("d", [0.0])  # double
    tree_h.Branch("pulse_height", pulse_height, "pulse_height/D")
    for val in raw_heights:
        pulse_height[0] = float(val)
        tree_h.Fill()
    tree_h.Write()

    lo_u, hi_u = _make_range(raw_heights, pad_frac=0.02)
    nb_u = _nbins_for(raw_heights, target_bins=1000)
    h_uncal = ROOT.TH1D("h_uncal", "Pulse Height (uncalibrated);ADC (deconvolved);Counts", nb_u, lo_u, hi_u)
    for val in raw_heights:
        h_uncal.Fill(float(val))
    h_uncal.Write()

    root_file_h.Close()
    print(f"check root file : {root_file_h}")

    # ---- Calibrated: TTree (double) + TH1D ----
    root_file_hc = ROOT.TFile(f"heightHistoCalibrated_{start_files}-{max_no_files}files-{run_date}-{ch}.root", "RECREATE")

    tree_hc = ROOT.TTree("tree", "Pulse Height (calibrated, continuous PE)")
    pulse_height_cal = array.array("d", [0.0])  # double
    tree_hc.Branch("pulse_height_calibrated", pulse_height_cal, "pulse_height_calibrated/D")
    for val in calibrated_heights:
        pulse_height_cal[0] = float(val)
        tree_hc.Fill()
    tree_hc.Write()

    lo_c, hi_c = _make_range(calibrated_heights, pad_frac=0.02)
    if hi_c < 0.5:   # make 0..1 visible if data is very close to 0
        hi_c = 1.0
    nb_c = _nbins_for(calibrated_heights, target_bins=1000)
    h_cal = ROOT.TH1D("h_cal", "Pulse Height (calibrated);PE;Counts", nb_c, lo_c, hi_c)
    for val in calibrated_heights:
        h_cal.Fill(float(val))
    h_cal.Write()

    root_file_hc.Close()
    print(f"Check Root files:{root_file_h} and {root_file_hc}")

end_time = time.perf_counter()
print("Runtime:",(end_time-start_time))

# =================== Trigger-sum (DOUBLE) + optional TH1D ===================
root_file_hc_triggerSum = ROOT.TFile(
    f"heightHistoCalibratedTriggerSum_{start_files}-{max_no_files}files-{run_date}.root", "RECREATE"
)
tree_hc_triggerSum = ROOT.TTree("tree", "Trigger Sum (calibrated PE)")
trigger_pe = array.array("d", [0.0])  # double
tree_hc_triggerSum.Branch("trigger_pe_sum", trigger_pe, "trigger_pe_sum/D")
for val in trigger_pe_sum:
    trigger_pe[0] = float(val)
    tree_hc_triggerSum.Fill()
tree_hc_triggerSum.Write()

# Optional: also write a histogram for quick browsing
lo_s, hi_s = _make_range(trigger_pe_sum, pad_frac=0.02)
nb_s = _nbins_for(trigger_pe_sum, target_bins=400)
h_trisum = ROOT.TH1D("h_trigger_sum", "Trigger SiPM PE Sum;PE (sum 9+10+11);Counts", nb_s, lo_s, hi_s)
for val in trigger_pe_sum:
    h_trisum.Fill(float(val))
h_trisum.Write()

root_file_hc_triggerSum.Close()
print(f"check ROOT file for sum of 3 triggerSiPM calibrated:{root_file_hc_triggerSum}")


# =================== Trigger-sum RAW (DOUBLE) + TH1D ===================
root_file_h_raw_triggerSum = ROOT.TFile(
    f"heightHistoRawTriggerSum_{start_files}-{max_no_files}files-{run_date}.root", "RECREATE"
)
tree_h_raw_triggerSum = ROOT.TTree("tree", "Trigger Sum (raw ADC)")
trigger_raw = array.array("d", [0.0])  # double
tree_h_raw_triggerSum.Branch("trigger_raw_sum", trigger_raw, "trigger_raw_sum/D")
for val in trigger_raw_sum:
    trigger_raw[0] = float(val)
    tree_h_raw_triggerSum.Fill()
tree_h_raw_triggerSum.Write()

# Optional: quick-browse histogram
lo_r, hi_r = _make_range(trigger_raw_sum, pad_frac=0.02)
nb_r = _nbins_for(trigger_raw_sum, target_bins=400)
h_trisum_raw = ROOT.TH1D("h_trigger_sum_raw", "Trigger SiPM Raw Sum;ADC (sum 9+10+11);Counts", nb_r, lo_r, hi_r)
for val in trigger_raw_sum:
    h_trisum_raw.Fill(float(val))
h_trisum_raw.Write()

root_file_h_raw_triggerSum.Close()
print(f"check ROOT file for RAW trigger sum:{root_file_h_raw_triggerSum}")


elapsed_time = time.perf_counter()
print("Run Time for code:",elapsed_time-start_time)

# ---- Print accounting summary (once per event counters) ----
print("\n=== Event Accounting Summary ===")
print(f"APPLY_TRIANGLE_CUT:                   {APPLY_TRIANGLE_CUT}")
print(f"Expected lines (sanity):              {ROWS_PER_FILE} x {NUM_FILES} = {ROWS_PER_FILE*NUM_FILES}")
print(f"Total unique events iterated:          {total_events_iterated}")
print(f"  Skipped: missing triggers           : {skipped_missing_triggers}")
print(f"  Skipped: >45 peaks in any trigger   : {skipped_gt45_peaks}")
print(f"  Skipped: total PE == 0              : {skipped_zero_total_pe}")
print(f"  Skipped: one trigger PE == 0        : {skipped_channel_zero_pe}")
if APPLY_TRIANGLE_CUT:
    print(f"Considered by triangular cut          : {accepted_tri_cut + rejected_tri_cut}")
    print(f"  Accepted (20-60% PE fractions)      : {accepted_tri_cut}")
    print(f"  Rejected (outside 20-60%)           : {rejected_tri_cut}")
else:
    print("Triangular cut DISABLED: all non-skipped events included.")
print("=================================\n")

## =================== Plotting ===================

plt.figure(figsize=(12,10))
for i, heights in enumerate(all_height_histo_per_channel):
    color = next(color_cycle)
    plt.hist(heights, bins=1000, range=(0,1), color=color, alpha=0.55, linewidth=1.5,
             histtype='step', label=f'Ch {i}')
plt.hist(summed_height_histo, bins=1000, color='black', alpha=0.85,
         histtype='step', linewidth=1.5, label='Summed')
plt.yscale('log')
try:
    title_slice = first_filename[18:24]
except Exception:
    title_slice = ""
plt.title(f"Pulse Height Histogram Uncalibrated (Summed + All Channels)\nFiles: {title_slice}")
plt.xlabel("ADC value in deconved")
plt.ylabel("Count")
plt.legend(fontsize='small', loc='upper right')
plt.tight_layout()
plt.show()

# Trigger sum plot (calibrated continuous)
plt.figure(figsize=(12, 8))
plt.hist(trigger_pe_sum, bins=400, range=(0,200), histtype='step', label='Sum of Ch 9+10+11 (PE)')
plt.xlim(0,200)
plt.title(f"Trigger SiPM PE Sum (calibrated continuous) — {run_date}")
plt.xlabel("PE units")
plt.ylabel("Count")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


#### Trigger sum plot (RAW ADC)
plt.figure(figsize=(12, 8))

lo_r, hi_r = _make_range(trigger_raw_sum, pad_frac=0.02)
nb_r = _nbins_for(trigger_raw_sum, target_bins=400)
plt.hist(trigger_raw_sum, bins=nb_r, range=(lo_r, hi_r), histtype='step', label='Sum of Ch 9+10+11 (RAW ADC)')
plt.title(f"Trigger SiPM Raw Sum (uncalibrated) — {run_date}")
plt.xlabel("ADC (deconvolved units)")
plt.ylabel("Count")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# ===  Ternary scatter of triangular-cut acceptance =======================
tern_f9  = np.array(tern_f9, dtype=float)
tern_f10 = np.array(tern_f10, dtype=float)
tern_f11 = np.array(tern_f11, dtype=float)
tern_accept = np.array(tern_accept, dtype=bool)

def _ternary_xy(f9, f10, f11):
    x = f10 + 0.5 * f11
    y = (math.sqrt(3)/2.0) * f11
    return x, y

tx, ty = _ternary_xy(tern_f9, tern_f10, tern_f11)

plt.figure(figsize=(9, 8))
ax = plt.gca()
tri = np.array([[0,0],[1,0],[0.5, math.sqrt(3)/2.0],[0,0]])
ax.plot(tri[:,0], tri[:,1], linewidth=1.5)
h = math.sqrt(3)/2.0
for c in np.arange(0.1, 1.0, 0.1):
    t = np.linspace(0, 1-c, 100)
    xx, yy = _ternary_xy(1-c-t, t, (1-c)-t); ax.plot(xx, yy, linewidth=0.5, alpha=0.3)
    xx, yy = _ternary_xy(t, c*np.ones_like(t), (1-c)-t); ax.plot(xx, yy, linewidth=0.5, alpha=0.3)
    xx, yy = _ternary_xy(t, (1-c)-t, c*np.ones_like(t)); ax.plot(xx, yy, linewidth=0.5, alpha=0.3)
bad = ~tern_accept; ok  = tern_accept
ax.scatter(tx[bad], ty[bad], s=14, marker='x', alpha=0.6, label=f"Rejected ({bad.sum()})")
ax.scatter(tx[ok],  ty[ok],  s=18, marker='o', alpha=0.7, label=f"Accepted ({ok.sum()})")
ax.text(-0.04, -0.035, "Det 9",  ha='right', va='top')
ax.text(1.04,  -0.035, "Det 10", ha='left',  va='top')
ax.text(0.5,    h + 0.035, "Det 11", ha='center', va='bottom')
ax.set_xticks([]); ax.set_yticks([])
ax.set_xlim(-0.05, 1.05); ax.set_ylim(-0.05, h + 0.08)
ax.set_aspect('equal', 'box')
title = "Triangular cut on PE fractions (20–60%) — Det 9/10/11" if APPLY_TRIANGLE_CUT else "PE fractions (no triangular cut applied)"
plt.title(title)
plt.legend(loc='upper right', frameon=True)
plt.tight_layout()
plt.show()
# ============================================================================
