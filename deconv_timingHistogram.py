############# read the generated txt file (deconv_peakHeights...txt) and make a histogram of pulse heights and timing from it
#### from the peak height will decipher SPE/DPE/TPE     it is heights in deconvolved data not original raw data
import sys
import glob
import uproot
import numpy as np
import ast
import time
import gi

gi.require_version("Gtk", "3.0") 


import matplotlib.pyplot as plt

import itertools

from scipy  import stats   as st
from scipy.optimize import curve_fit
from matplotlib.ticker import MultipleLocator

from scipy.fft import fft, ifft
from scipy.signal import correlate
from scipy.signal import wiener
from scipy.signal import deconvolve
from scipy.signal import savgol_filter
from scipy.signal import find_peaks
from pathlib import Path


import ROOT
import array

start_files = 150
max_no_files = 200
run_date    = '11_07_2025'
channels    = list(range(12))          #not including PMT in timing
input_tpl   = 'deconv_peakHeights-{start}-{maxfiles}files-{date}-{ch}.txt'

# Timing histogram definition (same for per-channel & summed)
nbins  = 7000          
xmin   = 500
xmax   = 7500
bin_edges = np.linspace(xmin, xmax, nbins+1)
bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

NOISE_THRESHOLD = 0.021 ###some noisy peaks creep in


# Matplotlib cosmetics
plt.rcParams.update({
    'font.size': 28,
    'axes.titlesize': 28,
    'axes.labelsize': 28,
    'xtick.labelsize': 28,
    'ytick.labelsize': 28,
    'legend.fontsize': 20,
})

# -----------------------
# Per-channel SPE/DPE/TPE ranges 
# -----------------------
def ranges_for_channel(ch: int):
    if ch == 9:
        return dict(
            SPE=(0.012, 0.0532),
            DPE=(0.0533, 0.0978),
            TPE=(0.0979, 0.1455),
        )
    elif ch == 11:
        return dict(
            SPE=(0.02, 0.05),
            DPE=(0.0531, 0.0985),
            TPE=(0.0986, 0.1347),
        )
    elif ch == 10:
        return dict(
            SPE=(0.02, 0.04),
            DPE=(0.0521, 0.080),
            TPE=(0.0881, 0.1246),
        )
    elif ch == 4:
        return dict(
            SPE=(0.027, 0.0532),
            DPE=(0.0535, 0.095),
            TPE=(0.0979, 0.1455),
        )
    elif ch == 2:
        return dict(
            SPE=(0.020, 0.048),
            DPE=(0.05, 0.085),
            TPE=(0.087, 0.1155),
        )
    else:
        return dict(
            SPE=(0.029, 0.0372),
            DPE=(0.0473, 0.0656),
            TPE=(0.0979, 0.1455),
        )

# -----------------------
# -----------------------
def classify_weight(height, rngs):
    """Return multiplicity weight based on height & ranges."""
    lo, hi = rngs['SPE']
    if lo <= height <= hi:
        return 1
    lo, hi = rngs['DPE']
    if lo <= height <= hi:
        return 2
    lo, hi = rngs['TPE']
    if lo <= height <= hi:
        return 3
    return 0  # not a PE (ignore)

def safe_parse_line(line: str):
   
   
   
    parts = line.strip().split('\t')
    if len(parts) < 3:
        return None
    try:
        row_idx = int(parts[0].strip().strip('[]'))
        indices = ast.literal_eval(parts[1])
        heights = ast.literal_eval(parts[2])
        if not (isinstance(indices, list) and isinstance(heights, list)):
            return None
        return row_idx, indices, heights
    except Exception:
        return None

def fill_hist_from_weighted_times(times, weights, edges):
    """
    
    each timestamp contributes 'weight' counts.
    """
    times = np.asarray(times, dtype=float)
    weights = np.asarray(weights, dtype=float)
    counts, _ = np.histogram(times, bins=edges, weights=weights)
    return counts

def write_hist_to_root(filename, hist_name, counts, edges, title="", xlab="", ylab=""):
 
    f = ROOT.TFile(filename, "UPDATE")  # UPDATE so we can add tree + hist to same file
    # TH1F requires bin edges; 
    edged = array.array('d', edges.tolist())
    h = ROOT.TH1F(hist_name, title, len(edges)-1, edged)
    # Set contents and errors explicitly
    for i, c in enumerate(counts, start=1):  # ROOT bins start at 1
        h.SetBinContent(i, float(c))
        h.SetBinError(i, np.sqrt(c))  # Poisson error !!!
    h.GetXaxis().SetTitle(xlab)
    h.GetYaxis().SetTitle(ylab)
    h.Write()
    f.Close()

def write_times_tree_to_root(filename, tree_name, branch_name, times_iterable):
    """
    Write a TTree with one float branch containing all timestamps (unweighted).
    """
    f = ROOT.TFile(filename, "UPDATE")
    tree = ROOT.TTree(tree_name, "Tree storing timing")
    val = array.array("f", [0.0])
    tree.Branch(branch_name, val, f"{branch_name}/F")
    for t in times_iterable:
        val[0] = float(t)
        tree.Fill()
    tree.Write()
    f.Close()

# -----------------------
# Main
# -----------------------
def main():
    t0 = time.perf_counter()

    color_cycle = itertools.cycle(plt.cm.tab20.colors)
    all_channel_counts = []
    all_channel_times_for_plot = []  # for matplotlib overlays (unweighted plotting like before)
    summed_counts = np.zeros(nbins, dtype=float)

    for ch in channels:
        infile = input_tpl.format(start=start_files, maxfiles=max_no_files, date=run_date, ch=ch)
        p = Path(infile)
        if not p.exists():
            print(f"[warn] Missing {p} — skipping channel {ch}")
            all_channel_counts.append(np.zeros(nbins))
            all_channel_times_for_plot.append([])
            continue

        rngs = ranges_for_channel(ch)

        # Collect timestamps and weights
        times = []
        weights = []
        # Also collect unweighted times like before for plotting (each DPE/TPE add multiple entries)
        times_for_plot = []

        with p.open("r") as f:
            for line in f:
                parsed = safe_parse_line(line)
                if parsed is None:
                    continue
                _, indices, heights = parsed
                # Iterate through peaks in the row:
                for idx, h in zip(indices, heights):

                    if h < NOISE_THRESHOLD:
                        continue 

                    w = classify_weight(h, rngs)
                    if w > 0:
                        times.append(idx)
                        weights.append(w)
                        # For plotting to match old behavior:
                        times_for_plot.extend([idx] * w)

        # Build per-channel histogram (weighted)
        ch_counts = fill_hist_from_weighted_times(times, weights, bin_edges)
        all_channel_counts.append(ch_counts)
        all_channel_times_for_plot.append(times_for_plot)
        summed_counts += ch_counts

        # Write to ROOT: one file per channel (like your current code)
        root_name = f"timingHisto_{start_files}-{max_no_files}files-{run_date}-{ch}.root"
        # 1) tree with raw timestamps (unweighted, but repeated for DPE/TPE to match previous append behavior)
        write_times_tree_to_root(
            root_name,
            tree_name="tree",
            branch_name="pulse_timing",
            times_iterable=times_for_plot
        )
        # 2) histogram with exact binning preserved
        write_hist_to_root(
            root_name,
            hist_name="hTiming",
            counts=ch_counts,
            edges=bin_edges,
            title=f"Timing Histogram Ch {ch}",
            xlab="Arrival time (sample)",
            ylab="Count"
        )
        print(f"[ok] Wrote ROOT outputs for channel {ch}: {root_name}")

    # ------------- Plotting (per-channel + summed) -------------
    plt.figure(figsize=(12, 10))
    for i, times_for_plot in enumerate(all_channel_times_for_plot):
        if len(times_for_plot) == 0:
            continue
        color = next(color_cycle)
        plt.hist(times_for_plot, bins=bin_edges, color=color, alpha=0.55, linewidth=1.5,
                 histtype='step', label=f'Ch {i}')

    # summed 
    plt.step((bin_edges[:-1] + bin_edges[1:]) * 0.5, summed_counts, where='mid',
             linewidth=2.0, label='Summed', color='black')

    plt.yscale('log')
    plt.title(f"Timing Histogram (Summed + All Channels)\nFiles: {str(start_files)+'-'+str(max_no_files)}  Date: {run_date}")
    plt.xlabel("Arrival time of PE in 'sample time'")
    plt.ylabel("Count")
    plt.legend(fontsize='small', loc='upper right')
    plt.tight_layout()
    plt.show()

    # Also write a single ROOT file with the **summed** histogram :
    root_summed = f"timingHisto_SUM_{start_files}-{max_no_files}files-{run_date}.root"
    fsum = ROOT.TFile(root_summed, "RECREATE")
    # Save the summed hist only
    edged = array.array('d', bin_edges.tolist())
    hsum = ROOT.TH1F("hTimingSUM", "Timing Histogram (SUM over channels)", len(bin_edges)-1, edged)
    for i, c in enumerate(summed_counts, start=1):
        hsum.SetBinContent(i, float(c))
        hsum.SetBinError(i, np.sqrt(c))
    hsum.GetXaxis().SetTitle("Arrival time (sample)")
    hsum.GetYaxis().SetTitle("Count")
    hsum.Write()
    fsum.Close()
    print(f"[ok] Wrote summed histogram: {root_summed}")

    print(f"Runtime: {time.perf_counter() - t0:.2f} s")

if __name__ == "__main__":
    main()
