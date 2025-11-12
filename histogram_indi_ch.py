#!/usr/bin/env python3
"""
Compare PMT peak histograms across multiple runs and always write to ROOT.


# [event]    [peak_indices]    [peak_heights]
"""
##later changed into individual channel list

import os
import re
import numpy as np
import matplotlib.pyplot as plt
import ROOT 

import matplotlib.ticker as ticker


# -------------------------------------------------------------------------
# -----------------------------
# -------------------------------------------------------------------------

channel = 6

files = [
    f"../files11_10_2025_new/deconv_peakHeights-0-50files-11_10_2025-{channel}.txt",
    f"../files11_07_2025_new/deconv_peakHeights-150-200files-11_07_2025-{channel}.txt",
    f"../files10_30_2025_new/deconv_peakHeights-0-50files-10_30_2025-{channel}.txt",
    f"../files10_23_2025_new/deconv_peakHeights-0-50files-10_23_2025-{channel}.txt",
    
]

# files = [
#     "../files11_10_2025_new/peaks_ch12-0-50files-11_10_2025.txt",
#     "../files11_07_2025_new/peaks_ch12-150-200files-11_07_2025.txt",
#     "../files10_30_2025_new/peaks_ch12-0-50files-10_30_2025.txt",
#     "../files10_23_2025_new/peaks_ch12-0-50files-10_23_2025.txt",
#     ]



labels = [
    f"ch{channel}_Run_11_10",
    f"ch{channel}_Run_11_07",
    f"ch{channel}_Run_10_30",
    f"ch{channel}_Run_10_23",
]

HEIGHT_MIN = 0.021          ##there are some noisy peaks in deconved wf



outdir = "figs"
root_outfile = f"ch{channel}_compare.root"

# Histogram settings
height_bins = 1000
index_bins = 2000
height_range = (0,1)         # (LO, HI) or None for auto
index_range = (200, 7000)   # 
density = False             ##for nmralizing histogram, not using

# -------------------------------------------------------------------------

# -------------------------------------------------------------------------
LINE_RE = re.compile(
    r"^\s*\[(?P<evt>\d+)\]\s*\[(?P<idx>[^\]]*)\]\s*\[(?P<hgt>[^\]]*)\]\s*$"
)

def parse_peak_file(path):
    indices_list, heights_list = [], []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            m = LINE_RE.match(line)
            if not m:
                continue
            idx_str = m.group("idx").strip()
            hgt_str = m.group("hgt").strip()
            idx_vals = [int(s) for s in idx_str.split(",") if s] if idx_str else []
            hgt_vals = [float(s) for s in hgt_str.split(",") if s] if hgt_str else []
            n = min(len(idx_vals), len(hgt_vals))
            if n > 0:
                indices_list.append(idx_vals[:n])
                heights_list.append(hgt_vals[:n])
    if indices_list:
        indices = np.array([v for row in indices_list for v in row], dtype=np.int32)
        heights = np.array([v for row in heights_list for v in row], dtype=np.float32)
    else:
        indices = np.empty((0,), dtype=np.int32)
        heights = np.empty((0,), dtype=np.float32)
    return indices, heights

def compute_common_range(data_dict):
    arrays = [v for v in data_dict.values() if v.size > 0]
    if not arrays:
        return (0.0, 1.0)
    lo = min(float(np.min(a)) for a in arrays)
    hi = max(float(np.max(a)) for a in arrays)
    if lo == hi:
        lo, hi = lo - 0.5, hi + 0.5
    return (lo, hi)

def plot_overlay(data_dict, bins, range_, xlabel, title, outfile, density=False):
    
    plt.figure(figsize=(12, 10))             # larger canvas
    plt.rcParams.update({
        "font.size": 20,                    # base font size
        "axes.labelsize": 20,
        "axes.titlesize": 22,
        "legend.fontsize": 18,
        "xtick.labelsize": 18,
        "ytick.labelsize": 18,
    })

    rng = compute_common_range(data_dict) if range_ is None else range_
    for label, arr in data_dict.items():
        plt.hist(
            arr,
            bins=bins,
            range=rng,
            histtype="step",
            label=label,
            density=density
        )
    plt.xlabel(xlabel)
    plt.ylabel("Counts")
    plt.yscale('log')
    plt.title(title)
    plt.legend()
    if "Index" in title or "index" in xlabel:
        plt.ylim(1e-1, 5e4)
    
    ax = plt.gca()
    ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=15)) 
    plt.grid(
        True,
        which="both",
        linestyle="--",
        linewidth=0.8,
        alpha=0.7
    )
    plt.tight_layout(pad = 1.5)
    os.makedirs(os.path.dirname(outfile), exist_ok=True)
    plt.savefig(outfile, dpi=200)
    plt.close()
    print(f"Saved figure: {outfile}")
    return rng

def sanitize_label(label):
    safe = re.sub(r"[^A-Za-z0-9_]+", "_", label)
    if safe[0].isdigit():
        safe = "L_" + safe
    return safe

def write_root_hists(
    root_path,
    height_data,
    index_data,
    height_bins,
    height_range,
    index_bins,
    index_range
):
    f = ROOT.TFile(root_path, "RECREATE")
    if f.IsZombie():
        raise RuntimeError(f"Could not create ROOT file: {root_path}")

    # Heights
    for label, arr in height_data.items():
        name = f"hHeights_{sanitize_label(label)}"
        title = f"ch{channel} Peak Height - {label};Height (Normalized ADC);"
        hist = ROOT.TH1F(name, title, height_bins, height_range[0], height_range[1])
        hist.Sumw2()
        for v in arr:
            hist.Fill(float(v))
        hist.Write()

    # Indices
    for label, arr in index_data.items():
        name = f"hIndices_{sanitize_label(label)}"
        title = f"ch{channel} Peak Index - {label};Time index (sample);Counts"
        hist = ROOT.TH1F(name, title, index_bins, index_range[0], index_range[1])
        hist.Sumw2()
        for v in arr:
            hist.Fill(float(v))
        hist.Write()

    f.Close()
    print(f"ROOT histograms written to: {root_path}")
    print("Height range:", height_range, " Index range:", index_range)

# -------------------------------------------------------------------------
# ------------------------------- MAIN -----------------------------------
# -------------------------------------------------------------------------
print(f"Loading ch{channel} comparison data...")

# Load runs
runs = {}
for file, label in zip(files, labels):
    indices, heights = parse_peak_file(file)

    # --- Height filter: keep only peaks with height >= HEIGHT_MIN ---
    if heights.size:
        mask = heights >= HEIGHT_MIN
        heights = heights[mask]
        indices = indices[mask]  # drop the paired indices too


    runs[label] = {"indices": indices, "heights": heights}
    print(f"  {label}: {heights.size} peaks")

height_data = {k: v["heights"] for k, v in runs.items()}
index_data = {k: v["indices"].astype(float) for k, v in runs.items()}

# Determine ranges
height_range_used = height_range or compute_common_range(height_data)
index_range_used = index_range or compute_common_range(index_data)

# Plot overlays
plot_overlay(
    height_data,
    bins=height_bins,
    range_=height_range_used,
    xlabel="Peak height (normalized deconv units)",
    title=f"ch{channel} Peak Height Distribution (Overlay)",
    outfile=os.path.join(outdir, f"ch{channel}_peak_heights_overlay.png"),
    density=density
)

plot_overlay(
    index_data,
    bins=index_bins,
    range_=index_range_used,
    xlabel="Peak time index (sample)",
    title=f"ch{channel} Peak Time Index Distribution (Overlay)",
    outfile=os.path.join(outdir, f"ch{channel}_peak_indices_overlay.png"),
    density=density
)

# Write ROOT histograms 
# write_root_hists(
#     root_outfile,
#     height_data,
#     index_data,
#     height_bins,
#     height_range_used,
#     index_bins,
#     index_range_used
# )

print("Done.")
