import uproot
import numpy as np
import time
import gi
gi.require_version("Gtk", "3.0")

from scipy import stats as st
from scipy.signal import wiener, savgol_filter, find_peaks
from scipy.signal.windows import tukey

import matplotlib.pyplot as plt

import pandas as pd

plt.rcParams.update({
    'font.size': 28,
    'axes.titlesize': 28,
    'axes.labelsize': 28,
    'xtick.labelsize': 28,
    'ytick.labelsize': 28,
    'legend.fontsize': 28,
})


channel = 9
root_path = "btbSimOffset-2025-09-09-16-09-1000000.root"  
tree_name = "SimTree"
branch_path = f"chan{channel}/hits/hits.startTime"


with uproot.open(root_path) as f:
    tree = f[tree_name]

    # confirm available branches
    print("Available branches:", tree.keys(filter_name="*startTime*"))

    # extract the leaf
    startTime = tree[branch_path].array(library="np")

# optional: save to CSV
pd.DataFrame({"startTime": startTime}).to_csv(f"startTime{channel}.csv", index=False)
print(f"Extracted {len(startTime)} entries from {branch_path} and saved to startTime{channel}.csv")