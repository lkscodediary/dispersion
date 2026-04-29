"""
SP500 Sector Dispersion Model
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from itertools import combinations
import glob


def load_data():
    """
    Loads Prices and sector weights

    Parameters
    ----------

    Returns
    -------
    dict
        {"sector": {"ticker": str, "weight": float}}
    """
    sp500 = pd.read_excel("data/SPY.xlsx")
    sp500 = sp500[~sp500.Ticker.isin(['-','2602335D'])][['Ticker','Weight']] #Just Tickers and Weight
    sp500 = dict(zip(sp500.Ticker.values, sp500.Weight.astype("float"))) #Convert to dictionary

    sector_files = glob.glob("data/*.xlsx")
    sectors = {}
    for file in sector_files: #Go through each file
        if "SPY" not in file:
            temp_dict = {} #To save sector weights
            temp = pd.read_excel(file) #read the excel files in to data frame
            for row in temp.itertuples():# A very fast itertool
                #If the sector ticker belongs in the SP500 then include it. This is to avoid non equity securities
                if row.Ticker in sp500.keys():
                    temp_dict[row.Ticker] = row.Weight/100 #State Streets provides them in percentage
                else:
                    print(f"Non Equity {row.Ticker} ticker in {file[5:8]}")
            print(f"Total weight in {file[5:8]}: {sum(temp_dict.values())}")
            sectors[file[5:8]] = temp_dict
    return sectors



def cross_sectional_std(returns: pd.Series,
                        tickers: list,
                        weights: list) -> float:
    """
    Compute WEIGHTED cross-sectional standard deviation.

    Formula
    -------
        μ_w  = Σ  w_i * r_i                          (weighted mean)
        σ²_w = Σ  w_i * (r_i − μ_w)²                (weighted variance)
        σ_w  = sqrt(σ²_w)

    Parameters
    ----------
    returns : pd.Series   — full universe annual returns
    tickers : list        — sector constituent tickers
    weights : dict        — {ticker: {"sector_weight": float, ...}}

    Returns
    -------
    float — weighted cross-sectional std dev (decimal)
    """
    r = returns[tickers].values
    w = np.array(weights)
    w = w / sum(weights) # re-normalise for safety. This may shift the weight just by tad bit but negligible 

    weighted_mean = np.dot(w, r)
    weighted_var  = np.dot(w, (r - weighted_mean) ** 2)
    return float(np.sqrt(weighted_var))




print("+" * 70)
print(" SP500 SECTOR DISPERSION MODEL")
print("+" * 70)

# ─────────────────────────────────────────────────────────────────────────────
# Data
# ─────────────────────────────────────────────────────────────────────────────
print("-" * 70)
print(" Gather Data")
sectors = load_data()
print("Study is done for the period of 04/1/2025 and 3/31/2026")
price_matrix = pd.read_csv("data/SP500_2026_04_28.csv", index_col=0)
price_matrix = price_matrix.loc["2025-04-01":"2026-03-31"]
price_matrix = price_matrix.bfill() #This is so that some tickers such as Q that was added later wont return NAs
print("-" * 70)

# ─────────────────────────────────────────────────────────────────────────────
# Annual Returns
# ─────────────────────────────────────────────────────────────────────────────
print("-" * 70)
print(" Compute Annual Returns")
annual_returns = price_matrix.iloc[-1] / price_matrix.iloc[0] - 1
print("Annual returns sample, first 10 tickers):")
annual_returns.head(10).map("{:+.2%}".format)
print("-" * 70)

# ─────────────────────────────────────────────────────────────────────────────
# Sector dispersion metrics
# ─────────────────────────────────────────────────────────────────────────────
print("-" * 70)
print(" Compute Sector dispersion metrics")
records = []
for sector in sectors.keys():
    records.append({
        "Sector"              : sector,
        "Cross-Sec Std (↑ = more dispersion)": cross_sectional_std(annual_returns, list(sectors[sector].keys()), list(sectors[sector].values())),
        "N Constituents"      : len(sectors[sector].keys()),
    })

dispersion_table = pd.DataFrame(records).set_index("Sector")
dispersion_table = dispersion_table.sort_values("Cross-Sec Std (↑ = more dispersion)",ascending=False)
temp = dispersion_table.copy()
temp["Cross-Sec Std (↑ = more dispersion)"] = temp["Cross-Sec Std (↑ = more dispersion)"].map("{:+.2%}".format)
print(temp)
print("-" * 70)

# ─────────────────────────────────────────────────────────────────────────────
# Capture Winning Sector for later
# ─────────────────────────────────────────────────────────────────────────────
print("-" * 70)
print(" Winning sector")
winner = dispersion_table.index[0]
print("-" * 70)

# ─────────────────────────────────────────────────────────────────────────────
# Compute L/S Spread
# ─────────────────────────────────────────────────────────────────────────────
print("-" * 70)
print(" L/S Spread & Interpretation")
records = []
for sector, tickers_weights in sectors.items():
    tickers = list(tickers_weights.keys())
    sect    = annual_returns[tickers]

    best_ticker  = sect.idxmax()
    worst_ticker = sect.idxmin()

    best_w  = tickers_weights[best_ticker]
    worst_w = tickers_weights[worst_ticker]

    records.append({
        "Sector"          : sector,
        "Best Stock"      : best_ticker,
        "Best Weight"     : best_w,
        "Best Ret"        : sect.max(),
        "Best Wtd Contrib": best_w  * sect.max(),
        "Worst Stock"     : worst_ticker,
        "Worst Weight"    : worst_w,
        "Worst Ret"       : sect.min(),
        "Worst Wtd Contrib": worst_w * sect.min(),
        "Raw L/S Spread"  : sect.max() - sect.min(),
        "Wtd L/S Spread"  : best_w * sect.max() - worst_w * sect.min(),
    })
ls_spread = pd.DataFrame(records).set_index("Sector").sort_values("Raw L/S Spread", ascending=False)

#Format
display_ls = ls_spread.copy()
for col in ["Best Ret", "Best Wtd Contrib", "Worst Ret", "Worst Wtd Contrib", "Raw L/S Spread", "Wtd L/S Spread"]:
    display_ls[col] = display_ls[col].map("{:+.2%}".format)
for col in ["Best Weight", "Worst Weight"]:
    display_ls[col] = display_ls[col].map("{:.3f}".format)
print(display_ls)
print("-" * 70)

# ─────────────────────────────────────────────────────────────────────────────
# PLOT
# ─────────────────────────────────────────────────────────────────────────────
print("-" * 70)
print(" Plots")

sector_colors = {'XLC': '#1f77b4',
                 'XLP': '#ff7f0e',
                 'XLY': '#2ca02c',
                 'XLU': '#d62728',
                 'XLR': '#9467bd',
                 'XLV': '#8c564b',
                 'XLI': '#e377c2',
                 'XLF': '#7f7f7f',
                 'XLK': '#bcbd22',
                 'XLB': '#17becf',
                 'XLE': '#4169E1'}

fig = plt.figure(figsize=(20, 16))
fig.suptitle("SP500 Sector Dispersion Model",
             fontsize=15, fontweight="bold", y=0.98)
gs = gridspec.GridSpec(2, 2, hspace=0.40, wspace=0.30)

#Panel 1: Normalised Price Paths so that we can see everything on the same scale
ax1 = fig.add_subplot(gs[0, 0])
normalised = price_matrix / price_matrix.iloc[0]
for sector, tickers_weights in sectors.items():
    for ticker, weight in tickers_weights.items():
        lw = 0.5 + weight * 15  # thicker = heavier weight
        ax1.plot(normalised[ticker].values, color=sector_colors[sector], alpha=0.4, linewidth=lw)
for sname, col in sector_colors.items():
    ax1.plot([], [], color=col, label=sname, linewidth=2)
ax1.legend(fontsize=8)
ax1.set_title("Normalised Price Paths (line thickness proportional to sector weight)",
              fontweight="bold")
ax1.set_xlabel("Trading Day")
ax1.set_ylabel("Normalised Price (Day 0 = 1.0)")

# Bubble plot — return vs sector weight ───────────────────────
ax2 = fig.add_subplot(gs[0, 1])
x_pos = dict(zip(list(sectors.keys()), list(range(0,11))))
jitter_rng = np.random.default_rng(0)
for sector, tickers_weights in sectors.items():
    for ticker,weight in tickers_weights.items():
        ret = annual_returns[ticker] * 100
        jitter = jitter_rng.uniform(-0.25, 0.25)
        ax2.scatter(x_pos[sector] + jitter, ret,
                    s = weight * 3000, color=sector_colors[sector], alpha=0.55, edgecolors="black", linewidths=0.4)
ax2.set_xticks(list(range(0,11)))
ax2.set_xticklabels(list(sectors.keys()))
ax2.axhline(0, color="grey", linestyle="--", linewidth=0.8)
ax2.set_title("Annual Returns by Constituent (bubble size proportional to sector weight)", fontweight="bold")
ax2.set_ylabel("Annual Return (%)")

# ── Panel 3: Weighted Cross-Sec Std Dev ──────────────────────────────────
ax3 = fig.add_subplot(gs[1, 0])
std_vals   = dispersion_table["Cross-Sec Std (↑ = more dispersion)"] * 100
bar_colors = [sector_colors[s] for s in std_vals.index]
bars = ax3.bar(std_vals.index, std_vals.values, color=bar_colors, edgecolor="black", linewidth=0.7)
for bar, label in zip(bars, std_vals.index):
    if label == winner:
        bar.set_edgecolor("red")
        bar.set_linewidth(2.5)
        ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                 "★ HIGHEST DISPERSION", ha="center", va="bottom", fontsize=8, color="red", fontweight="bold")
ax3.set_title(" Weighted Cross-Sectional Std Dev",fontweight="bold")
ax3.set_ylabel("Weighted Cross-Sec Std Dev (%)")

# ── Panel 4: Raw vs Weighted L/S Spread ─────────────────────────────────
ax4 = fig.add_subplot(gs[1, 1])
sectors_ordered = ls_spread.index.tolist()
x      = np.arange(len(sectors_ordered))
width  = 0.35
raw_vals = ls_spread["Raw L/S Spread"] * 100
wtd_vals = ls_spread["Wtd L/S Spread"] * 100
# b1 = ax4.bar(x - width/2, raw_vals.values, width, label="Raw Spread",
#              color=[sector_colors[s] for s in sectors_ordered],
#              edgecolor="black", linewidth=0.7, alpha=0.9)
b2 = ax4.bar(x + width/2, wtd_vals.values, width, label="Weighted Spread",
             color=[sector_colors[s] for s in sectors_ordered],
             edgecolor="black", linewidth=0.7, alpha=0.45, hatch="//")
ax4.set_xticks(x)
ax4.set_xticklabels(sectors_ordered)
ax4.legend(fontsize=8)
ax4.set_title("Weight-Adjusted L/S Spread",fontweight="bold")
ax4.set_ylabel("L/S Spread (%)")
plt.savefig("sp500_dispersion.png",dpi=150, bbox_inches="tight")
plt.show()
print("-" * 70)