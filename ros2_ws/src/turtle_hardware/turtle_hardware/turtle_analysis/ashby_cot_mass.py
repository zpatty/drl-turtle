"""
Ashby-style plot: Cost of Transport (COT, dimensionless) vs. body mass (kg)
for biomimetic swimming/amphibious robots.

Log-log axes (standard for Ashby plots) since both COT and mass span
orders of magnitude. Marker SHAPE encodes tethered/untethered.
Marker COLOR encodes autonomy level. Point size can optionally encode
a third variable (left as constant here, but see note at bottom).

Fill in / correct values as needed -- COT values here are the
dimensionless form (COT = P / (m*g*v)); where a paper only reported
J/kg/m, divide by g=9.81 to convert before adding here.

Requires: pip install adjustText
"""

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from adjustText import adjust_text

# ---------------------------------------------------------------------------
# DATA -- edit this list. Each dict is one robot (or one operating point).
# autonomy: 'none' (rig-constrained/open-loop), 'semi' (programmed gaits,
#           externally powered/commanded), 'full' (closed-loop autonomous)
# tethered: True / False
# cot: dimensionless cost of transport (P / mgv). Use None if not reported.
# mass_kg: body mass in kg
# ---------------------------------------------------------------------------
robots = [
    dict(name="Cornelia",      mass_kg=30.0,  cot=0.072, tethered=True,  autonomy="none"),
    dict(name="ART (aquatic)", mass_kg=9.0,   cot=3.0,   tethered=True,  autonomy="semi"),
    dict(name="ART (terrestrial)", mass_kg=9.0, cot=10.0, tethered=True, autonomy="semi"),
    dict(name="Tunabot",       mass_kg=0.306, cot=2.83,  tethered=True,  autonomy="none"),
    dict(name="Tunabot Flex",  mass_kg=0.190, cot=1.876, tethered=True,  autonomy="none"),
    dict(name="FINBOT",        mass_kg=0.160, cot=8.2,   tethered=False, autonomy="full"),
    dict(name="Paddle robot",  mass_kg=0.414, cot=2.5,   tethered=False, autonomy="semi"),
    # dict(name="Eel-inspired",  mass_kg=None,  cot=10.72, tethered=None,  autonomy=None), can't do bc no access to paper
    dict(name="Squid-inspired",mass_kg= 1.23,  cot=0.087, tethered=True,  autonomy="none"),
    dict(name="Flexible robotic fish", mass_kg=1.67, cot=0.293, tethered=False, autonomy="none"),
    dict(name="HASEL jellyfish", mass_kg=0.17, cot=1.619, tethered=False, autonomy="semi"),
    dict(name="SoFi",          mass_kg=1.6,   cot=14,  tethered=False, autonomy="semi"),  # remote-controlled, not closed-loop autonomous
    dict(name="CUREE",         mass_kg=24.9,  cot=None,  tethered=False, autonomy="full"),
    dict(name="Crush",       mass_kg=10.0,  cot=2.9,  tethered=False,  autonomy="full"),  # <-- add your robot here
]

# ---------------------------------------------------------------------------
# STYLE MAPPINGS
# ---------------------------------------------------------------------------
marker_map = {
    True:  "o",   # tethered -> circle
    False: "^",   # untethered -> triangle
    None:  "s",   # unknown -> square
}

color_map = {
    "none": "#B0B0B0",   # gray  -- no autonomy / rig-constrained
    "semi": "#4C72B0",   # blue  -- semi-autonomous / remote-controlled
    "full": "#DD8452",   # orange -- fully autonomous closed-loop
    None:   "#000000",
}

fig, ax = plt.subplots(figsize=(9, 7))

# Plot each robot individually so marker/color can vary per point
plotted = [r for r in robots if r["mass_kg"] is not None and r["cot"] is not None]
texts = []
for r in plotted:
    ax.scatter(
        r["mass_kg"], r["cot"],
        marker=marker_map[r["tethered"]],
        color=color_map[r["autonomy"]],
        s=140, edgecolor="black", linewidth=0.8, zorder=3,
    )
    t = ax.text(r["mass_kg"], r["cot"], r["name"], fontsize=8.5, zorder=5)
    texts.append(t)

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("Body mass (kg)", fontsize=12)
ax.set_ylabel("Cost of transport, COT = P / (mgv)  (dimensionless)", fontsize=12)
ax.set_title("Cost of transport vs. body mass across biomimetic robots", fontsize=13)
ax.grid(True, which="both", ls=":", alpha=0.5)

# --- Automatic label placement: repel overlapping labels, draw leader lines ---
adjust_text(
    texts,
    ax=ax,
    expand_points=(1.4, 1.6),
    expand_text=(1.2, 1.4),
    force_text=0.6,
    force_points=0.3,
)

# --- Legends: one for marker shape (tethered), one for color (autonomy) ---
tether_legend = [
    Line2D([0], [0], marker="o", color="w", markerfacecolor="gray",
           markeredgecolor="black", markersize=10, label="Tethered"),
    Line2D([0], [0], marker="^", color="w", markerfacecolor="gray",
           markeredgecolor="black", markersize=10, label="Untethered"),
]
autonomy_legend = [
    Line2D([0], [0], marker="o", color="w", markerfacecolor=color_map["none"],
           markeredgecolor="black", markersize=10, label="No autonomy"),
    Line2D([0], [0], marker="o", color="w", markerfacecolor=color_map["semi"],
           markeredgecolor="black", markersize=10, label="Semi-autonomous"),
    Line2D([0], [0], marker="o", color="w", markerfacecolor=color_map["full"],
           markeredgecolor="black", markersize=10, label="Fully autonomous"),
]

leg1 = ax.legend(handles=tether_legend, loc="upper right", title="Tether status", fontsize=9)
ax.add_artist(leg1)
ax.legend(handles=autonomy_legend, loc="lower left", title="Autonomy level", fontsize=9)

fig.tight_layout()
fig.savefig("ashby_cot_vs_mass.png", dpi=300)
plt.show()

# ---------------------------------------------------------------------------
# NOTE on adding a third categorical/continuous variable:
# - A 4th dimension (e.g. "field-deployed vs lab-only") can be shown by
#   marker size (s=...) if continuous, or by adding hatching/edge color
#   if categorical (edgecolor="red" for field-tested, "black" otherwise).
# ---------------------------------------------------------------------------