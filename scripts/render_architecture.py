"""Renders the Elderly Bed-Activity Monitor agentic vision architecture diagram to docs/architecture.png."""
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches

os.makedirs("docs", exist_ok=True)
fig, ax = plt.subplots(figsize=(15, 8), dpi=300)
ax.set_xlim(0, 15)
ax.set_ylim(0, 8)
ax.axis("off")

# Color palette
c_input = "#E3F2FD"       # light blue
c_percept = "#E8F5E9"     # light green
c_temporal = "#FFF3E0"    # light orange
c_agent = "#F3E5F5"       # light purple
c_alert = "#FFEBEE"       # light red
c_output = "#EDE7F6"      # light violet
c_border = "#37474F"

def draw_box(x, y, w, h, text, bg, subtitle="", style="round,pad=0.3"):
    bbox = patches.FancyBboxPatch((x, y), w, h, boxstyle=style,
                                 facecolor=bg, edgecolor=c_border, linewidth=1.5, zorder=2)
    ax.add_patch(bbox)
    ax.text(x + w/2, y + h/2 + (0.15 if subtitle else 0), text,
            ha="center", va="center", fontsize=10, fontweight="bold", color="#212121", zorder=3)
    if subtitle:
        ax.text(x + w/2, y + h/2 - 0.22, subtitle,
                ha="center", va="center", fontsize=8, color="#546E7A", style="italic", zorder=3)

def draw_arrow(x1, y1, x2, y2, label="", dashed=False, color="#455A64"):
    ls = "--" if dashed else "-"
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", lw=1.8, color=color, linestyle=ls, shrinkA=3, shrinkB=3),
                zorder=4)
    if label:
        ax.text((x1 + x2)/2, (y1 + y2)/2 + 0.12, label,
                ha="center", va="center", fontsize=8, fontweight="bold", color="#37474F", zorder=5)

# --- Top row: Ingestion & Perception ---
draw_box(0.5, 6.2, 2.0, 1.0, "Video Stream", c_input, "Raw RGB / CCTV")
draw_box(3.2, 6.2, 2.5, 1.0, "1. Perception", c_percept, "YOLO11s-Pose + ByteTrack (5 fps)")
draw_box(6.4, 6.2, 2.5, 1.0, "2. Patient Selection", c_percept, "Track prior & re-association")
draw_box(9.6, 6.2, 2.5, 1.0, "3. Features & Geometry", c_percept, "Angles, ratios, speed, 1Hz bins")

draw_arrow(2.5, 6.7, 3.2, 6.7)
draw_arrow(5.7, 6.7, 6.4, 6.7)
draw_arrow(8.9, 6.7, 9.6, 6.7)

# Polygon input
draw_box(9.6, 4.7, 2.5, 0.9, "Bed Polygon JSON", c_input, "Normalized coordinates")
draw_arrow(10.85, 5.6, 10.85, 6.2)

# --- Middle row: Temporal & Events & Triage ---
draw_box(9.6, 3.0, 2.5, 1.1, "4. Viterbi Decoder", c_temporal, "Temporal matrix + min-seg hold")
draw_arrow(10.85, 6.2, 10.85, 4.1)

draw_box(6.4, 3.0, 2.5, 1.1, "5. Event FSM", c_temporal, "Exit, return, stand-attempt")
draw_box(6.4, 1.3, 2.5, 1.1, "6. Triage Engine", c_temporal, "Flags low-conf / bed vs floor")

draw_arrow(9.6, 3.55, 8.9, 3.55)
draw_arrow(10.1, 3.0, 8.9, 1.85)

# --- Agentic Loop ---
draw_box(3.2, 1.3, 2.5, 1.1, "7. Bounded Agent", c_agent, "Tool loop (max 4) + cached VLM")
draw_arrow(6.4, 1.85, 5.7, 1.85)
draw_arrow(4.45, 2.4, 6.4, 3.3, label="relabel (if needed)", color="#7B1FA2")

# --- Alerts & Reports ---
draw_box(3.2, 3.0, 2.5, 1.1, "8. Alert Engine", c_alert, "NORMAL / MONITOR / ALERT")
draw_arrow(6.4, 3.55, 5.7, 3.55)

draw_box(0.5, 3.0, 2.0, 1.1, "9. Reports", c_output, "Timeline, Summary, Decisions")
draw_arrow(3.2, 3.55, 2.5, 3.55)

# --- Separate Evaluation Step ---
draw_box(0.5, 1.3, 2.0, 1.1, "Evaluation & Ablation", "#ECEFF1", "Ground-truth metrics (F1, tol±5s)", style="round,pad=0.2")
draw_arrow(1.5, 3.0, 1.5, 2.4, dashed=True, color="#78909C")

plt.title("Elderly Bed-Activity Monitor — Agentic Vision Architecture", fontsize=14, fontweight="bold", pad=20, color="#1A237E")
plt.tight_layout()
plt.savefig("docs/architecture.png", dpi=300, bbox_inches="tight")
plt.close()
print("Saved docs/architecture.png successfully.")

