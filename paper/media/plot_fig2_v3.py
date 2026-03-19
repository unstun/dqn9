"""Fig 2 – UGV Platform architecture (v3: coloured dots + dashed arrows)."""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

# ── style ────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 11,
    'figure.dpi': 300,
})

fig_width, fig_height = 9.0, 5.2
fig, ax = plt.subplots(figsize=(fig_width, fig_height))
ax.axis('off')
ax.set_xlim(0, fig_width)
ax.set_ylim(0, fig_height)
fig.patch.set_facecolor('#FFFFFF')
ax.set_facecolor('#FFFFFF')

# ── images ───────────────────────────────────────────────────────────────
img_robot   = plt.imread('fig_platform_robot.png')
img_jetson  = plt.imread('fig_platform_jetson.png')
img_lidar   = plt.imread('fig_platform_lidar.png')
img_chassis = plt.imread('fig_platform_chassis.png')


def add_image(ax, img, xy, zoom=1.0):
    ab = AnnotationBbox(OffsetImage(img, zoom=zoom), xy,
                        xycoords='data', frameon=False)
    ax.add_artist(ab)

# ── left: robot ──────────────────────────────────────────────────────────
x_robot, y_robot = 2.05, 2.55
add_image(ax, img_robot, (x_robot, y_robot), zoom=0.058)
ax.text(x_robot, 0.25, "UGV Platform", ha='center', va='bottom',
        fontsize=13, fontweight='bold', color='#1A1A1A')

# ── colour palette per subsystem ─────────────────────────────────────────
colours = {
    'jetson':  {'dot': '#E74C3C', 'line': '#E74C3C', 'bg': '#FDF2F2', 'edge': '#F5C6CB'},  # red
    'lidar':   {'dot': '#2E86C1', 'line': '#2E86C1', 'bg': '#EBF5FB', 'edge': '#AED6F1'},  # blue
    'chassis': {'dot': '#27AE60', 'line': '#27AE60', 'bg': '#EAFAF1', 'edge': '#A9DFBF'},  # green
}

# ── right: subsystem cards ───────────────────────────────────────────────
x_sub = 6.5
card_w, card_h = 4.2, 1.40
y_cards = [4.00, 2.55, 1.10]

cards = [
    {'key': 'jetson',  'img': img_jetson,  'zoom': 0.38,
     'title': '(a) Jetson AGX Orin',  'desc': 'Onboard AI Computing (275 TOPs)', 'y': y_cards[0]},
    {'key': 'lidar',   'img': img_lidar,   'zoom': 0.38,
     'title': '(b) Livox Mid-360',    'desc': '360° FoV 3-D Point Cloud',        'y': y_cards[1]},
    {'key': 'chassis', 'img': img_chassis, 'zoom': 0.42,
     'title': '(c) Ackermann Chassis', 'desc': 'Bicycle Kinematic Model',        'y': y_cards[2]},
]

for c in cards:
    clr = colours[c['key']]
    yc = c['y']

    # card background
    rect = patches.FancyBboxPatch(
        (x_sub - card_w / 2, yc - card_h / 2), card_w, card_h,
        boxstyle="round,pad=0.08,rounding_size=0.12",
        linewidth=1.6, edgecolor=clr['edge'], facecolor=clr['bg'])
    ax.add_patch(rect)

    # coloured left accent bar
    accent = patches.FancyBboxPatch(
        (x_sub - card_w / 2 - 0.02, yc - card_h / 2 + 0.08),
        0.08, card_h - 0.16,
        boxstyle="round,pad=0,rounding_size=0.04",
        linewidth=0, edgecolor='none', facecolor=clr['dot'])
    ax.add_patch(accent)

    # title
    ax.text(x_sub - card_w / 2 + 0.25, yc + 0.35, c['title'],
            ha='left', va='center', fontsize=11, fontweight='bold', color='#1A1A1A')

    # image
    add_image(ax, c['img'], (x_sub + 0.05, yc - 0.1), zoom=c['zoom'])

    # description
    ax.text(x_sub, yc - card_h / 2 + 0.12, c['desc'],
            ha='center', va='bottom', fontsize=9.5, color='#555555', style='italic')

# ── dots on robot photo ──────────────────────────────────────────────────
dot_positions = {
    'jetson':  (2.30, 2.70),   # mid-centre (Jetson board area)
    'lidar':   (1.65, 3.70),   # top (LiDAR sensor on top)
    'chassis': (2.15, 1.35),   # bottom (chassis / wheels)
}

for key, (dx, dy) in dot_positions.items():
    clr = colours[key]
    # outer glow ring
    ax.plot(dx, dy, 'o', color=clr['dot'], markersize=14, alpha=0.25, zorder=5)
    # solid dot
    ax.plot(dx, dy, 'o', color=clr['dot'], markersize=8, zorder=6)
    # white centre highlight
    ax.plot(dx, dy, 'o', color='white', markersize=3, alpha=0.7, zorder=7)

# ── dashed arrows: dot → card ────────────────────────────────────────────
def draw_dashed_connector(ax, xy_from, xy_to, elbow_x, color):
    """Right-angle dashed connector with arrowhead."""
    # horizontal from dot, then vertical, then horizontal to card
    xs = [xy_from[0], elbow_x, elbow_x, xy_to[0]]
    ys = [xy_from[1], xy_from[1], xy_to[1], xy_to[1]]
    ax.plot(xs, ys, color=color, lw=1.5, ls='--', dash_capstyle='round',
            dashes=(5, 3), alpha=0.85, zorder=4)
    # arrowhead at the end
    ax.annotate("", xy=xy_to, xytext=(elbow_x, xy_to[1]),
                arrowprops=dict(arrowstyle="-|>,head_length=0.35,head_width=0.18",
                                color=color, lw=1.5), zorder=4)


elbow_x = 3.90
card_left = x_sub - card_w / 2 - 0.05

for c in cards:
    key = c['key']
    dot_xy = dot_positions[key]
    card_xy = (card_left, c['y'])
    draw_dashed_connector(ax, dot_xy, card_xy, elbow_x, colours[key]['line'])

# ── save ─────────────────────────────────────────────────────────────────
plt.tight_layout(pad=0)
for ext, kw in [('pdf', {'format': 'pdf'}), ('png', {'format': 'png', 'dpi': 300})]:
    out = f"fig2_platform_v3.{ext}"
    plt.savefig(out, bbox_inches='tight', transparent=False, facecolor='#FFFFFF', **kw)
    print(f"Saved {out}")
