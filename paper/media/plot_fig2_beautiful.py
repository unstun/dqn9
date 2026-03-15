import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import os

# 1. 设置现代无衬线字体，类似 Apple 的风格或者顶级扁平化UI
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 11,
    'figure.dpi': 300,
})

fig_width = 10.5
fig_height = 6.0
fig, ax = plt.subplots(figsize=(fig_width, fig_height))
ax.axis('off')
ax.set_xlim(0, fig_width)
ax.set_ylim(0, fig_height)

# 纯白背景
fig.patch.set_facecolor('#FFFFFF')
ax.set_facecolor('#FFFFFF')

# 2. 读取素材
img_robot   = plt.imread('fig_platform_robot.png')
img_lidar   = plt.imread('fig_platform_lidar.png')
img_jetson  = plt.imread('fig_platform_jetson.png')
img_chassis = plt.imread('fig_platform_chassis.png')

def add_image(ax, img, xy, zoom=1.0, box_alignment=(0.5, 0.5)):
    imagebox = OffsetImage(img, zoom=zoom)
    ab = AnnotationBbox(imagebox, xy,
                        xycoords='data',
                        frameon=False,
                        box_alignment=box_alignment)
    ax.add_artist(ab)
    return ab

# 3. 布局设计 - 左侧机器人主板
# 画一个很淡的高级灰底板托底机器人
left_box = patches.FancyBboxPatch(
    (0.2, 0.2), 4.3, 5.0, 
    boxstyle="round,pad=0.1,rounding_size=0.15",
    linewidth=1.0, edgecolor='#E6ECEF', facecolor='#F8FAFC', zorder=0
)
ax.add_patch(left_box)

x_robot, y_robot = 2.4, 2.7
add_image(ax, img_robot, (x_robot, y_robot), zoom=0.065) 

# 主标题
ax.text(fig_width/2, 5.7, "Figure 1: Architecture of the Autonomous UGV Platform with Key Sub-systems", 
        ha='center', va='center', fontsize=16, fontweight='bold', color='#1A202C')

ax.text(x_robot, 0.5, "PLATFORM OVERVIEW", ha='center', va='center', fontsize=14, fontweight='black', color='#2D3748', family='sans-serif')

# 在机器人图上添加精细的局部标注引线 (类似透视图的爆炸线)
def label_robot_part(ax, text, text_xy, point_xy, rad=0.2):
    ax.annotate(text,
                xy=point_xy, xycoords='data',
                xytext=text_xy, textcoords='data',
                fontsize=10, color='#4A5568', ha='center', va='center',
                arrowprops=dict(arrowstyle="-|>,head_length=0.5,head_width=0.25",
                                color='#A0AEC0', lw=1.2,
                                connectionstyle=f"arc3,rad={rad}"))

label_robot_part(ax, "Livox Mid-360\nLiDAR", (0.8, 4.3), (2.0, 4.0), rad=-0.2)
label_robot_part(ax, "Jetson AGX\nOrin", (4.0, 3.8), (2.3, 2.8), rad=0.2)
label_robot_part(ax, "4-wheel drive\nGround Vehicle", (0.8, 3.2), (1.5, 2.5), rad=0.1)
label_robot_part(ax, "Ackermann\nSteering Chassis", (3.8, 1.2), (2.5, 1.4), rad=-0.1)


# 4. 右侧三个高定颜色的子系统模块
x_sub = 7.7
box_width = 4.4
box_height = 1.65
y_subs = [4.4, 2.55, 0.7] 

subs_info = [
    {
        'img': img_lidar, 
        'title': '(a) Livox Mid-360 LiDAR for 3D Sensing', 
        'desc': 'High-density, 360° FoV LiDAR for precise mapping & obstacle detection',
        'zoom': 0.055,
        'y': y_subs[0],
        'bg': '#F0F9FF',   # Sky Blue
        'edge': '#BAE6FD',
        'line': '#0284C7'
    },
    {
        'img': img_jetson, 
        'title': '(b) Jetson AGX Orin for AI Computing', 
        'desc': 'Onboard Edge AI processing (275 TOPs) for path planning & perception',
        'zoom': 0.085,
        'y': y_subs[1],
        'bg': '#F0FDF4',   # Emerald Green
        'edge': '#BBF7D0',
        'line': '#059669'
    },
    {
        'img': img_chassis, 
        'title': '(c) Ackermann Chassis & Bicycle Model', 
        'desc': 'Robust 4WD chassis utilizing Ackermann steering and Bicycle Kinematic',
        'zoom': 0.09,
        'y': y_subs[2],
        'bg': '#FFFBEB',   # Amber Orange
        'edge': '#FEF3C7',
        'line': '#D97706'
    }
]

for info in subs_info:
    y_center = info['y']
    
    # 绘制带颜色的优美圆角矩形
    rect = patches.FancyBboxPatch(
        (x_sub - box_width/2, y_center - box_height/2), 
        box_width, box_height, 
        boxstyle="round,pad=0.1,rounding_size=0.12",
        linewidth=1.5, edgecolor=info['edge'], facecolor=info['bg']
    )
    ax.add_patch(rect)
    
    # 标题 (顶部居中对齐)
    title_y = y_center + 0.6
    ax.text(x_sub - 2.05, title_y, info['title'], ha='left', va='center', fontsize=12, fontweight='bold', color='#1E293B')
    
    # 分隔线
    ax.plot([x_sub - 2.05, x_sub + 2.05], [title_y - 0.2, title_y - 0.2], color=info['edge'], lw=1.5)

    # 图片放在中间偏上
    img_x = x_sub - 0.0
    img_y = y_center - 0.05
    add_image(ax, info['img'], (img_x, img_y), zoom=info['zoom'])
    
    # 将长描述放在底部
    desc_y = y_center - 0.65
    ax.text(x_sub, desc_y, info['desc'], ha='center', va='center', fontsize=10, color='#475569')

# 5. 绘制优雅夺目的引导大曲线
def draw_curved_connector(ax, xy_from, xy_to, rad=0.2, color='#2B4C7E'):
    ax.annotate("",
                xy=xy_to, xycoords='data',
                xytext=xy_from, textcoords='data',
                arrowprops=dict(arrowstyle="-|>,head_length=0.5,head_width=0.2", 
                                color=color,
                                lw=1.8,
                                alpha=0.9,
                                connectionstyle=f"arc3,rad={rad}"))

# 连接对应的物理位置
pos_liDAR_robot = (2.2, 4.0)    
pos_liDAR_box   = (x_sub - box_width/2 - 0.1, y_subs[0])

pos_jetson_robot = (2.4, 2.7)   
pos_jetson_box   = (x_sub - box_width/2 - 0.1, y_subs[1])

pos_chassis_robot = (2.6, 1.4)  
pos_chassis_box   = (x_sub - box_width/2 - 0.1, y_subs[2])

draw_curved_connector(ax, pos_liDAR_robot, pos_liDAR_box, rad=-0.25, color=subs_info[0]['line'])
draw_curved_connector(ax, pos_jetson_robot, pos_jetson_box, rad=0.1, color=subs_info[1]['line'])
draw_curved_connector(ax, pos_chassis_robot, pos_chassis_box, rad=0.3, color=subs_info[2]['line'])

out_pdf = "fig2_platform_beautiful.pdf"
out_png = "fig2_platform_beautiful_rendered.png"

plt.tight_layout(pad=0)
plt.savefig(out_pdf, format='pdf', bbox_inches='tight', transparent=False, facecolor='#FFFFFF')
plt.savefig(out_png, format='png', bbox_inches='tight', dpi=300, transparent=False, facecolor='#FFFFFF')
print(f"Generated: {out_png}")
