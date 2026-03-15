import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import os

# 1. 字体与绘图设定
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'font.size': 11,
    'figure.dpi': 300,
})

fig_width = 8.0
fig_height = 4.5
fig, ax = plt.subplots(figsize=(fig_width, fig_height))
ax.axis('off')
ax.set_xlim(0, fig_width)
ax.set_ylim(0, fig_height)

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

# 3. 布局设计 (更紧凑、去掉突兀的底色框)
x_robot, y_robot = 2.0, 2.2
add_image(ax, img_robot, (x_robot, y_robot), zoom=0.055) 
ax.text(x_robot, 0.2, "Autonomous UGV Platform", ha='center', va='bottom', fontsize=12, fontweight='bold')

# 右侧三个子系统模块
x_sub = 5.6
y_subs = [3.6, 2.2, 0.8] 

subs_info = [
    {
        'img': img_lidar, 
        'title': '(a) Livox Mid-360 LiDAR', 
        'desc': '3D Point Cloud Sensing',
        'zoom': 0.045,
        'y': y_subs[0]
    },
    {
        'img': img_jetson, 
        'title': '(b) Jetson AGX Orin', 
        'desc': 'Onboard AI Computing',
        'zoom': 0.08,
        'y': y_subs[1]
    },
    {
        'img': img_chassis, 
        'title': '(c) Ackermann Chassis', 
        'desc': 'Bicycle Kinematic Model',
        'zoom': 0.09,
        'y': y_subs[2]
    }
]

for i, info in enumerate(subs_info):
    y_center = info['y']
    
    # 将字放在图片右侧，或调整样式
    # 你觉得原来的字在右侧不协调，我们现在尝试一种更像学术论文的结构图排版：
    # 上面黑体主标题，下面斜体小标题，最下面放图片
    ax.text(x_sub, y_center + 0.55, info['title'], ha='center', va='bottom', fontsize=11, fontweight='bold', color='black')
    ax.text(x_sub, y_center + 0.35, info['desc'], ha='center', va='bottom', fontsize=10, color='#444444', style='italic')
    
    # 图片放在稍微下方一点
    add_image(ax, info['img'], (x_sub, y_center - 0.2), zoom=info['zoom'])
    
    # 底部画一条淡淡的分隔线 (学术排版常见)
    if i < 2:
       ax.plot([x_sub - 1.2, x_sub + 1.2], [y_center - 0.7, y_center - 0.7], color='black', alpha=0.1, lw=1)


# 4. 绘制硬朗的学术引线 (直角折线)
def draw_bracket_connector(ax, xy_from, xy_to, x_elbow):
    """绘制直角连接：xy_from -> x_elbow -> xy_to"""
    # 先横向，再纵向，再横向。这种比较硬朗，适合硬件图
    line_x = [xy_from[0], x_elbow, x_elbow, xy_to[0]]
    line_y = [xy_from[1], xy_from[1], xy_to[1], xy_to[1]]
    ax.plot(line_x, line_y, color='#555555', lw=1.2, ls=':')
    # 在起点打一个圆点
    ax.plot(xy_from[0], xy_from[1], 'o', color='#555555', markersize=4)
    # 终点画个小箭头
    ax.annotate("", xy=xy_to, xytext=(x_elbow, xy_to[1]), 
                arrowprops=dict(arrowstyle="->", color='#555555', lw=1.2))

pos_liDAR_robot = (2.1, 3.4)    
pos_liDAR_box   = (x_sub - 1.2, y_subs[0] - 0.2)

pos_jetson_robot = (2.0, 2.4)   
pos_jetson_box   = (x_sub - 1.2, y_subs[1] - 0.2)

pos_chassis_robot = (2.0, 1.4)  
pos_chassis_box   = (x_sub - 1.2, y_subs[2] - 0.2)

# elbow 设定在它们中心偏右一点
elbow_x = 3.8
draw_bracket_connector(ax, pos_liDAR_robot, pos_liDAR_box, elbow_x)
draw_bracket_connector(ax, pos_jetson_robot, pos_jetson_box, elbow_x)
draw_bracket_connector(ax, pos_chassis_robot, pos_chassis_box, elbow_x)

out_pdf = "fig2_platform_academic.pdf"
out_png = "fig2_platform_academic_rendered.png"

plt.tight_layout(pad=0)
plt.savefig(out_pdf, format='pdf', bbox_inches='tight', transparent=True)
plt.savefig(out_png, format='png', bbox_inches='tight', dpi=300, transparent=False)
