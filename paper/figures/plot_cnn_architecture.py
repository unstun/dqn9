#!/usr/bin/env python3
"""
PlotNeuralNet风格的CNN-PDDQN架构可视化脚本

生成高质量的神经网络架构图，适用于SCI论文。

输出:
    - cnn_pddqn_architecture.pdf (矢量图，推荐用于论文)
    - cnn_pddqn_architecture.png (位图，用于预览)

使用方法:
    python plot_cnn_architecture.py
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Polygon, Circle, Rectangle
from matplotlib.lines import Line2D
import numpy as np
from pathlib import Path


class CNNArchitectureVisualizer:
    """PlotNeuralNet风格的CNN架构可视化器"""
    
    def __init__(self, figsize=(16, 9), dpi=300):
        self.fig, self.ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)
        self.ax.set_xlim(0, 16)
        self.ax.set_ylim(0, 9)
        self.ax.axis('off')
        self.ax.set_aspect('equal')
        
        self.colors = {
            'input': '#E8E8E8',
            'input_border': '#666666',
            'conv': '#5DADE2',
            'conv_border': '#2471A3',
            'conv_top': '#85C1E9',
            'conv_side': '#5499C7',
            'fc': '#58D68D',
            'fc_border': '#1E8449',
            'value': '#E74C3C',
            'value_border': '#922B21',
            'advantage': '#27AE60',
            'advantage_border': '#1E8449',
            'output': '#9B59B6',
            'output_border': '#6C3483',
            'mask': '#F39C12',
            'mask_border': '#D68910',
            'concat': '#F4D03F',
            'concat_border': '#B7950B',
            'mha': '#17A2B8',
            'mha_border': '#117A8B',
            'arrow': '#333333',
        }
        
    def draw_3d_block(self, x, y, width, height, depth, label, 
                      face_color, border_color, top_color=None, side_color=None,
                      fontsize=9, alpha=0.9):
        """
        绘制3D效果的立方体块（类似PlotNeuralNet风格）
        
        Args:
            x, y: 前面左下角坐标
            width, height: 前面尺寸
            depth: 深度（3D效果）
            label: 标签文字
            face_color: 前面颜色
            border_color: 边框颜色
            top_color: 顶面颜色（默认自动计算）
            side_color: 侧面颜色（默认自动计算）
        """
        if top_color is None:
            top_color = self._lighten_color(face_color, 0.3)
        if side_color is None:
            side_color = self._darken_color(face_color, 0.2)
        
        dx, dy = depth * 0.6, depth * 0.6
        
        vertices_front = [
            (x, y),
            (x + width, y),
            (x + width, y + height),
            (x, y + height)
        ]
        
        vertices_top = [
            (x, y + height),
            (x + dx, y + height + dy),
            (x + width + dx, y + height + dy),
            (x + width, y + height)
        ]
        
        vertices_side = [
            (x + width, y),
            (x + width + dx, y + dy),
            (x + width + dx, y + height + dy),
            (x + width, y + height)
        ]
        
        front = Polygon(vertices_front, facecolor=face_color, 
                       edgecolor=border_color, linewidth=1.5, alpha=alpha)
        top = Polygon(vertices_top, facecolor=top_color,
                     edgecolor=border_color, linewidth=1, alpha=alpha)
        side = Polygon(vertices_side, facecolor=side_color,
                      edgecolor=border_color, linewidth=1, alpha=alpha)
        
        self.ax.add_patch(front)
        self.ax.add_patch(top)
        self.ax.add_patch(side)
        
        self.ax.text(x + width/2, y + height/2, label,
                    ha='center', va='center', fontsize=fontsize,
                    fontweight='bold', color='white' if self._is_dark(face_color) else 'black')
        
        return x + width/2, y + height/2
    
    def draw_flat_block(self, x, y, width, height, label, 
                        face_color, border_color, fontsize=9, alpha=0.9):
        """绘制扁平块（用于FC层、输入等）"""
        rect = FancyBboxPatch((x, y), width, height,
                              boxstyle="round,pad=0.02,rounding_size=0.1",
                              facecolor=face_color, edgecolor=border_color,
                              linewidth=1.5, alpha=alpha)
        self.ax.add_patch(rect)
        
        self.ax.text(x + width/2, y + height/2, label,
                    ha='center', va='center', fontsize=fontsize,
                    fontweight='bold', color='white' if self._is_dark(face_color) else 'black')
        
        return x + width/2, y + height/2
    
    def draw_input_maps(self, x, y):
        """绘制输入地图通道（3个12x12地图）"""
        map_size = 1.0
        gap = 0.15
        
        labels = ['Occ', 'GDF', 'EDT']
        colors = ['#D5D8DC', '#F5B041', '#5DADE2']
        
        for i, (label, color) in enumerate(zip(labels, colors)):
            cx = x + i * (map_size + gap)
            
            rect = FancyBboxPatch((cx, y), map_size, map_size,
                                  boxstyle="round,pad=0.01",
                                  facecolor=color, edgecolor='#333',
                                  linewidth=1, alpha=0.8)
            self.ax.add_patch(rect)
            
            for j in range(4):
                for k in range(4):
                    rx = cx + 0.1 + j * 0.2
                    ry = y + 0.1 + k * 0.2
                    if label == 'Occ':
                        circle = Circle((rx + 0.05, ry + 0.05), 0.03, 
                                       facecolor='#1C2833', alpha=0.7)
                        self.ax.add_patch(circle)
            
            self.ax.text(cx + map_size/2, y - 0.15, label,
                        ha='center', va='top', fontsize=7, color='#555')
        
        self.ax.text(x + 1.5, y - 0.4, '3×12×12', ha='center', fontsize=8, 
                    fontweight='bold', color='#333')
        
        return x + 1.5, y + 0.5
    
    def draw_scalar_input(self, x, y):
        """绘制标量输入向量"""
        width = 3.5
        height = 0.5
        
        cell_width = 0.3
        n_cells = 11
        
        for i in range(n_cells):
            cx = x + i * cell_width
            color = '#F9E79F' if i < 2 else ('#ABEBC6' if i < 4 else '#D6EAF8')
            rect = Rectangle((cx, y), cell_width - 0.02, height,
                            facecolor=color, edgecolor='#333', linewidth=0.5)
            self.ax.add_patch(rect)
        
        self.ax.text(x + width/2, y - 0.15, '11-d Scalar Features',
                    ha='center', va='top', fontsize=8, fontweight='bold', color='#333')
        
        return x + width/2, y + height/2
    
    def draw_arrow(self, start, end, color=None, linewidth=2, style='->'):
        """绘制箭头连接"""
        if color is None:
            color = self.colors['arrow']
        
        self.ax.annotate('', xy=end, xytext=start,
                        arrowprops=dict(arrowstyle=style, color=color,
                                       lw=linewidth, mutation_scale=15))
    
    def draw_concat_node(self, x, y, radius=0.2):
        """绘制拼接节点"""
        circle = Circle((x, y), radius, facecolor='white',
                        edgecolor=self.colors['concat_border'], linewidth=2)
        self.ax.add_patch(circle)
        self.ax.text(x, y, '⊕', ha='center', va='center', 
                    fontsize=14, fontweight='bold', color='#333')
        return x, y
    
    def draw_dueling_head(self, x, y):
        """绘制Dueling头部（Value和Advantage分支）"""
        value_x = x
        adv_x = x + 2.5
        
        self.draw_3d_block(value_x, y, 0.8, 2.0, 0.4,
                          'V(s)', self.colors['value'], self.colors['value_border'],
                          fontsize=10)
        
        self.ax.text(value_x + 0.4, y - 0.2, '1-d', ha='center', 
                    fontsize=7, color='#922B21')
        
        self.draw_3d_block(adv_x, y, 1.2, 2.0, 0.4,
                          'A(s,a)', self.colors['advantage'], self.colors['advantage_border'],
                          fontsize=10)
        
        self.ax.text(adv_x + 0.6, y - 0.2, '35-d', ha='center',
                    fontsize=7, color='#1E8449')
        
        return (value_x + 0.4, y + 1.0), (adv_x + 0.6, y + 1.0)
    
    def draw_action_mask(self, x, y):
        """绘制Action Mask可视化"""
        width = 2.0
        height = 1.2
        
        rect = FancyBboxPatch((x, y), width, height,
                              boxstyle="round,pad=0.02",
                              facecolor='#FEF9E7', edgecolor=self.colors['mask_border'],
                              linewidth=2)
        self.ax.add_patch(rect)
        
        bar_width = 0.12
        bar_gap = 0.08
        n_bars = 12
        
        for i in range(n_bars):
            bx = x + 0.15 + i * (bar_width + bar_gap)
            bar_height = 0.3 + np.random.random() * 0.5
            
            if i in [2, 5, 8]:
                color = '#E6B0AA'
                rect = Rectangle((bx, y + 0.2), bar_width, bar_height,
                                facecolor=color, edgecolor='#C0392B', linewidth=0.5)
                self.ax.add_patch(rect)
                self.ax.plot([bx, bx + bar_width], [y + 0.2, y + 0.2 + bar_height],
                           color='#C0392B', linewidth=1.5)
                self.ax.plot([bx, bx + bar_width], [y + 0.2 + bar_height, y + 0.2],
                           color='#C0392B', linewidth=1.5)
            else:
                color = '#82E0AA'
                rect = Rectangle((bx, y + 0.2), bar_width, bar_height,
                                facecolor=color, edgecolor='#27AE60', linewidth=0.5)
                self.ax.add_patch(rect)
        
        self.ax.text(x + width/2, y + height + 0.1, 'Action Mask ★',
                    ha='center', fontsize=9, fontweight='bold', color='#D68910')
        
        return x + width/2, y + height/2
    
    def _lighten_color(self, color, amount=0.5):
        """使颜色变亮"""
        import matplotlib.colors as mc
        import colorsys
        try:
            c = mc.cnames[color]
        except:
            c = color
        c = colorsys.rgb_to_hls(*mc.to_rgb(c))
        return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])
    
    def _darken_color(self, color, amount=0.5):
        """使颜色变暗"""
        import matplotlib.colors as mc
        import colorsys
        try:
            c = mc.cnames[color]
        except:
            c = color
        c = colorsys.rgb_to_hls(*mc.to_rgb(c))
        return colorsys.hls_to_rgb(c[0], c[1] * (1 - amount), c[2])
    
    def _is_dark(self, color):
        """判断颜色是否为深色"""
        import matplotlib.colors as mc
        rgb = mc.to_rgb(color)
        luminance = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]
        return luminance < 0.5
    
    def add_legend(self, x, y):
        """添加图例"""
        legend_items = [
            ('Conv Layer', self.colors['conv']),
            ('FC Layer', self.colors['fc']),
            ('Value V(s)', self.colors['value']),
            ('Advantage A(s,a)', self.colors['advantage']),
            ('Output', self.colors['output']),
        ]
        
        for i, (label, color) in enumerate(legend_items):
            lx = x + i * 2.0
            rect = Rectangle((lx, y), 0.3, 0.2, facecolor=color, edgecolor='#333', linewidth=0.5)
            self.ax.add_patch(rect)
            self.ax.text(lx + 0.4, y + 0.1, label, va='center', fontsize=7)
    
    def draw_full_architecture(self):
        """绘制完整的CNN-PDDQN架构"""
        
        self.ax.text(8, 8.5, 'CNN-PDDQN Architecture', ha='center', 
                    fontsize=14, fontweight='bold', color='#333')
        
        map_x, map_y = 0.5, 5.5
        map_center = self.draw_input_maps(map_x, map_y)
        
        scalar_x, scalar_y = 0.3, 3.5
        scalar_center = self.draw_scalar_input(scalar_x, scalar_y)
        
        conv_x = 4.0
        conv_y = 5.0
        
        conv1_center = self.draw_3d_block(conv_x, conv_y, 1.5, 1.5, 0.5,
                                          '32', self.colors['conv'], self.colors['conv_border'],
                                          fontsize=12)
        self.ax.text(conv_x + 0.75, conv_y - 0.3, 'Conv 3×3', ha='center', fontsize=7, color='#2471A3')
        self.ax.text(conv_x + 0.75, conv_y - 0.5, '12×12', ha='center', fontsize=6, color='#666')
        
        conv2_x = conv_x + 2.2
        conv2_y = conv_y + 0.2
        conv2_center = self.draw_3d_block(conv2_x, conv2_y, 1.2, 1.2, 0.5,
                                          '64', self.colors['conv'], self.colors['conv_border'],
                                          fontsize=11)
        self.ax.text(conv2_x + 0.6, conv2_y - 0.3, 'Conv 3×3', ha='center', fontsize=7, color='#2471A3')
        self.ax.text(conv2_x + 0.6, conv2_y - 0.5, 's=2, 6×6', ha='center', fontsize=6, color='#666')
        
        conv3_x = conv2_x + 1.8
        conv3_y = conv2_y + 0.2
        conv3_center = self.draw_3d_block(conv3_x, conv3_y, 0.9, 0.9, 0.5,
                                          '64', self.colors['conv'], self.colors['conv_border'],
                                          fontsize=10)
        self.ax.text(conv3_x + 0.45, conv3_y - 0.3, 'Conv 3×3', ha='center', fontsize=7, color='#2471A3')
        self.ax.text(conv3_x + 0.45, conv3_y - 0.5, 's=2, 3×3', ha='center', fontsize=6, color='#666')
        
        mha_x = conv3_x + 1.3
        mha_y = conv3_y + 0.15
        mha_center = self.draw_flat_block(mha_x, mha_y, 0.7, 0.6,
                                          'MHA\nh=4', self.colors['mha'], self.colors['mha_border'],
                                          fontsize=7)
        
        flatten_x = mha_x + 1.0
        flatten_y = mha_y
        flatten_center = self.draw_flat_block(flatten_x, flatten_y, 1.0, 0.4,
                                              'Flatten\n576-d', self.colors['conv'], self.colors['conv_border'],
                                              fontsize=7)
        
        self.draw_arrow(map_center, (conv_x + 0.2, conv_y + 0.75))
        self.draw_arrow((conv_x + 1.5, conv_y + 0.75), (conv2_x, conv2_y + 0.6))
        self.draw_arrow((conv2_x + 1.2, conv2_y + 0.6), (conv3_x, conv3_y + 0.45))
        self.draw_arrow((conv3_x + 0.9, conv3_y + 0.45), (mha_x, mha_y + 0.3))
        self.draw_arrow((mha_x + 0.7, mha_y + 0.3), (flatten_x, flatten_y + 0.2))
        
        fc_scalar_x = 4.5
        fc_scalar_y = 3.5
        fc_scalar_center = self.draw_flat_block(fc_scalar_x, fc_scalar_y, 1.2, 0.5,
                                                'FC 64', self.colors['fc'], self.colors['fc_border'],
                                                fontsize=8)
        self.draw_arrow(scalar_center, (fc_scalar_x, fc_scalar_y + 0.25))
        
        concat_x = 10.5
        concat_y = 4.5
        concat_center = self.draw_concat_node(concat_x, concat_y)
        
        self.draw_arrow((flatten_x + 1.0, flatten_y + 0.2), (concat_x, concat_y + 0.2))
        self.draw_arrow((fc_scalar_x + 1.2, fc_scalar_y + 0.25), (concat_x, concat_y - 0.2))
        
        shared_x = concat_x + 0.8
        shared_y = 4.3
        shared_center = self.draw_flat_block(shared_x, shared_y, 1.3, 0.5,
                                             'FC 256', self.colors['fc'], self.colors['fc_border'],
                                             fontsize=8)
        self.draw_arrow((concat_x + 0.2, concat_y), (shared_x, shared_y + 0.25))
        
        dueling_x = 12.5
        dueling_y = 3.0
        value_center, adv_center = self.draw_dueling_head(dueling_x, dueling_y)
        
        split_x = shared_x + 0.65
        split_y = shared_y + 0.25
        self.draw_arrow((shared_x + 1.3, split_y), (split_x + 0.3, dueling_y + 2.0))
        
        qagg_x = 12.5
        qagg_y = 1.5
        qagg_center = self.draw_flat_block(qagg_x, qagg_y, 2.0, 0.6,
                                           'Q = V + A - Ā\n35-d', self.colors['output'], 
                                           self.colors['output_border'],
                                           fontsize=8)
        
        self.draw_arrow((value_center[0], dueling_y), (qagg_x + 0.5, qagg_y + 0.6))
        self.draw_arrow((adv_center[0], dueling_y), (qagg_x + 1.5, qagg_y + 0.6))
        
        mask_x = 12.3
        mask_y = 0.2
        mask_center = self.draw_action_mask(mask_x, mask_y)
        
        self.draw_arrow((qagg_x + 1.0, qagg_y), (mask_x + 1.0, mask_y + 1.2))
        
        output_x = 15.0
        output_y = 0.5
        output_center = self.draw_flat_block(output_x, output_y, 0.8, 0.8,
                                             'a*\n7δ̇×5a', self.colors['output'], 
                                             self.colors['output_border'],
                                             fontsize=8)
        self.draw_arrow((mask_x + 2.0, mask_y + 0.6), (output_x, output_y + 0.4))
        
        self.add_legend(0.5, 0.3)
        
        self.ax.text(0.5, 8.0, '★ Key Components:', fontsize=9, fontweight='bold', color='#C0392B')
        self.ax.text(0.5, 7.7, '• Dual-branch: CNN (spatial) + FC (scalar)', fontsize=7, color='#555')
        self.ax.text(0.5, 7.5, '• Spatial MHA: multi-head attention', fontsize=7, color='#555')
        self.ax.text(0.5, 7.3, '• Dueling: V(s) + A(s,a) decomposition', fontsize=7, color='#555')
        self.ax.text(0.5, 7.1, '• Action Mask: kinematic feasibility', fontsize=7, color='#555')
    
    def save(self, output_dir='.', filename='cnn_pddqn_architecture'):
        """保存图像"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        pdf_path = output_path / f'{filename}.pdf'
        png_path = output_path / f'{filename}.png'
        
        self.fig.savefig(pdf_path, bbox_inches='tight', format='pdf')
        self.fig.savefig(png_path, bbox_inches='tight', dpi=300, format='png')
        
        print(f"✓ 已保存: {pdf_path}")
        print(f"✓ 已保存: {png_path}")
        
        return pdf_path, png_path


def main():
    """主函数"""
    print("=" * 60)
    print("CNN-PDDQN Architecture Visualizer")
    print("PlotNeuralNet风格 3D可视化")
    print("=" * 60)
    
    viz = CNNArchitectureVisualizer(figsize=(16, 9), dpi=300)
    viz.draw_full_architecture()
    
    output_dir = Path(__file__).parent
    pdf_path, png_path = viz.save(output_dir)
    
    print("\n" + "=" * 60)
    print("生成完成!")
    print(f"PDF (推荐用于论文): {pdf_path}")
    print(f"PNG (用于预览): {png_path}")
    print("=" * 60)
    
    plt.show()


if __name__ == '__main__':
    main()
