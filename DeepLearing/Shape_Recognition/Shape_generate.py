# -*- coding: utf-8 -*-
import torch
import numpy as np
import random
import math
class Shape:
    def __init__(self,num_samples,type=0,length=100):
        self.noise=np.random.randint(0,2)
        self.type=type
        self.length=length
        self.shapes=torch.zeros((num_samples,length,length))
        self.num_samples=num_samples
    def generate_hole(self):
        for i in range(self.num_samples):
            self.shapes[i]=self.hole()
        return self.shapes
    def generate_stain(self):
        for i in range(self.num_samples):
            self.shapes[i]=self.stain()
        return self.shapes
    def generate_scratch(self):
        for i in range(self.num_samples):
            self.shapes[i]=self.scratch()
        return self.shapes
    def generate_damage(self):
        for i in range(self.num_samples):
            self.shapes[i]=self.damage()
        return self.shapes
    
    def hole(self):
        type=random.randint(0,2)
        length=random.randint(10,self.length//2)
        x1,y1=random.randint(length//2,self.length-length//2),random.randint(length//2,self.length-length//2)
        
        if type==0:
            # 空心圆
            return self.hollow_circle(x1, y1, length)
        elif type==1:
            # 空心矩形
            return self.hollow_rectangle(x1, y1, length)
        elif type==2:
            # 空心三角形
            return self.hollow_triangle(x1, y1)
    
    def hollow_circle(self, center_x, center_y, radius):
        """创建空心圆，使用bool值表示黑白像素 - True=白色背景，False=黑色形状"""
        # 创建坐标网格
        y, x = torch.meshgrid(torch.arange(self.length), torch.arange(self.length), indexing='ij')
        
        # 计算到中心的距离
        distance = torch.sqrt((x - center_x)**2 + (y - center_y)**2)
        
        # 修复：确保内半径合理
        outer_radius = radius
        inner_radius = max(4, radius - 1)  # 确保内半径至少为1，厚度为2
        
        # 修复：创建空心圆逻辑
        # 空心圆：在外半径内但不在内半径内
        hollow_circle = (distance <= outer_radius) & (distance >= inner_radius)
        # 取反：背景白色(True)，形状黑色(False)
        hollow_circle = ~hollow_circle
        
        # 添加噪点
        hollow_circle = self.add_noise(hollow_circle)
        
        return hollow_circle
    
    def hollow_rectangle(self, center_x, center_y, size):
        """创建空心矩形 - True=白色背景，False=黑色形状"""

        canvas = torch.ones((self.length, self.length), dtype=torch.bool)  # 初始化为白色背景
        half_size = size // 2
        thickness = 1
        
        # 计算矩形边界
        left = max(0, center_x - half_size)
        right = min(self.length, center_x + half_size)
        top = max(0, center_y - half_size)
        bottom = min(self.length, center_y + half_size)
        self.draw_line(canvas, left, top, right, top)
        self.draw_line(canvas, right, top, right, bottom)
        self.draw_line(canvas, right, bottom, left, bottom)
        self.draw_line(canvas, left, bottom, left, top)
        canvas = self.add_noise(canvas)
        
        return canvas
    
    def hollow_triangle(self, x1, y1):
        """创建空心三角形 - True=白色背景，False=黑色形状"""
        canvas = torch.ones((self.length, self.length), dtype=torch.bool)  # 初始化为白色背景
        
        # 设置最小距离（可以根据需要调整）
        min_distance = 20
        
        # 生成第二个顶点，确保与第一个顶点有足够距离
        max_attempts = 10
        attempts = 0
        while attempts < max_attempts:
            x2 = random.randint(0, self.length-1)
            y2 = random.randint(0, self.length-1)
            distance1 = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            if distance1 >= min_distance:
                break
            attempts += 1
        
        # 如果无法找到合适的第二个点，使用默认位置
        if attempts >= max_attempts:
            x2 = (x1 + min_distance) % self.length
            y2 = (y1 + min_distance) % self.length
        
        # 生成第三个顶点，确保与前两个顶点都有足够距离
        attempts = 0
        while attempts < max_attempts:
            x3 = random.randint(0, self.length-1)
            y3 = random.randint(0, self.length-1)
            distance2 = math.sqrt((x3 - x1)**2 + (y3 - y1)**2)
            distance3 = math.sqrt((x3 - x2)**2 + (y3 - y2)**2)
            if distance2 >= min_distance and distance3 >= min_distance:
                break
            attempts += 1
        
        # 如果无法找到合适的第三个点，使用默认位置
        if attempts >= max_attempts:
            x3 = (x1 - min_distance) % self.length
            y3 = (y1 + min_distance) % self.length
        
        # 绘制三角形的三条边
        self.draw_line(canvas, x1, y1, x2, y2)
        self.draw_line(canvas, x2, y2, x3, y3)
        self.draw_line(canvas, x3, y3, x1, y1)
        
        # 添加噪点
        canvas = self.add_noise(canvas)
        
        return canvas
    
    def stain(self):
        canvas = torch.ones((self.length, self.length), dtype=torch.bool)  # 初始化为白色背景
        r = random.randint(self.length*0.1, self.length*0.75)
        center = random.randint(r//2, self.length-r//2)
        
        # 修复：确保切片操作不会越界
        start_y = max(0, center-r//2)
        end_y = min(self.length, center+r//2)
        start_x = max(0, center-r//2)
        end_x = min(self.length, center+r//2)
        
        canvas[start_y:end_y, start_x:end_x] = False
        canvas = self.add_noise(canvas)
        return canvas
    
    def scratch(self):
        canvas = torch.ones((self.length, self.length), dtype=torch.bool)  # 初始化为白色背景
        # 修复：确保初始坐标在有效范围内
        x = random.randint(3, self.length-4)  # 确保有足够的边界空间
        y = random.randint(3, self.length-4)
        length = random.randint(self.length*2, self.length*3)
        
        while length > 0:
            # 修复：使用elif确保逻辑正确
            if x < 3:
                x += 1
            elif x > self.length-4:
                x -= 1
            else:
                x += random.randint(-1, 1)
            
            if y < 3:
                y += 1
            elif y > self.length-4:
                y -= 1
            else:
                y += random.randint(-1, 1)
            
            # 修复：确保坐标在有效范围内
            x = max(0, min(self.length-1, x))
            y = max(0, min(self.length-1, y))
            
            canvas[y, x] = False  # 修复：注意坐标顺序
            length -= 1
            
        canvas = self.add_noise(canvas)
        return canvas
    
    def damage(self):
        canvas = torch.ones((self.length, self.length), dtype=torch.bool)  # 初始化为白色背景
        # 修复：确保坐标在有效范围内
        x1 = random.randint(0, self.length-1)
        y1 = random.randint(0, self.length-1)
        x2 = random.randint(0, self.length-1)
        y2 = random.randint(0, self.length-1)
        x3 = random.randint(0, self.length-1)
        y3 = random.randint(0, self.length-1)
        self.draw_line(canvas, x1, y1, x2, y2)
        self.draw_line(canvas, x2, y2, x3, y3)
        canvas = self.add_noise(canvas)
        return canvas
    
    def add_noise(self, canvas, noise_probability=0.01):
        """为图像添加噪点"""
        # 创建随机噪点掩码
        noise_mask = torch.rand(self.length, self.length) < noise_probability
        
        # 在噪点位置翻转像素值
        canvas = canvas ^ noise_mask
    
        return canvas
    
    def draw_line(self, canvas, x1, y1, x2, y2):
        """使用Bresenham算法绘制直线"""
        # 确保坐标在画布范围内
        x1, y1 = max(0, min(self.length-1, int(x1))), max(0, min(self.length-1, int(y1)))
        x2, y2 = max(0, min(self.length-1, int(x2))), max(0, min(self.length-1, int(y2)))
        
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        
        # 确定步进方向
        sx = 1 if x1 < x2 else -1
        sy = 1 if y1 < y2 else -1
        
        err = dx - dy
        
        x, y = x1, y1
        
        while True:
            # 绘制当前点
            canvas[y, x] = False
            
            # 检查是否到达终点
            if x == x2 and y == y2:
                break
            
            e2 = 2 * err
            
            if e2 > -dy:
                err -= dy
                x += sx
            
            if e2 < dx:
                err += dx
                y += sy
        return canvas