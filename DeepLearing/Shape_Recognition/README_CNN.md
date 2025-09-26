# CNN形状识别系统

这是一个基于卷积神经网络(CNN)的形状识别系统，能够识别四种不同类型的形状：hole（孔洞）、stain（污渍）、scratch（划痕）和damage（损坏）。

## 系统架构

### CNN模型结构
- **输入**: 100x100像素的灰度图像
- **卷积层**: 3层卷积层，分别有32、64、128个滤波器
- **池化层**: MaxPool2d(2,2)进行下采样
- **全连接层**: 3层全连接层，包含Dropout防止过拟合
- **输出**: 4个类别的概率分布

### 模型特点
- 使用ReLU激活函数
- Dropout正则化(0.5)
- Adam优化器
- 学习率调度器
- 交叉熵损失函数

## 文件说明

### 核心文件
- `Shape_classify.py`: 主要的CNN分类器类，包含模型定义、训练、评估和预测功能
- `Shape_generate.py`: 形状生成器，用于生成训练和测试数据

### 脚本文件
- `train_simple.py`: 简化训练脚本，用于训练CNN模型并保存核心信息
- `test_simple.py`: 简化测试脚本，用于评估模型性能
- `simple_demo.py`: 简化演示脚本，展示完整的训练→保存→加载→测试流程
- `quick_test.py`: 快速验证脚本，测试保存加载功能

## 使用方法

### 1. 快速演示
```bash
python simple_demo.py
```
这会运行一个快速演示，包括模型训练、保存、加载和分类展示。

### 2. 训练模型
```bash
python train_simple.py
```
这会进行模型训练，包括：
- 生成1200个训练样本（每类300个）
- 训练30个epoch
- 保存模型参数和准确率

### 3. 测试模型
```bash
python test_simple.py
```
这会加载训练好的模型并进行测试，包括：
- 加载保存的模型
- 测试分类功能
- 显示准确率

### 4. 快速验证
```bash
python quick_test.py
```
这会快速验证保存加载功能是否正常工作。

## 代码示例

### 基本使用
```python
from Shape_classify import Shape_classify

# 创建分类器
classifier = Shape_classify()

# 训练模型
classifier.train_model(num_epochs=30)

# 分类单个图像
result = classifier.classify(image_tensor)
print(f"预测类别: {result['predicted_class']}")
print(f"置信度: {result['confidence']:.3f}")
```

### 保存和加载模型
```python
# 训练并保存模型
classifier = Shape_classify()
training_history = classifier.train_model(num_epochs=30)
model_info = {'description': '我的模型', 'final_accuracy': classifier.evaluate_model()}
classifier.save_model("my_model.pth", training_history, model_info)

# 加载模型
new_classifier = Shape_classify()
checkpoint = new_classifier.load_model("my_model.pth")
if checkpoint:
    print(f"模型准确率: {checkpoint['training_history']['final_test_acc']:.2f}%")

# 进行分类
result = new_classifier.classify(image)
print(f"预测: {result['predicted_class']}")
```

## 模型性能

在标准测试集上的预期性能：
- 总体准确率: >90%
- 各类别准确率:
  - Hole: >85%
  - Stain: >90%
  - Scratch: >85%
  - Damage: >90%

## 依赖库

- torch: PyTorch深度学习框架
- torchvision: 图像处理工具
- numpy: 数值计算
- matplotlib: 数据可视化
- sklearn: 数据分割工具

## 安装依赖
```bash
pip install torch torchvision numpy matplotlib scikit-learn
```

## 模型保存和持久化

### 保存的内容
简化版只保存核心内容：
- **模型权重**: 训练好的神经网络参数
- **类别名称**: 形状类别列表
- **准确率**: 模型性能指标

### 保存的文件类型
- `.pth`: 模型文件（包含模型参数、类别名称、准确率）

### 保存功能
- **完整保存**: 使用 `save_model()` 方法保存模型权重、训练历史和元数据
- **完整加载**: 使用 `load_model()` 方法加载所有保存的信息
- **训练历史**: 保存完整的训练过程和学习曲线

## 注意事项

1. **GPU支持**: 如果系统有CUDA支持，模型会自动使用GPU加速训练
2. **内存要求**: 训练过程需要足够的内存来存储训练数据
3. **训练时间**: 完整训练可能需要几分钟到几十分钟，取决于硬件配置
4. **模型保存**: 简化版只保存模型参数和准确率，文件更小更简洁
5. **文件管理**: 模型文件为.pth格式，包含所有必要信息用于后续使用

## 扩展功能

### 添加新的形状类别
1. 在`Shape_generate.py`中添加新的形状生成函数
2. 在`Shape_classify.py`中更新`class_names`列表
3. 调整CNN模型的`num_classes`参数
4. 重新训练模型

### 调整模型架构
可以修改`ShapeCNN`类来：
- 增加更多的卷积层
- 调整滤波器数量
- 修改全连接层结构
- 添加批归一化层

## 故障排除

### 常见问题
1. **CUDA内存不足**: 减少batch_size或使用CPU
2. **训练不收敛**: 调整学习率或增加训练数据
3. **过拟合**: 增加Dropout比例或使用数据增强

### 调试技巧
- 使用`classifier.demo_classification()`可视化预测结果
- 使用`classifier.evaluate_model()`获取模型准确率
- 使用`python quick_test.py`快速验证保存加载功能
