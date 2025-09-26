import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
import matplotlib
import matplotlib.pyplot as plt

# 设置matplotlib支持中文显示
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']  # 用来正常显示中文标签
matplotlib.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
from Shape_generate import Shape

class ShapeCNN(nn.Module):
    """CNN模型用于形状识别"""
    def __init__(self, num_classes=4):
        super(ShapeCNN, self).__init__()
        
        # 卷积层
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        # 池化层
        self.pool = nn.MaxPool2d(2, 2)
        
        # 全连接层
        self.fc1 = nn.Linear(128 * 12 * 12, 512)  # 100x100 -> 50x50 -> 25x25 -> 12x12
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, num_classes)
        
        # Dropout层防止过拟合
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        # 卷积层 + 激活函数 + 池化
        x = self.pool(F.relu(self.conv1(x)))  # [batch, 1, 100, 100] -> [batch, 32, 50, 50]
        x = self.pool(F.relu(self.conv2(x)))  # [batch, 32, 50, 50] -> [batch, 64, 25, 25]
        x = self.pool(F.relu(self.conv3(x)))  # [batch, 64, 25, 25] -> [batch, 128, 12, 12]
        
        # 展平
        x = x.view(-1, 128 * 12 * 12)
        
        # 全连接层
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x

class Shape_classify:
    def __init__(self, model_path=None):
        """初始化形状分类器"""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = ShapeCNN(num_classes=4).to(self.device)
        self.class_names = ['hole', 'stain', 'scratch', 'damage']
        
        # 如果提供了模型路径，加载预训练模型
        if model_path:
            self.load_model(model_path)
    
    def prepare_data(self, num_samples_per_class=500):
        """准备训练数据"""
        print("正在生成训练数据...")
        
        # 生成各类形状数据
        hole_generator = Shape(num_samples_per_class, 0, 100)
        hole_data = hole_generator.generate_hole()
        
        stain_generator = Shape(num_samples_per_class, 1, 100)
        stain_data = stain_generator.generate_stain()
        
        scratch_generator = Shape(num_samples_per_class, 2, 100)
        scratch_data = scratch_generator.generate_scratch()
        
        damage_generator = Shape(num_samples_per_class, 3, 100)
        damage_data = damage_generator.generate_damage()
        
        # 合并数据
        X = torch.cat([hole_data, stain_data, scratch_data, damage_data], dim=0)
        
        # 创建标签
        y = torch.cat([
            torch.zeros(num_samples_per_class, dtype=torch.long),  # hole
            torch.ones(num_samples_per_class, dtype=torch.long),   # stain
            torch.full((num_samples_per_class,), 2, dtype=torch.long),  # scratch
            torch.full((num_samples_per_class,), 3, dtype=torch.long)   # damage
        ], dim=0)
        
        # 转换为float并添加通道维度
        X = X.float().unsqueeze(1)  # [N, 1, H, W]
        
        # 数据归一化
        X = X * 2.0 - 1.0  # 将[0,1]映射到[-1,1]
        
        print(f"数据形状: {X.shape}")
        print(f"标签形状: {y.shape}")
        
        return X, y
    
    def train_model(self, num_epochs=50, batch_size=32, learning_rate=0.001):
        """训练CNN模型"""
        print("开始训练模型...")
        
        # 准备数据
        X, y = self.prepare_data(num_samples_per_class=300)
        
        # 划分训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # 创建数据加载器
        train_dataset = TensorDataset(X_train, y_train)
        test_dataset = TensorDataset(X_test, y_test)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        # 定义损失函数和优化器
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5)
        
        # 训练历史
        train_losses = []
        train_accuracies = []
        test_accuracies = []
        
        for epoch in range(num_epochs):
            # 训练阶段
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_total += batch_y.size(0)
                train_correct += (predicted == batch_y).sum().item()
            
            # 测试阶段
            self.model.eval()
            test_correct = 0
            test_total = 0
            
            with torch.no_grad():
                for batch_X, batch_y in test_loader:
                    batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                    outputs = self.model(batch_X)
                    _, predicted = torch.max(outputs.data, 1)
                    test_total += batch_y.size(0)
                    test_correct += (predicted == batch_y).sum().item()
            
            # 记录指标
            train_loss = train_loss / len(train_loader)
            train_acc = 100 * train_correct / train_total
            test_acc = 100 * test_correct / test_total
            
            train_losses.append(train_loss)
            train_accuracies.append(train_acc)
            test_accuracies.append(test_acc)
            
            scheduler.step()
            
            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], '
                      f'Train Loss: {train_loss:.4f}, '
                      f'Train Acc: {train_acc:.2f}%, '
                      f'Test Acc: {test_acc:.2f}%')
        
        print("训练完成!")
        
        # 准备训练历史数据
        training_history = {
            'train_losses': train_losses,
            'train_accuracies': train_accuracies,
            'test_accuracies': test_accuracies,
            'num_epochs': num_epochs,
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'final_train_acc': train_accuracies[-1] if train_accuracies else 0,
            'final_test_acc': test_accuracies[-1] if test_accuracies else 0
        }
        
        # 绘制训练曲线
        self.plot_training_history(train_losses, train_accuracies, test_accuracies)
        
        return training_history
    
    def plot_training_history(self, train_losses, train_accuracies, test_accuracies):
        """绘制训练历史"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # 损失曲线
        ax1.plot(train_losses)
        ax1.set_title('训练损失')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.grid(True)
        
        # 准确率曲线
        ax2.plot(train_accuracies, label='训练准确率')
        ax2.plot(test_accuracies, label='测试准确率')
        ax2.set_title('准确率')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def classify(self, image_tensor):
        """对单个图像进行分类"""
        self.model.eval()
        
        # 确保输入格式正确
        if len(image_tensor.shape) == 2:
            image_tensor = image_tensor.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
        elif len(image_tensor.shape) == 3:
            image_tensor = image_tensor.unsqueeze(0)  # [1, 1, H, W]
        
        # 数据预处理
        image_tensor = image_tensor.float()
        image_tensor = image_tensor * 2.0 - 1.0  # 归一化到[-1,1]
        image_tensor = image_tensor.to(self.device)
        
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = F.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            
        predicted_class = predicted.item()
        confidence = probabilities[0][predicted_class].item()
        
        return {
            'predicted_class': self.class_names[predicted_class],
            'class_id': predicted_class,
            'confidence': confidence,
            'all_probabilities': probabilities[0].cpu().numpy()
        }
    
    def evaluate_model(self, test_data=None):
        """评估模型性能"""
        if test_data is None:
            # 生成测试数据
            X_test, y_test = self.prepare_data(num_samples_per_class=100)
        else:
            X_test, y_test = test_data
        
        self.model.eval()
        correct = 0
        total = 0
        class_correct = [0] * 4
        class_total = [0] * 4
        
        with torch.no_grad():
            for i in range(len(X_test)):
                image = X_test[i:i+1].to(self.device)
                label = y_test[i].item()
                
                outputs = self.model(image)
                _, predicted = torch.max(outputs, 1)
                predicted_class = predicted.item()
                
                total += 1
                class_total[label] += 1
                
                if predicted_class == label:
                    correct += 1
                    class_correct[label] += 1
        
        overall_accuracy = 100 * correct / total
        print(f"总体准确率: {overall_accuracy:.2f}%")
        
        for i, class_name in enumerate(self.class_names):
            if class_total[i] > 0:
                accuracy = 100 * class_correct[i] / class_total[i]
                print(f"{class_name} 准确率: {accuracy:.2f}% ({class_correct[i]}/{class_total[i]})")
        
        return overall_accuracy
    
    def save_model(self, filepath, training_history=None, model_info=None):
        """保存模型及其相关信息"""
        import datetime
        
        # 准备保存的数据
        save_data = {
            'model_state_dict': self.model.state_dict(),
            'class_names': self.class_names,
            'model_architecture': {
                'num_classes': len(self.class_names),
                'input_size': (1, 100, 100),
                'device': str(self.device)
            },
            'save_time': datetime.datetime.now().isoformat(),
            'model_info': model_info or {}
        }
        
        # 如果有训练历史，也保存
        if training_history:
            save_data['training_history'] = training_history
        
        torch.save(save_data, filepath)
        print(f"模型已保存到: {filepath}")
        print(f"保存时间: {save_data['save_time']}")
        
        # 同时保存一个备份文件
        backup_path = filepath.replace('.pth', '_backup.pth')
        torch.save(save_data, backup_path)
        print(f"备份文件已保存到: {backup_path}")
    
    def load_model(self, filepath, strict=True):
        """加载模型"""
        try:
            checkpoint = torch.load(filepath, map_location=self.device)
            
            # 加载模型权重
            if strict:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                # 非严格模式，允许部分权重不匹配
                model_dict = self.model.state_dict()
                pretrained_dict = checkpoint['model_state_dict']
                pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and model_dict[k].shape == v.shape}
                model_dict.update(pretrained_dict)
                self.model.load_state_dict(model_dict)
            
            # 加载其他信息
            self.class_names = checkpoint.get('class_names', self.class_names)
            
            # 显示加载信息
            print(f"模型已从 {filepath} 加载")
            if 'save_time' in checkpoint:
                print(f"模型保存时间: {checkpoint['save_time']}")
            if 'model_info' in checkpoint:
                info = checkpoint['model_info']
                if info:
                    print("模型信息:")
                    for key, value in info.items():
                        print(f"  {key}: {value}")
            
            return checkpoint
            
        except Exception as e:
            print(f"加载模型时出错: {e}")
            return None
    
    def demo_classification(self):
        """演示分类功能"""
        print("生成演示图像...")
        
        # 生成每种类型的示例图像
        generators = [
            Shape(1, 0, 100).generate_hole(),
            Shape(1, 1, 100).generate_stain(),
            Shape(1, 2, 100).generate_scratch(),
            Shape(1, 3, 100).generate_damage()
        ]
        
        fig, axes = plt.subplots(2, 2, figsize=(10, 10))
        axes = axes.ravel()
        
        for i, (generator, class_name) in enumerate(zip(generators, self.class_names)):
            image = generator[0]
            result = self.classify(image)
            
            axes[i].imshow(image.numpy(), cmap='gray')
            axes[i].set_title(f'真实: {class_name}\n预测: {result["predicted_class"]}\n置信度: {result["confidence"]:.3f}')
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def get_model_info(self):
        """获取模型信息"""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        return {
            'model_name': 'ShapeCNN',
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'num_classes': len(self.class_names),
            'class_names': self.class_names,
            'input_size': (1, 100, 100),
            'device': str(self.device)
        }
    
    def compare_models(self, model_path1, model_path2):
        """比较两个模型的性能"""
        print("=== 模型比较 ===")
        
        # 加载第一个模型
        classifier1 = Shape_classify()
        checkpoint1 = classifier1.load_model(model_path1)
        
        # 加载第二个模型
        classifier2 = Shape_classify()
        checkpoint2 = classifier2.load_model(model_path2)
        
        if not checkpoint1 or not checkpoint2:
            print("无法加载模型进行比较")
            return
        
        # 生成测试数据
        X_test, y_test = self.prepare_data(num_samples_per_class=50)
        
        # 测试两个模型
        results = []
        for classifier, name in [(classifier1, "模型1"), (classifier2, "模型2")]:
            correct = 0
            total = len(X_test)
            
            for i in range(total):
                result = classifier.classify(X_test[i])
                if result['class_id'] == y_test[i].item():
                    correct += 1
            
            accuracy = 100 * correct / total
            results.append((name, accuracy))
            print(f"{name} 准确率: {accuracy:.2f}%")
        
        # 显示比较结果
        if results[0][1] > results[1][1]:
            print(f"\n{results[0][0]} 性能更好，准确率高出 {results[0][1] - results[1][1]:.2f}%")
        elif results[1][1] > results[0][1]:
            print(f"\n{results[1][0]} 性能更好，准确率高出 {results[1][1] - results[0][1]:.2f}%")
        else:
            print("\n两个模型性能相同")
    
    def export_model_info(self, filepath):
        """导出模型信息到文件"""
        import json
        
        model_info = self.get_model_info()
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(model_info, f, indent=2, ensure_ascii=False)
        
        print(f"模型信息已导出到: {filepath}")
    
    def create_model_report(self, filepath=None):
        """创建模型报告"""
        import datetime
        
        model_info = self.get_model_info()
        
        report = f"""
CNN形状识别模型报告
==================
生成时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

模型基本信息:
- 模型名称: {model_info['model_name']}
- 总参数数量: {model_info['total_parameters']:,}
- 可训练参数数量: {model_info['trainable_parameters']:,}
- 输入尺寸: {model_info['input_size']}
- 类别数量: {model_info['num_classes']}
- 运行设备: {model_info['device']}

类别信息:
"""
        
        for i, class_name in enumerate(model_info['class_names']):
            report += f"- 类别 {i}: {class_name}\n"
        
        if filepath:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(report)
            print(f"模型报告已保存到: {filepath}")
        else:
            print(report)
        
        return report
    