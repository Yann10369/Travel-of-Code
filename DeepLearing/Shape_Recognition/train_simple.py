# -*- coding: utf-8 -*-
"""
简化版CNN形状识别训练脚本
只保存模型参数和准确率，去除复杂的数据分析功能
"""

import torch
from Shape_classify import Shape_classify

def main():
    """简化的训练流程"""
    print("=== 简化版CNN形状识别训练 ===")
    
    # 1. 创建分类器
    print("1. 创建CNN分类器...")
    classifier = Shape_classify()
    print(f"   设备: {classifier.device}")
    print(f"   类别: {classifier.class_names}")
    
    # 2. 训练模型
    print("\n2. 开始训练模型...")
    print("   训练参数: 30轮, 批次大小32, 学习率0.001")
    
    # 只获取训练历史，不显示详细过程
    training_history = classifier.train_model(
        num_epochs=30,
        batch_size=32,
        learning_rate=0.001
    )
    
    # 3. 获取最终准确率
    print("\n3. 评估模型性能...")
    final_accuracy = classifier.evaluate_model()
    
    # 4. 保存模型
    print("\n4. 保存模型...")
    model_path = "simple_cnn_model.pth"
    model_info = {
        'description': '简化训练模型',
        'final_accuracy': final_accuracy,
        'training_params': {
            'num_epochs': 30,
            'batch_size': 32,
            'learning_rate': 0.001
        }
    }
    classifier.save_model(model_path, training_history, model_info)
    
    # 5. 验证保存的模型
    print("\n5. 验证保存的模型...")
    test_classifier = Shape_classify()
    checkpoint = test_classifier.load_model(model_path)
    saved_accuracy = checkpoint.get('training_history', {}).get('final_test_acc', 0) if checkpoint else 0
    
    if saved_accuracy:
        print(f"   保存时的准确率: {saved_accuracy:.2f}%")
        
        # 测试加载的模型
        test_accuracy = test_classifier.evaluate_model()
        print(f"   加载后测试准确率: {test_accuracy:.2f}%")
        
        # 简单演示分类功能
        print("\n6. 演示分类功能...")
        test_classifier.demo_classification()
        
        print(f"\n训练完成! 最终准确率: {final_accuracy:.2f}%")
        print(f"模型已保存到: {model_path}")
    else:
        print("模型保存或加载失败!")

if __name__ == "__main__":
    main()
