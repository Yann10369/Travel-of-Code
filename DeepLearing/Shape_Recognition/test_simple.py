# -*- coding: utf-8 -*-
"""
简化版CNN形状识别测试脚本
加载保存的模型并进行分类测试
"""

import torch
from Shape_classify import Shape_classify
from Shape_generate import Shape

def test_saved_model():
    """测试保存的模型"""
    print("=== 简化版CNN形状识别测试 ===")
    
    # 1. 加载模型
    print("1. 加载保存的模型...")
    classifier = Shape_classify()
    model_path = "simple_cnn_model.pth"
    
    try:
        checkpoint = classifier.load_model(model_path)
        if checkpoint is None:
            print("模型加载失败!")
            return
        saved_accuracy = checkpoint.get('training_history', {}).get('final_test_acc', 0)
    except FileNotFoundError:
        print(f"模型文件 {model_path} 不存在!")
        print("请先运行以下命令训练模型:")
        print("  python train_simple.py")
        print("\n或者运行完整演示:")
        print("  python simple_demo.py")
        return
    
    # 2. 测试分类功能
    print("\n2. 测试分类功能...")
    
    # 生成测试图像
    test_shapes = [
        ("hole", Shape(1, 0, 100).generate_hole()[0]),
        ("stain", Shape(1, 1, 100).generate_stain()[0]),
        ("scratch", Shape(1, 2, 100).generate_scratch()[0]),
        ("damage", Shape(1, 3, 100).generate_damage()[0])
    ]
    
    correct_predictions = 0
    total_predictions = len(test_shapes)
    
    print("\n分类结果:")
    print("-" * 50)
    
    for true_class, image in test_shapes:
        # 进行分类
        result = classifier.classify(image)
        predicted_class = result['predicted_class']
        confidence = result['confidence']
        
        # 检查预测是否正确
        is_correct = predicted_class == true_class
        if is_correct:
            correct_predictions += 1
        
        status = "✓" if is_correct else "✗"
        print(f"{status} 真实: {true_class:8} | 预测: {predicted_class:8} | 置信度: {confidence:.3f}")
    
    # 3. 显示测试结果
    test_accuracy = 100 * correct_predictions / total_predictions
    print("-" * 50)
    print(f"测试准确率: {test_accuracy:.1f}% ({correct_predictions}/{total_predictions})")
    print(f"保存时准确率: {saved_accuracy:.1f}%")
    
    # 4. 简单演示
    print("\n3. 随机形状分类演示...")
    import random
    
    for i in range(3):
        # 随机生成一个形状
        class_id = random.randint(0, 3)
        generator = Shape(1, class_id, 100)
        
        if class_id == 0:
            image = generator.generate_hole()[0]
        elif class_id == 1:
            image = generator.generate_stain()[0]
        elif class_id == 2:
            image = generator.generate_scratch()[0]
        else:
            image = generator.generate_damage()[0]
        
        result = classifier.classify(image)
        true_class = classifier.class_names[class_id]
        
        print(f"测试 {i+1}: 真实={true_class}, 预测={result['predicted_class']}, 置信度={result['confidence']:.3f}")

def quick_test():
    """快速测试功能"""
    print("\n=== 快速测试 ===")
    
    classifier = Shape_classify()
    
    try:
        # 尝试加载模型
        checkpoint = classifier.load_model("simple_cnn_model.pth")
        if checkpoint:
            saved_accuracy = checkpoint.get('training_history', {}).get('final_test_acc', 0)
            print("✓ 模型加载成功")
            print(f"✓ 模型准确率: {saved_accuracy:.2f}%")
            
            # 测试一个随机形状
            import random
            class_id = random.randint(0, 3)
            generator = Shape(1, class_id, 100)
            
            if class_id == 0:
                image = generator.generate_hole()[0]
            elif class_id == 1:
                image = generator.generate_stain()[0]
            elif class_id == 2:
                image = generator.generate_scratch()[0]
            else:
                image = generator.generate_damage()[0]
            
            result = classifier.classify(image)
            true_class = classifier.class_names[class_id]
            
            print(f"✓ 分类测试: 真实={true_class}, 预测={result['predicted_class']}")
            print("✓ 模型工作正常!")
            
        else:
            print("✗ 模型加载失败")
            
    except FileNotFoundError:
        print("✗ 模型文件不存在，请先训练模型")
    except Exception as e:
        print(f"✗ 测试过程中出错: {e}")

def main():
    """主函数"""
    # 完整测试
    test_saved_model()
    
    # 快速测试
    quick_test()

if __name__ == "__main__":
    main()
