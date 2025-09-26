# -*- coding: utf-8 -*-
"""
简化版CNN形状识别演示
展示简化的训练、保存、加载和使用流程
"""

import torch
from Shape_classify import Shape_classify
from Shape_generate import Shape

def train_and_save():
    """训练并保存模型"""
    print("=== 训练模型 ===")
    
    # 创建分类器
    classifier = Shape_classify()
    
    # 快速训练（演示用）
    print("开始训练...")
    training_history = classifier.train_model(
        num_epochs=15,  # 减少训练轮数用于快速演示
        batch_size=16,
        learning_rate=0.001
    )
    
    # 保存模型
    print("\n保存模型...")
    model_info = {'description': '演示模型', 'final_accuracy': classifier.evaluate_model()}
    classifier.save_model("demo_model.pth", training_history, model_info)
    
    return classifier

def load_and_test():
    """加载并测试模型"""
    print("\n=== 加载模型 ===")
    
    # 创建新的分类器实例
    classifier = Shape_classify()
    
    # 加载模型
    checkpoint = classifier.load_model("demo_model.pth")
    saved_accuracy = checkpoint.get('training_history', {}).get('final_test_acc', 0) if checkpoint else None
    
    if saved_accuracy is None:
        print("模型加载失败!")
        return None
    
    print(f"模型加载成功! 保存时准确率: {saved_accuracy:.2f}%")
    
    # 测试分类
    print("\n=== 分类测试 ===")
    
    # 生成每种类型的测试图像
    test_cases = [
        ("hole", Shape(1, 0, 100).generate_hole()[0]),
        ("stain", Shape(1, 1, 100).generate_stain()[0]),
        ("scratch", Shape(1, 2, 100).generate_scratch()[0]),
        ("damage", Shape(1, 3, 100).generate_damage()[0])
    ]
    
    correct = 0
    total = len(test_cases)
    
    for true_class, image in test_cases:
        result = classifier.classify(image)
        predicted = result['predicted_class']
        confidence = result['confidence']
        
        is_correct = predicted == true_class
        if is_correct:
            correct += 1
        
        status = "✓" if is_correct else "✗"
        print(f"{status} {true_class:8} -> {predicted:8} (置信度: {confidence:.3f})")
    
    test_accuracy = 100 * correct / total
    print(f"\n测试结果: {correct}/{total} 正确, 准确率: {test_accuracy:.1f}%")
    
    return classifier

def interactive_classification():
    """交互式分类演示"""
    print("\n=== 交互式分类 ===")
    
    classifier = Shape_classify()
    
    try:
        checkpoint = classifier.load_model("demo_model.pth")
        if not checkpoint:
            raise Exception("模型加载失败")
    except:
        print("模型文件不存在，请先运行训练")
        return
    
    print("输入数字选择要生成的形状类型:")
    print("0 - hole (孔洞)")
    print("1 - stain (污渍)")  
    print("2 - scratch (划痕)")
    print("3 - damage (损坏)")
    print("输入 'q' 退出")
    
    while True:
        choice = input("\n请选择 (0-3 或 q): ").strip()
        
        if choice.lower() == 'q':
            break
        
        try:
            class_id = int(choice)
            if 0 <= class_id <= 3:
                # 生成对应形状
                generator = Shape(1, class_id, 100)
                if class_id == 0:
                    image = generator.generate_hole()[0]
                elif class_id == 1:
                    image = generator.generate_stain()[0]
                elif class_id == 2:
                    image = generator.generate_scratch()[0]
                else:
                    image = generator.generate_damage()[0]
                
                # 分类
                result = classifier.classify(image)
                true_class = classifier.class_names[class_id]
                
                print(f"真实类型: {true_class}")
                print(f"预测类型: {result['predicted_class']}")
                print(f"置信度: {result['confidence']:.3f}")
                print(f"预测正确: {'是' if result['predicted_class'] == true_class else '否'}")
                
            else:
                print("请输入 0-3 之间的数字")
        except ValueError:
            print("请输入有效数字或 'q' 退出")

def main():
    """主演示流程"""
    print("简化版CNN形状识别演示")
    print("=" * 40)
    
    # 1. 训练并保存模型
    trained_classifier = train_and_save()
    
    # 2. 加载并测试模型
    loaded_classifier = load_and_test()
    
    # 3. 交互式演示
    if loaded_classifier:
        interactive_classification()
    
    print("\n演示完成!")
    print("\n使用说明:")
    print("1. 运行 python train_simple.py 进行完整训练")
    print("2. 运行 python test_simple.py 进行测试")
    print("3. 模型文件: demo_model.pth")

if __name__ == "__main__":
    main()
