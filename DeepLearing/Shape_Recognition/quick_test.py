# -*- coding: utf-8 -*-
"""
快速测试简化版CNN功能
验证模型保存和加载是否正常工作
"""

import torch
from Shape_classify import Shape_classify
from Shape_generate import Shape

def test_simple_save_load():
    """测试简化的保存和加载功能"""
    print("=== 测试简化保存加载功能 ===")
    
    try:
        # 1. 创建分类器
        print("1. 创建分类器...")
        classifier = Shape_classify()
        
        # 2. 快速训练（仅用于测试）
        print("2. 快速训练模型...")
        print("   注意: 这是快速测试，只训练3个epoch")
        training_history = classifier.train_model(num_epochs=3, batch_size=8)
        
        # 3. 测试保存
        print("3. 测试保存功能...")
        model_info = {'description': '测试模型', 'final_accuracy': classifier.evaluate_model()}
        classifier.save_model("test_model.pth", training_history, model_info)
        
        # 4. 测试加载
        print("4. 测试加载功能...")
        new_classifier = Shape_classify()
        checkpoint = new_classifier.load_model("test_model.pth")
        saved_accuracy = checkpoint.get('training_history', {}).get('final_test_acc', 0) if checkpoint else None
        
        if saved_accuracy is not None:
            print(f"   ✓ 模型加载成功，准确率: {saved_accuracy:.2f}%")
        else:
            print("   ✗ 模型加载失败")
            return False
        
        # 5. 测试分类功能
        print("5. 测试分类功能...")
        test_image = Shape(1, 0, 100).generate_hole()[0]
        result = new_classifier.classify(test_image)
        
        print(f"   ✓ 分类成功: {result['predicted_class']} (置信度: {result['confidence']:.3f})")
        
        # 6. 清理测试文件
        import os
        if os.path.exists("test_model.pth"):
            os.remove("test_model.pth")
            print("6. 清理测试文件完成")
        
        print("\n✅ 所有测试通过！简化功能正常工作")
        return True
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        return False

def test_model_parameters():
    """测试模型参数保存"""
    print("\n=== 测试模型参数保存 ===")
    
    try:
        # 创建两个分类器
        classifier1 = Shape_classify()
        classifier2 = Shape_classify()
        
        # 训练第一个
        print("训练第一个模型...")
        classifier1.train_model(num_epochs=2, batch_size=4)
        
        # 保存第一个
        model_info = {'description': '临时模型'}
        classifier1.save_model("temp_model.pth", None, model_info)
        
        # 加载到第二个
        classifier2.load_model("temp_model.pth")
        
        # 比较参数是否相同
        params1 = list(classifier1.model.parameters())
        params2 = list(classifier2.model.parameters())
        
        all_same = True
        for p1, p2 in zip(params1, params2):
            if not torch.equal(p1, p2):
                all_same = False
                break
        
        if all_same:
            print("✅ 模型参数保存和加载正确")
        else:
            print("❌ 模型参数保存或加载有问题")
        
        # 清理
        import os
        if os.path.exists("temp_model.pth"):
            os.remove("temp_model.pth")
        
        return all_same
        
    except Exception as e:
        print(f"❌ 参数测试失败: {e}")
        return False

def main():
    """主测试函数"""
    print("简化版CNN功能快速测试")
    print("=" * 40)
    
    # 测试基本功能
    test1_passed = test_simple_save_load()
    
    # 测试参数保存
    test2_passed = test_model_parameters()
    
    # 总结
    print("\n" + "=" * 40)
    if test1_passed and test2_passed:
        print("🎉 所有测试通过！简化版功能正常")
        print("\n可以使用以下脚本:")
        print("- python train_simple.py    # 训练模型")
        print("- python test_simple.py     # 测试模型")
        print("- python simple_demo.py     # 完整演示")
    else:
        print("❌ 部分测试失败，请检查代码")

if __name__ == "__main__":
    main()
