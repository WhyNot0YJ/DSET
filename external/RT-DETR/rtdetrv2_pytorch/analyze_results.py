#!/usr/bin/env python3
"""
MOE RT-DETR训练结果分析脚本
分析训练结果并生成报告
"""

import os
import json
import torch
import numpy as np
from collections import defaultdict
import argparse

class MOEResultsAnalyzer:
    def __init__(self, checkpoint_path: str):
        self.checkpoint_path = checkpoint_path
        self.results = {}
        
    def load_checkpoint(self):
        """加载检查点"""
        if not os.path.exists(self.checkpoint_path):
            print(f"检查点文件不存在: {self.checkpoint_path}")
            return False
        
        try:
            checkpoint = torch.load(self.checkpoint_path, map_location='cpu')
            self.results = checkpoint
            return True
        except Exception as e:
            print(f"加载检查点失败: {e}")
            return False
    
    def analyze_expert_usage(self):
        """分析专家使用情况"""
        if 'expert_usage' not in self.results:
            print("没有找到专家使用数据")
            return
        
        expert_usage = self.results['expert_usage']
        print("\n" + "="*50)
        print("专家使用情况分析")
        print("="*50)
        
        for i, usage in enumerate(expert_usage):
            print(f"专家 {i}: 使用次数 {usage}")
        
        # 计算使用率
        total_usage = sum(expert_usage)
        if total_usage > 0:
            print(f"\n专家使用率:")
            for i, usage in enumerate(expert_usage):
                rate = usage / total_usage * 100
                print(f"专家 {i}: {rate:.2f}%")
        
        # 分析负载均衡
        if len(expert_usage) > 1:
            usage_std = np.std(expert_usage)
            usage_mean = np.mean(expert_usage)
            cv = usage_std / usage_mean if usage_mean > 0 else 0
            print(f"\n负载均衡分析:")
            print(f"使用次数标准差: {usage_std:.2f}")
            print(f"变异系数: {cv:.4f}")
            if cv < 0.1:
                print("负载均衡: 优秀")
            elif cv < 0.2:
                print("负载均衡: 良好")
            else:
                print("负载均衡: 需要改进")
    
    def analyze_losses(self):
        """分析损失情况"""
        print("\n" + "="*50)
        print("损失分析")
        print("="*50)
        
        if 'train_loss' in self.results:
            train_loss = self.results['train_loss']
            print(f"最终训练损失: {train_loss:.4f}")
        
        if 'val_loss' in self.results:
            val_loss = self.results['val_loss']
            print(f"最终验证损失: {val_loss:.4f}")
        
        if 'router_loss' in self.results:
            router_loss = self.results['router_loss']
            print(f"最终路由器损失: {router_loss:.4f}")
        
        if 'expert_losses' in self.results:
            expert_losses = self.results['expert_losses']
            print(f"\n专家损失:")
            for i, loss in enumerate(expert_losses):
                print(f"专家 {i}: {loss:.4f}")
    
    def analyze_accuracy(self):
        """分析准确率"""
        print("\n" + "="*50)
        print("准确率分析")
        print("="*50)
        
        if 'accuracy' in self.results:
            accuracy = self.results['accuracy']
            print(f"最终验证准确率: {accuracy:.4f} ({accuracy*100:.2f}%)")
        
        if 'correct_predictions' in self.results and 'total_predictions' in self.results:
            correct = self.results['correct_predictions']
            total = self.results['total_predictions']
            print(f"正确预测: {correct}/{total}")
    
    def analyze_routing_entropy(self):
        """分析路由熵"""
        print("\n" + "="*50)
        print("路由熵分析")
        print("="*50)
        
        if 'routing_entropy' in self.results:
            entropy = self.results['routing_entropy']
            print(f"最终路由熵: {entropy:.4f}")
            
            # 解释路由熵
            if entropy > 1.5:
                print("路由策略: 多样化（高熵）")
            elif entropy > 1.0:
                print("路由策略: 平衡")
            else:
                print("路由策略: 集中化（低熵）")
    
    def generate_report(self, output_path: str = None):
        """生成分析报告"""
        report = {
            'checkpoint_path': self.checkpoint_path,
            'analysis_time': str(np.datetime64('now')),
            'results': self.results
        }
        
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            print(f"\n分析报告已保存到: {output_path}")
        
        return report
    
    def run_analysis(self):
        """运行完整分析"""
        if not self.load_checkpoint():
            return
        
        print("MOE RT-DETR 训练结果分析")
        print("="*60)
        
        self.analyze_expert_usage()
        self.analyze_losses()
        self.analyze_accuracy()
        self.analyze_routing_entropy()
        
        # 生成总结
        print("\n" + "="*50)
        print("分析总结")
        print("="*50)
        
        if 'accuracy' in self.results:
            accuracy = self.results['accuracy']
            if accuracy > 0.8:
                print("模型性能: 优秀")
            elif accuracy > 0.6:
                print("模型性能: 良好")
            else:
                print("模型性能: 需要改进")
        
        if 'routing_entropy' in self.results:
            entropy = self.results['routing_entropy']
            if 1.0 < entropy < 1.5:
                print("路由策略: 平衡良好")
            else:
                print("路由策略: 需要调整")

def main():
    parser = argparse.ArgumentParser(description='MOE RT-DETR训练结果分析')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='检查点文件路径')
    parser.add_argument('--output', type=str, default=None,
                       help='输出报告文件路径')
    
    args = parser.parse_args()
    
    analyzer = MOEResultsAnalyzer(args.checkpoint)
    analyzer.run_analysis()
    
    if args.output:
        analyzer.generate_report(args.output)

if __name__ == "__main__":
    main()
