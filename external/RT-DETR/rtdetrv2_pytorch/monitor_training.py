#!/usr/bin/env python3
"""
MOE RT-DETR训练监控脚本
实时监控训练进度和指标
"""

import os
import time
import re
import matplotlib.pyplot as plt
from collections import defaultdict
import argparse

class TrainingMonitor:
    def __init__(self, log_file_path: str):
        self.log_file_path = log_file_path
        self.metrics = defaultdict(list)
        self.epochs = []
        
    def parse_log_line(self, line: str):
        """解析日志行，提取指标"""
        # 解析epoch信息
        epoch_match = re.search(r'Epoch (\d+):', line)
        if epoch_match:
            epoch = int(epoch_match.group(1))
            self.epochs.append(epoch)
            return epoch, None
        
        # 解析训练损失
        train_loss_match = re.search(r'训练损失: ([\d.]+)', line)
        if train_loss_match:
            return None, ('train_loss', float(train_loss_match.group(1)))
        
        # 解析验证损失
        val_loss_match = re.search(r'验证损失: ([\d.]+)', line)
        if val_loss_match:
            return None, ('val_loss', float(val_loss_match.group(1)))
        
        # 解析验证准确率
        accuracy_match = re.search(r'验证准确率: ([\d.]+)', line)
        if accuracy_match:
            return None, ('accuracy', float(accuracy_match.group(1)))
        
        # 解析路由器损失
        router_loss_match = re.search(r'路由器损失: ([\d.]+)', line)
        if router_loss_match:
            return None, ('router_loss', float(router_loss_match.group(1)))
        
        # 解析路由熵
        entropy_match = re.search(r'路由熵: ([\d.]+)', line)
        if entropy_match:
            return None, ('routing_entropy', float(entropy_match.group(1)))
        
        return None, None
    
    def update_metrics(self):
        """更新指标"""
        if not os.path.exists(self.log_file_path):
            return
        
        with open(self.log_file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        current_epoch = None
        for line in lines:
            epoch, metric = self.parse_log_line(line)
            if epoch is not None:
                current_epoch = epoch
            elif metric is not None and current_epoch is not None:
                key, value = metric
                self.metrics[key].append(value)
    
    def plot_metrics(self, save_path: str = None):
        """绘制训练指标"""
        if not self.metrics:
            print("没有找到训练指标数据")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('MOE RT-DETR Training Metrics', fontsize=16)
        
        # 损失曲线
        if 'train_loss' in self.metrics and 'val_loss' in self.metrics:
            axes[0, 0].plot(self.metrics['train_loss'], label='Train Loss', color='blue')
            axes[0, 0].plot(self.metrics['val_loss'], label='Val Loss', color='red')
            axes[0, 0].set_title('Training and Validation Loss')
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].legend()
            axes[0, 0].grid(True)
        
        # 准确率曲线
        if 'accuracy' in self.metrics:
            axes[0, 1].plot(self.metrics['accuracy'], label='Accuracy', color='green')
            axes[0, 1].set_title('Validation Accuracy')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Accuracy')
            axes[0, 1].legend()
            axes[0, 1].grid(True)
        
        # 路由器损失
        if 'router_loss' in self.metrics:
            axes[1, 0].plot(self.metrics['router_loss'], label='Router Loss', color='orange')
            axes[1, 0].set_title('Router Loss')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Loss')
            axes[1, 0].legend()
            axes[1, 0].grid(True)
        
        # 路由熵
        if 'routing_entropy' in self.metrics:
            axes[1, 1].plot(self.metrics['routing_entropy'], label='Routing Entropy', color='purple')
            axes[1, 1].set_title('Routing Entropy')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Entropy')
            axes[1, 1].legend()
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"训练指标图已保存到: {save_path}")
        else:
            plt.show()
    
    def print_current_status(self):
        """打印当前训练状态"""
        if not self.metrics:
            print("没有找到训练指标数据")
            return
        
        print("\n" + "="*60)
        print("MOE RT-DETR 训练状态")
        print("="*60)
        
        if 'train_loss' in self.metrics:
            latest_train_loss = self.metrics['train_loss'][-1]
            print(f"最新训练损失: {latest_train_loss:.4f}")
        
        if 'val_loss' in self.metrics:
            latest_val_loss = self.metrics['val_loss'][-1]
            print(f"最新验证损失: {latest_val_loss:.4f}")
        
        if 'accuracy' in self.metrics:
            latest_accuracy = self.metrics['accuracy'][-1]
            print(f"最新验证准确率: {latest_accuracy:.4f}")
        
        if 'router_loss' in self.metrics:
            latest_router_loss = self.metrics['router_loss'][-1]
            print(f"最新路由器损失: {latest_router_loss:.4f}")
        
        if 'routing_entropy' in self.metrics:
            latest_entropy = self.metrics['routing_entropy'][-1]
            print(f"最新路由熵: {latest_entropy:.4f}")
        
        print(f"已完成的epoch数: {len(self.epochs)}")
        print("="*60)

def main():
    parser = argparse.ArgumentParser(description='MOE RT-DETR训练监控')
    parser.add_argument('--log_file', type=str, default='moe_training.log',
                       help='训练日志文件路径')
    parser.add_argument('--plot', action='store_true',
                       help='生成训练指标图表')
    parser.add_argument('--save_plot', type=str, default=None,
                       help='保存图表到指定路径')
    parser.add_argument('--watch', action='store_true',
                       help='实时监控训练进度')
    parser.add_argument('--interval', type=int, default=30,
                       help='监控间隔（秒）')
    
    args = parser.parse_args()
    
    monitor = TrainingMonitor(args.log_file)
    
    if args.watch:
        print(f"开始监控训练日志: {args.log_file}")
        print("按 Ctrl+C 停止监控")
        try:
            while True:
                monitor.update_metrics()
                monitor.print_current_status()
                time.sleep(args.interval)
        except KeyboardInterrupt:
            print("\n监控已停止")
    else:
        monitor.update_metrics()
        monitor.print_current_status()
        
        if args.plot:
            monitor.plot_metrics(args.save_plot)

if __name__ == "__main__":
    main()
