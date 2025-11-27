#!/usr/bin/env python3
"""
分析max_det从300减少到100对mAP的影响
"""

def analyze_maxdet_impact():
    """分析检测框数量对mAP的影响"""
    
    print("="*80)
    print("检测框数量对mAP影响的理论分析")
    print("="*80)
    
    print("\n1. mAP计算原理\n")
    print("mAP@0.5:0.95的计算过程:")
    print("  - 对每个IoU阈值（0.5, 0.55, ..., 0.95）计算AP")
    print("  - AP = 曲线下面积（Precision-Recall曲线）")
    print("  - mAP = 所有IoU阈值AP的平均值")
    print("  - Precision = TP / (TP + FP)")
    print("  - Recall = TP / (TP + FN)")
    
    print("\n" + "="*80)
    print("2. 检测框数量对指标的影响\n")
    
    print("从300减少到100的影响:")
    print("\n【Recall（召回率）影响】")
    print("  ✓ 下降：")
    print("    - 可能漏检一些真实目标")
    print("    - 特别是在目标密集场景中（>100个目标）")
    print("    - 小目标和困难目标更容易被漏检")
    print("    - 在高IoU阈值（0.75-0.95）时影响更明显")
    print("  ✗ 不变：")
    print("    - 如果场景中目标数量 < 100，影响较小")
    
    print("\n【Precision（精确率）影响】")
    print("  ✓ 可能提升：")
    print("    - 只保留置信度最高的100个框")
    print("    - 可能减少一些低置信度的误检")
    print("  ✗ 可能下降：")
    print("    - 如果高置信度框中有误检，无法通过更多框来稀释")
    
    print("\n【mAP综合影响】")
    print("  ⚠️  通常mAP会下降，原因：")
    print("    1. Recall下降对mAP的影响通常大于Precision提升")
    print("    2. mAP@0.5:0.95对recall更敏感（特别是高IoU阈值）")
    print("    3. 在目标密集场景中，漏检会导致mAP显著下降")
    
    print("\n" + "="*80)
    print("3. 不同场景下的影响分析\n")
    
    print("场景1: 稀疏场景（每张图像 < 10个目标）")
    print("  - 影响：很小")
    print("  - 原因：100个框足够覆盖所有目标")
    print("  - mAP变化：可能略有下降（0-1%）")
    
    print("\n场景2: 中等密度（每张图像 10-50个目标）")
    print("  - 影响：中等")
    print("  - 原因：100个框通常足够，但可能漏检部分目标")
    print("  - mAP变化：可能下降（1-3%）")
    
    print("\n场景3: 密集场景（每张图像 > 50个目标）")
    print("  - 影响：较大")
    print("  - 原因：100个框可能无法覆盖所有目标")
    print("  - mAP变化：可能显著下降（3-5%或更多）")
    
    print("\n场景4: 非常密集（每张图像 > 100个目标）")
    print("  - 影响：很大")
    print("  - 原因：必然会有目标被漏检")
    print("  - mAP变化：可能显著下降（5%+）")
    
    print("\n" + "="*80)
    print("4. 不同IoU阈值下的影响\n")
    
    print("低IoU阈值（0.5-0.6）:")
    print("  - 影响：较小")
    print("  - 原因：对定位精度要求不高，主要看覆盖率")
    print("  - mAP@0.5可能下降较少")
    
    print("\n中IoU阈值（0.65-0.75）:")
    print("  - 影响：中等")
    print("  - 原因：需要较好的定位精度")
    print("  - mAP@0.75可能下降中等")
    
    print("\n高IoU阈值（0.75-0.95）:")
    print("  - 影响：较大")
    print("  - 原因：需要非常精确的定位，更多框意味着更多机会找到高IoU匹配")
    print("  - 这部分对mAP@0.5:0.95的影响最大")
    
    print("\n" + "="*80)
    print("5. 预期结果\n")
    
    print("基于DAIR-V2X数据集的特点（路测场景，可能有密集目标）:")
    print("  - 预期mAP@0.5:0.95会下降")
    print("  - 下降幅度：可能在2-5%之间")
    print("  - mAP@0.5可能下降较少（1-2%）")
    print("  - mAP@0.75和mAP@0.5:0.95下降更明显（2-5%）")
    
    print("\n具体预测（基于当前结果）:")
    print("  YOLOv8-L (max_det=300): mAP@0.5:0.95 = 0.6222")
    print("  YOLOv8-L (max_det=100): 预期 mAP@0.5:0.95 ≈ 0.59-0.60")
    print("  （下降约2-3%，与DETR系列更接近）")
    
    print("\n" + "="*80)
    print("6. 为什么这更公平？\n")
    
    print("公平性体现在：")
    print("  1. 检测框数量上限相同（100 vs 100）")
    print("  2. 在相同约束下比较模型性能")
    print("  3. 消除了检测框数量优势带来的不公平")
    
    print("\n但需要注意：")
    print("  - 这仍然不是完全公平的（候选框生成机制不同）")
    print("  - YOLOv8可以从更多候选框中选择100个")
    print("  - DETR系列固定只有100个查询")
    print("  - 但这是目前最接近的公平对比方式")
    
    print("\n" + "="*80)
    print("7. 建议\n")
    
    print("在论文中应该：")
    print("  1. 报告两种结果：")
    print("     - max_det=300（YOLOv8默认设置）")
    print("     - max_det=100（公平对比设置）")
    print("  2. 说明检测框数量对结果的影响")
    print("  3. 强调在相同约束下DSET的性能表现")
    print("  4. 讨论不同场景下的性能差异")
    
    print("\n" + "="*80)

if __name__ == '__main__':
    analyze_maxdet_impact()

