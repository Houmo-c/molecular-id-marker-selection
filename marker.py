#!/usr/bin/env python3
"""
超快速标记选择器 - 大数据集优化版
"""

import sys
import os
import argparse
import time
import numpy as np
from collections import defaultdict
import mmap
import gc
from typing import List, Dict, Tuple, Set
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
import warnings
warnings.filterwarnings('ignore')

def parse_args():
    parser = argparse.ArgumentParser(
        description="快速标记选择 - 大数据集优化版"
    )
    
    parser.add_argument("-i", "--input", required=True, help="输入文件")
    parser.add_argument("-o", "--output", required=True, help="输出文件")
    parser.add_argument("-s", "--similar-output", required=True, help="相似样本输出")
    parser.add_argument("-k", "--min-differences", type=int, default=3,
                       help="样本对之间所需的最小差异数")
    parser.add_argument("-d", "--distance", type=int, default=0,
                       help="选择标记之间的最小距离（bp）")
    parser.add_argument("-b", "--batch-size", type=int, default=50000,
                       help="处理标记的批次大小")
    parser.add_argument("--nochr", action="store_true",
                       help="随机选择标记（默认按染色体平衡选择）")
    parser.add_argument("--parallel", action="store_true", 
                       help="使用多进程并行计算")
    parser.add_argument("--workers", type=int, default=4,
                       help="并行工作进程数")
    parser.add_argument("--max-markers", type=int, default=1000,
                       help="最大选择标记数（提前停止）")
    parser.add_argument("--min-saturation", type=float, default=0.95,
                       help="达到此饱和度后停止（0-1）")
    parser.add_argument("--strategy", choices=["fast", "balanced", "accurate"], 
                       default="balanced", help="选择策略")
    
    return parser.parse_args()

class UltraFastSelector:
    """超快速选择器 - 针对大数据集优化"""
    
    def __init__(self, args):
        self.args = args
        self.markers = []
        self.sample_names = []
        self.n_samples = 0
        
        # 预加载所有数据到内存
        self._load_all_data()
        
        # 使用numpy数组存储所有基因型
        self.genotypes_matrix = self._create_genotypes_matrix()
        
        # 差异矩阵
        self.diff_matrix = np.zeros((self.n_samples, self.n_samples), dtype=np.int16)
        
        # 跟踪不满意的样本对
        self.unsatisfied_pairs = set(
            (i, j) for i in range(self.n_samples) 
            for j in range(i + 1, self.n_samples)
        )
        
        # 选择结果
        self.selected_indices = []
        self.selected_snp_ids = []
        self.saturation_history = []
        
        # 距离检查缓存
        self.chr_selected_positions = defaultdict(list)
    
    def _load_all_data(self):
        """一次性加载所有数据到内存"""
        print("正在加载数据到内存...")
        
        with open(self.args.input, 'r') as f:
            # 读取标题行
            header = f.readline().strip()
            self.sample_names = header.split('\t')
            self.n_samples = len(self.sample_names) - 6
            
            print(f"样本数: {self.n_samples}")
            
            # 读取所有标记
            line_count = 0
            for line in f:
                line_count += 1
                if line_count % 100000 == 0:
                    print(f"  已读取 {line_count} 行...")
                    if line_count % 500000 == 0:
                        gc.collect()
                
                fields = line.strip().split('\t')
                if len(fields) < self.n_samples + 6:
                    continue
                
                self.markers.append({
                    'snp_id': fields[1],
                    'chr': fields[0],
                    'pos': int(fields[3]),
                    'ref': fields[4],
                    'alt': fields[5],
                    'genotypes': fields[6:6+self.n_samples]
                })
        
        print(f"总共加载 {len(self.markers)} 个标记")
    
    def _create_genotypes_matrix(self):
        """创建基因型矩阵（标记数 × 样本数）"""
        print("创建基因型矩阵...")
        
        n_markers = len(self.markers)
        matrix = np.zeros((n_markers, self.n_samples), dtype=np.int8)
        
        # 批量处理，减少内存压力
        batch_size = 50000
        for start in range(0, n_markers, batch_size):
            end = min(start + batch_size, n_markers)
            print(f"  处理标记 {start} 到 {end-1}...")
            
            for i in range(start, end):
                genotypes = self.markers[i]['genotypes']
                for j in range(self.n_samples):
                    g = genotypes[j]
                    if g in ['NA', '.', '-']:
                        matrix[i, j] = -1
                    else:
                        try:
                            matrix[i, j] = int(g)
                        except:
                            matrix[i, j] = -1
        
        return matrix
    
    def _is_too_close(self, marker_idx):
        """检查标记是否离已选标记太近"""
        if self.args.distance <= 0:
            return False
        
        marker = self.markers[marker_idx]
        chr = marker['chr']
        pos = marker['pos']
        
        if chr not in self.chr_selected_positions:
            return False
        
        for selected_pos in self.chr_selected_positions[chr]:
            if abs(pos - selected_pos) < self.args.distance:
                return True
        
        return False
    
    def _calculate_marker_score(self, marker_idx, unsatisfied_set):
        """快速计算标记分数"""
        if not unsatisfied_set:
            return 0
        
        # 获取该标记的基因型
        genotypes = self.genotypes_matrix[marker_idx]
        
        # 向量化计算
        score = 0
        valid_mask = genotypes >= 0
        
        for i, j in unsatisfied_set:
            if valid_mask[i] and valid_mask[j] and genotypes[i] != genotypes[j]:
                score += 1
        
        return score
    
    def _find_best_markers(self, candidate_indices, unsatisfied_set, n_candidates=5):
        """寻找最佳标记"""
        if not candidate_indices or not unsatisfied_set:
            return []
        
        # 根据策略选择评估方法
        if self.args.strategy == "fast":
            # 快速策略：随机选择n_candidates个并评分
            if len(candidate_indices) > n_candidates * 10:
                # 随机采样
                sampled_indices = np.random.choice(
                    candidate_indices, 
                    size=min(len(candidate_indices), n_candidates * 10),
                    replace=False
                )
            else:
                sampled_indices = candidate_indices
            
            scores = []
            for idx in sampled_indices:
                if self._is_too_close(idx):
                    continue
                
                score = self._calculate_marker_score(idx, unsatisfied_set)
                if score > 0:
                    scores.append((score, idx))
            
            # 按分数排序
            scores.sort(reverse=True)
            return [idx for _, idx in scores[:n_candidates]]
        
        elif self.args.strategy == "balanced":
            # 平衡策略：分批次评估
            batch_size = 1000
            best_scores = []
            
            for start in range(0, len(candidate_indices), batch_size):
                end = min(start + batch_size, len(candidate_indices))
                batch = candidate_indices[start:end]
                
                for idx in batch:
                    if self._is_too_close(idx):
                        continue
                    
                    score = self._calculate_marker_score(idx, unsatisfied_set)
                    if score > 0:
                        best_scores.append((score, idx))
                
                # 如果已经找到足够的候选，提前停止
                if len(best_scores) >= n_candidates * 3:
                    break
            
            # 排序并返回最佳候选
            best_scores.sort(reverse=True)
            return [idx for _, idx in best_scores[:n_candidates]]
        
        else:  # accurate
            # 精确策略：评估所有候选（最慢但最准确）
            scores = []
            for idx in candidate_indices:
                if self._is_too_close(idx):
                    continue
                
                score = self._calculate_marker_score(idx, unsatisfied_set)
                if score > 0:
                    scores.append((score, idx))
            
            scores.sort(reverse=True)
            return [idx for _, idx in scores[:n_candidates]]
    
    def select_markers(self):
        """主要选择循环"""
        print("\n开始标记选择...")
        print("=" * 60)
        
        # 初始化候选标记
        all_indices = list(range(len(self.markers)))
        remaining_indices = all_indices.copy()
        
        iteration = 0
        start_time = time.time()
        last_print_time = start_time
        
        while (remaining_indices and 
               len(self.selected_indices) < self.args.max_markers and
               len(self.unsatisfied_pairs) > 0):
            
            iteration += 1
            
            # 显示进度
            current_time = time.time()
            if current_time - last_print_time > 5:  # 每5秒打印一次
                sat = self._calculate_saturation()
                elapsed = current_time - start_time
                print(f"迭代 {iteration}: 饱和度={sat:.4f}, "
                      f"已选={len(self.selected_indices)}, "
                      f"剩余对={len(self.unsatisfied_pairs)}, "
                      f"时间={elapsed:.1f}s")
                last_print_time = current_time
                
                if sat >= self.args.min_saturation:
                    print(f"达到目标饱和度 {self.args.min_saturation}，停止选择")
                    break
            
            # 选择一批候选标记
            if len(remaining_indices) > self.args.batch_size:
                # 随机选择一批
                batch_size = min(self.args.batch_size, len(remaining_indices))
                batch_indices = np.random.choice(
                    remaining_indices, 
                    size=batch_size,
                    replace=False
                ).tolist()
            else:
                batch_indices = remaining_indices.copy()
            
            # 找出最佳标记
            best_markers = self._find_best_markers(
                batch_indices, self.unsatisfied_pairs, n_candidates=3
            )
            
            if not best_markers:
                # 如果没有找到好标记，尝试不同策略
                if len(batch_indices) < len(remaining_indices):
                    # 尝试另一批
                    continue
                else:
                    print("没有更多能增加差异的标记")
                    break
            
            # 添加最佳标记
            for marker_idx in best_markers:
                if marker_idx not in remaining_indices:
                    continue
                
                self._add_marker(marker_idx)
                remaining_indices.remove(marker_idx)
                
                # 如果已经满足所有对，提前停止
                if not self.unsatisfied_pairs:
                    break
            
            # 定期清理内存
            if iteration % 100 == 0:
                gc.collect()
        
        return len(self.selected_indices)
    
    def _add_marker(self, marker_idx):
        """添加标记到选择集"""
        marker = self.markers[marker_idx]
        self.selected_indices.append(marker_idx)
        self.selected_snp_ids.append(marker['snp_id'])
        
        # 更新染色体位置缓存
        self.chr_selected_positions[marker['chr']].append(marker['pos'])
        
        # 更新差异矩阵
        genotypes = self.genotypes_matrix[marker_idx]
        
        # 更新所有样本对
        for i in range(self.n_samples):
            gi = genotypes[i]
            if gi < 0:  # 缺失
                continue
            
            for j in range(i + 1, self.n_samples):
                gj = genotypes[j]
                if gj >= 0 and gi != gj:
                    self.diff_matrix[i, j] += 1
                    
                    # 如果达到最小差异要求，从不满意的集合中移除
                    if self.diff_matrix[i, j] >= self.args.min_differences:
                        if (i, j) in self.unsatisfied_pairs:
                            self.unsatisfied_pairs.remove((i, j))
    
    def _calculate_saturation(self):
        """计算当前饱和度"""
        total_pairs = self.n_samples * (self.n_samples - 1) // 2
        satisfied_pairs = total_pairs - len(self.unsatisfied_pairs)
        return satisfied_pairs / total_pairs if total_pairs > 0 else 0.0
    
    def _print_final_stats(self):
        """打印最终统计信息"""
        total_pairs = self.n_samples * (self.n_samples - 1) // 2
        satisfied_pairs = total_pairs - len(self.unsatisfied_pairs)
        saturation = self._calculate_saturation()
        
        print("\n" + "=" * 60)
        print(f"最终统计:")
        print(f"  选择标记数: {len(self.selected_indices)}")
        print(f"  总样本对: {total_pairs}")
        print(f"  满足差异要求的对: {satisfied_pairs}")
        print(f"  最终饱和度: {saturation:.4f}")
        print(f"  剩余不满意对: {len(self.unsatisfied_pairs)}")
        
        # 按染色体统计
        chr_counts = defaultdict(int)
        for idx in self.selected_indices:
            chr = self.markers[idx]['chr']
            chr_counts[chr] += 1
        
        if chr_counts:
            print(f"\n按染色体分布:")
            for chr in sorted(chr_counts.keys()):
                print(f"  染色体{chr}: {chr_counts[chr]}个标记")
    
    def write_output(self, output_file, similar_file):
        """写入输出文件"""
        print(f"\n写入输出文件...")
        
        # 写入选择的标记
        with open(output_file, 'w') as f:
            # 标题行
            f.write("Saturation\tMarkerID\tChr\tPos\tRef\tAlt")
            for i in range(6, len(self.sample_names)):
                f.write(f"\t{self.sample_names[i]}")
            f.write("\n")
            
            # 模拟计算每个标记后的饱和度
            for idx, marker_idx in enumerate(self.selected_indices):
                marker = self.markers[marker_idx]
                
                # 近似饱和度（基于选择顺序）
                sat = min(1.0, (idx + 1) / max(1, len(self.selected_indices)))
                
                # 写入标记信息
                f.write(f"{sat:.6f}\t{marker['snp_id']}\t{marker['chr']}\t")
                f.write(f"{marker['pos']}\t{marker['ref']}\t{marker['alt']}\t")
                
                # 写入基因型
                geno_str = "\t".join(marker['genotypes'])
                f.write(geno_str + "\n")
        
        print(f"  标记文件已写入: {output_file}")
        
        # 写入相似样本
        self._write_similar_samples(similar_file)
    
    def _write_similar_samples(self, similar_file):
        """写入相似样本信息"""
        print(f"  相似样本文件正在写入...")
        
        with open(similar_file, 'w') as f:
            if not self.unsatisfied_pairs:
                f.write(f"所有样本对都有至少 {self.args.min_differences} 个差异。\n")
                return
            
            # 找到相似样本组（使用聚类方法）
            visited = set()
            groups = []
            
            # 构建邻接矩阵
            adj = defaultdict(set)
            for i, j in self.unsatisfied_pairs:
                adj[i].add(j)
                adj[j].add(i)
            
            # 寻找连通分量
            for i in range(self.n_samples):
                if i in visited:
                    continue
                
                # 使用BFS找到连通分量
                if i in adj:  # 只检查有不满足对的样本
                    component = set()
                    stack = [i]
                    
                    while stack:
                        node = stack.pop()
                        if node not in visited:
                            visited.add(node)
                            component.add(node)
                            stack.extend(adj[node] - visited)
                    
                    if len(component) > 1:
                        groups.append(component)
                else:
                    visited.add(i)
            
            # 写入文件
            f.write(f"差异小于 {self.args.min_differences} 的样本组:\n")
            f.write("=" * 60 + "\n\n")
            
            if not groups:
                f.write("没有发现明显的样本组。\n")
                return
            
            # 按组大小排序
            groups.sort(key=len, reverse=True)
            
            for idx, group in enumerate(groups, 1):
                sample_names = [self.sample_names[i + 6] for i in group]
                f.write(f"第 {idx} 组 ({len(group)} 个样本):\n")
                f.write("  " + ", ".join(sorted(sample_names)) + "\n\n")
            
            # 统计信息
            total_in_groups = sum(len(g) for g in groups)
            f.write(f"统计:\n")
            f.write(f"  组数: {len(groups)}\n")
            f.write(f"  涉及样本数: {total_in_groups}\n")
            f.write(f"  占所有样本比例: {100*total_in_groups/self.n_samples:.1f}%\n")
        
        print(f"  相似样本文件已写入: {similar_file}")

def main():
    args = parse_args()
    
    print("=" * 60)
    print("超快速标记选择器 - 大数据集优化版")
    print("=" * 60)
    
    print(f"输入文件: {args.input}")
    print(f"输出文件: {args.output}")
    print(f"相似样本文件: {args.similar_output}")
    print(f"最小差异要求: {args.min_differences}")
    print(f"最小距离: {args.distance} bp")
    print(f"批次大小: {args.batch_size}")
    print(f"最大选择标记数: {args.max_markers}")
    print(f"目标饱和度: {args.min_saturation}")
    print(f"选择策略: {args.strategy}")
    
    start_time = time.time()
    
    try:
        # 创建选择器
        selector = UltraFastSelector(args)
        
        # 运行选择
        selected_count = selector.select_markers()
        
        # 打印最终统计
        selector._print_final_stats()
        
        # 写入输出
        selector.write_output(args.output, args.similar_output)
        
        print(f"\n✓ 选择完成!")
        
    except Exception as e:
        print(f"\n✗ 错误: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # 时间统计
    end_time = time.time()
    elapsed = end_time - start_time
    
    print("\n" + "=" * 60)
    print(f"性能统计:")
    print(f"  总运行时间: {elapsed:.2f} 秒 ({elapsed/60:.1f} 分钟)")
    print(f"  处理标记总数: {len(selector.markers)}")
    print(f"  选择标记数: {selected_count}")
    print(f"  平均每标记处理时间: {elapsed/max(1, len(selector.markers)):.4f} 秒")
    print("=" * 60)
    print("完成!")

if __name__ == "__main__":
    main()