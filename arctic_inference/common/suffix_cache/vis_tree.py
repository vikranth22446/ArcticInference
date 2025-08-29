#!/usr/bin/env python3
"""
Visualization tools for debugging suffix trees in speculative decoding
"""

import json
import pickle
from typing import Dict, List, Optional, Set
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle, FancyBboxPatch
import networkx as nx
from collections import deque
import argparse
import os


class SuffixTreeVisualizer:
    """Visualize suffix tree structure for debugging"""
    
    def __init__(self, suffix_cache=None, cache_file=None, tokenizer=None):
        """Initialize with either a SuffixCache object or a cache file"""
        if suffix_cache:
            self.suffix_cache = suffix_cache
            self.suffix_tree = suffix_cache._suffix_tree
        elif cache_file and os.path.exists(cache_file):
            # Load from pickle file
            with open(cache_file, 'rb') as f:
                cache_data = pickle.load(f)
                if hasattr(cache_data, '_suffix_tree'):
                    self.suffix_cache = cache_data
                    self.suffix_tree = cache_data._suffix_tree
                elif isinstance(cache_data, dict) and 'suffix_tree' in cache_data:
                    self.suffix_tree = cache_data['suffix_tree']
                    self.suffix_cache = None
                else:
                    raise ValueError("Could not find suffix tree in cache file")
        else:
            raise ValueError("Must provide either suffix_cache object or valid cache_file")
        
        self.tokenizer = tokenizer  # 支持在初始化时设置tokenizer
        
        # 检查是否支持调试萃取器
        self.has_extractor = self._check_extractor_support()
    
    def set_tokenizer(self, tokenizer):
        """Set tokenizer for decoding token IDs to text"""
        self.tokenizer = tokenizer
    
    def _check_extractor_support(self):
        """检查是否支持调试萃取器"""
        try:
            # 使用模块级函数检查调试功能是否可用
            from arctic_inference.common.suffix_cache._C import debug_visualization_available
            return debug_visualization_available()
        except ImportError:
            # 如果函数不存在，说明是旧版本，回退到属性检查
            try:
                return hasattr(self.suffix_tree, '_debug_extract_tree_info')
            except:
                return False
    
    def _use_extractor_if_available(self):
        """如果可用，使用萃取器获取真实树结构"""
        if not self.has_extractor:
            return None
        
        try:
            # 调用C++萃取器接口
            tree_info = self.suffix_tree._debug_extract_tree_info()
            return tree_info
        except AttributeError:
            # 萃取器不可用，降级到行为分析
            return None
    
    def print_tree_structure(self, max_depth: int = 16, show_probs: bool = True):
        """Print tree structure in text format"""
        print("\n" + "="*80)
        print("SUFFIX TREE STRUCTURE")
        print("="*80)
        
        # 尝试使用萃取器获取真实树结构
        tree_info = self._use_extractor_if_available()
        
        if tree_info:
            print("Using DEBUG_VISUALIZATION extractor - Full tree structure available")
            self._print_extracted_tree_structure(tree_info, max_depth, show_probs)
        else:
            print("C++ Suffix Tree (limited access to internal structure)")
            print("Using behavioral analysis - compile with DEBUG_VISUALIZATION for full structure")
            self._print_behavioral_tree_structure(max_depth)
    
    def _print_extracted_tree_structure(self, tree_info, max_depth: int = 5, show_probs: bool = True):
        """使用萃取器数据打印真实树结构"""
        print(f"Number of sequences: {len(tree_info.get('sequences', {}))}")
        print(f"Total nodes: {tree_info.get('total_nodes', 0)}")
        print(f"Max depth: {tree_info.get('max_depth', 0)}")
        
        nodes = tree_info.get('nodes', [])
        root_ptr = tree_info.get('root_ptr', 0)
        
        if not nodes:
            print("No node information available")
            return
        
        # 构建节点映射
        node_map = {node['node_ptr']: node for node in nodes}
        
        def print_node(node_ptr, prefix="", depth=0):
            if depth > max_depth or node_ptr not in node_map:
                return
            
            node = node_map[node_ptr]
            node_info = []
            
            node_info.append(f"Count: {node.get('count', 0)}")
            if node.get('length', 0) > 1:
                # 获取压缩token的具体内容
                compressed_tokens = self._get_compressed_tokens(node, tree_info)
                if compressed_tokens:
                    tokens_str = " ".join([self._token_to_str(t) for t in compressed_tokens])
                    # 限制显示长度，避免太长
                    if len(tokens_str) > 60:
                        tokens_str = tokens_str[:57] + "..."
                    node_info.append(f"Compressed: {node['length']} tokens [{tokens_str}]")
                else:
                    node_info.append(f"Compressed: {node['length']} tokens")
            if node.get('seq_id', -1) >= 0:
                node_info.append(f"Seq: {node['seq_id']}")
            
            # 打印节点
            if node_info:
                print(f"{prefix}{'└── ' if depth > 0 else ''}{' | '.join(node_info)}")
            
            # 打印子节点
            children = node.get('children', [])
            if children:
                sorted_children = sorted(children, key=lambda x: x[0])  # 按token_id排序
                
                for i, (token_id, child_ptr) in enumerate(sorted_children):
                    is_last = i == len(sorted_children) - 1
                    extension = "    " if is_last else "│   "
                    child_prefix = prefix + ("" if depth == 0 else extension)
                    
                    token_str = self._token_to_str(token_id)
                    print(f"{child_prefix}├── Token: {token_str}")
                    print_node(child_ptr, child_prefix + ("    " if is_last else "│   "), depth + 1)
        
        print(f"\nTree structure (max depth: {max_depth}):")
        print_node(root_ptr)
    
    def _print_behavioral_tree_structure(self, max_depth: int = 5):
        """使用行为分析打印树信息（降级方案）"""
        print(f"Number of sequences: {self.suffix_tree.num_seqs()}")
        
        # Test some common patterns to show tree behavior
        test_patterns = self._generate_test_patterns(max_depth)
        
        print(f"\nTesting {len(test_patterns)} patterns to explore tree behavior:")
        print("-" * 60)
        
        for i, pattern in enumerate(test_patterns):
            if i >= 20:  # Limit to prevent too much output
                print(f"... and {len(test_patterns) - i} more patterns")
                break
                
            try:
                result = self.suffix_tree.speculate(
                    pattern, 
                    max_spec_tokens=5,
                    max_spec_factor=2.0,
                    min_token_prob=0.1,
                    use_tree_spec=True
                )
                
                pattern_str = " ".join([self._token_to_str(t) for t in pattern])
                if len(pattern_str) > 40:
                    pattern_str = pattern_str[:37] + "..."
                    
                if result.token_ids:
                    spec_tokens = " ".join([self._token_to_str(t) for t in result.token_ids[:3]])
                    if len(result.token_ids) > 3:
                        spec_tokens += "..."
                    print(f"Pattern: {pattern_str:>40} -> {spec_tokens:>20} (score: {result.score:.2f}, match: {result.match_len})")
                else:
                    print(f"Pattern: {pattern_str:>40} -> {'(no speculation)':>20}")
            except Exception as e:
                pattern_str = " ".join([self._token_to_str(t) for t in pattern])
                print(f"Pattern: {pattern_str:>40} -> Error: {str(e)[:30]}")
        
        print("-" * 60)
    
    def visualize_tree_graph(self, max_nodes: int = 50, output_file: str = "suffix_tree.png",
                           highlight_path: List[int] = None):
        """Create a graphical visualization based on speculation results (adapted for C++ binding)"""
        
        # Since we can't access internal tree structure, create a graph based on speculation results
        G = nx.DiGraph()
        node_labels = {}
        node_colors = {}
        edge_labels = {}
        
        # Generate test patterns and their speculation results
        test_patterns = self._generate_test_patterns(8)[:max_nodes//5]  # Limit patterns
        
        # Add root node
        G.add_node("root")
        node_labels["root"] = "ROOT"
        node_colors["root"] = 'lightblue'
        
        node_count = 1
        
        for i, pattern in enumerate(test_patterns):
            if node_count >= max_nodes:
                break
                
            try:
                result = self.suffix_tree.speculate(
                    pattern,
                    max_spec_tokens=5,
                    max_spec_factor=2.0,
                    min_token_prob=0.1,
                    use_tree_spec=True
                )
                
                if result.token_ids and result.match_len > 0:
                    # Create pattern node
                    pattern_id = f"pattern_{i}"
                    pattern_str = " ".join([self._token_to_str(t) for t in pattern[-2:]])  # Show last 2 tokens
                    
                    G.add_node(pattern_id)
                    node_labels[pattern_id] = f"P: {pattern_str}\nM: {result.match_len}"
                    node_colors[pattern_id] = 'lightgreen' if result.score > 1.0 else 'lightyellow'
                    G.add_edge("root", pattern_id)
                    node_count += 1
                    
                    # Add speculation nodes
                    for j, (token_id, parent_idx, prob) in enumerate(zip(result.token_ids[:3], result.parents[:3], result.probs[:3])):
                        if node_count >= max_nodes:
                            break
                            
                        spec_id = f"spec_{i}_{j}"
                        token_str = self._token_to_str(token_id)
                        
                        G.add_node(spec_id)
                        node_labels[spec_id] = f"{token_str}\n{prob:.2f}"
                        node_colors[spec_id] = 'lightcoral'
                        
                        # Connect to parent
                        if parent_idx == -1:
                            G.add_edge(pattern_id, spec_id)
                        else:
                            parent_spec_id = f"spec_{i}_{parent_idx}"
                            if parent_spec_id in G:
                                G.add_edge(parent_spec_id, spec_id)
                        
                        node_count += 1
                        
            except Exception as e:
                # Skip problematic patterns
                continue
        
        if len(G.nodes()) <= 1:
            # If no speculation results, create a simple informational graph
            G.add_node("info")
            node_labels["info"] = f"C++ Tree\n{self.suffix_tree.num_seqs()} seqs"
            node_colors["info"] = 'lightgray'
            G.add_edge("root", "info")
        
        # Create visualization
        plt.figure(figsize=(14, 10))
        pos = nx.spring_layout(G, k=3, iterations=50)
        
        # Draw nodes
        nx.draw_networkx_nodes(G, pos, 
                             node_color=[node_colors.get(n, 'lightgray') for n in G.nodes()],
                             node_size=2000,
                             alpha=0.9)
        
        # Draw edges
        nx.draw_networkx_edges(G, pos, 
                             edge_color='gray',
                             arrows=True,
                             arrowsize=20,
                             arrowstyle='->')
        
        # Draw labels
        nx.draw_networkx_labels(G, pos, node_labels, font_size=9)
        
        plt.title("Suffix Tree Behavior (based on speculation results)\nC++ binding - internal structure not accessible")
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Tree behavior visualization saved to {output_file}")
    
    def analyze_pattern_matches(self, pattern: List[int], max_spec_tokens: int = 10):
        """Analyze and visualize pattern matching for debugging (adapted for C++ binding)"""
        print("\n" + "="*80)
        print("PATTERN MATCHING ANALYSIS")
        print("="*80)
        
        pattern_str = [self._token_to_str(t) for t in pattern]
        print(f"Pattern: {pattern_str}")
        print(f"Pattern length: {len(pattern)}")
        
        # Since we can't access internal _match_pattern, use speculate method
        try:
            result = self.suffix_tree.speculate(
                pattern,
                max_spec_tokens=max_spec_tokens,
                max_spec_factor=2.0,
                min_token_prob=0.1,
                use_tree_spec=True
            )
            
            print(f"\nSpeculation result:")
            print(f"  Match length: {result.match_len}")
            print(f"  Score: {result.score:.3f}")
            print(f"  Number of speculation tokens: {len(result.token_ids)}")
            
            if result.token_ids:
                print(f"\nSpeculated tokens:")
                for i, (token_id, parent_idx, prob) in enumerate(zip(result.token_ids, result.parents, result.probs)):
                    token_str = self._token_to_str(token_id)
                    parent_info = f"parent: {parent_idx}" if parent_idx >= 0 else "root"
                    print(f"  {i}: {token_str} (prob: {prob:.3f}, {parent_info})")
            else:
                print("  No speculation tokens generated.")
                
            # Test variations of the pattern
            print(f"\nTesting pattern variations:")
            variations = []
            
            # Try shorter patterns
            for length in range(max(1, len(pattern)-3), len(pattern)):
                if length > 0:
                    var_pattern = pattern[-length:]
                    variations.append((f"Last {length} tokens", var_pattern))
            
            # Try longer patterns if we have data
            if len(pattern) < 10:
                # Add some common tokens to extend pattern
                for extra_token in [100, 101, 102]:
                    extended = pattern + [extra_token]
                    variations.append((f"+ T{extra_token}", extended))
                    
            for desc, var_pattern in variations:
                try:
                    var_result = self.suffix_tree.speculate(
                        var_pattern,
                        max_spec_tokens=5,
                        max_spec_factor=2.0,
                        min_token_prob=0.1,
                        use_tree_spec=True
                    )
                    var_pattern_str = [self._token_to_str(t) for t in var_pattern[-3:]]
                    print(f"  {desc:>15} (...{var_pattern_str}): match={var_result.match_len}, score={var_result.score:.2f}, specs={len(var_result.token_ids)}")
                except:
                    print(f"  {desc:>15}: Error in speculation")
                    
        except Exception as e:
            print(f"Error analyzing pattern: {str(e)}")
            print("Note: Pattern analysis limited due to C++ binding constraints.")
    
    def visualize_speculation_tree(self, pattern: List[int], max_spec_tokens: int = 10,
                                 output_file: str = "speculation_tree.png"):
        """Visualize the speculation tree built from a pattern"""
        
        # Get speculation result
        result = self.suffix_tree.speculate(
            pattern, max_spec_tokens,
            max_spec_factor=1.0,
            min_token_prob=0.1,
            use_tree_spec=True
        )
        
        if isinstance(result, list):
            result = result[0]  # Take first if multiple
        
        if not result.token_ids:
            print("No speculation candidates found!")
            return
        
        # Build tree structure from parents
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Calculate positions
        levels = {}
        for i, parent in enumerate(result.parents):
            if parent == -1:
                level = 0
            else:
                level = levels[parent] + 1
            levels[i] = level
        
        max_level = max(levels.values()) if levels else 0
        
        # Position nodes
        level_counts = {}
        level_positions = {}
        
        for i, level in levels.items():
            if level not in level_counts:
                level_counts[level] = 0
            level_counts[level] += 1
            level_positions[i] = (level_counts[level], level)
        
        # Draw nodes and edges
        for i, (token_id, parent, prob) in enumerate(zip(result.token_ids, 
                                                         result.parents, 
                                                         result.probs)):
            x_pos, y_pos = level_positions[i]
            
            # Adjust x position for centering
            total_at_level = sum(1 for l in levels.values() if l == y_pos)
            x_pos = x_pos - total_at_level / 2
            
            # Draw node
            circle = plt.Circle((x_pos * 2, -y_pos * 2), 0.4, 
                               color='lightblue', ec='black', linewidth=2)
            ax.add_patch(circle)
            
            # Add text
            token_str = self._token_to_str(token_id)
            ax.text(x_pos * 2, -y_pos * 2, f"{token_str}\n{prob:.2f}", 
                   ha='center', va='center', fontsize=8)
            
            # Draw edge to parent
            if parent >= 0:
                parent_x, parent_y = level_positions[parent]
                parent_x = parent_x - sum(1 for l in levels.values() if l == parent_y) / 2
                
                ax.arrow(parent_x * 2, -parent_y * 2 - 0.4,
                        (x_pos - parent_x) * 2, -(y_pos - parent_y) * 2 + 0.8,
                        head_width=0.2, head_length=0.1, fc='gray', ec='gray')
        
        # Add pattern at top
        pattern_str = ' '.join([self._token_to_str(t) for t in pattern[-10:]])
        ax.text(0, 1, f"Pattern: ...{pattern_str}", ha='center', fontsize=10, 
               bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.5))
        
        ax.set_xlim(-5, 5)
        ax.set_ylim(-max_level * 2 - 1, 2)
        ax.axis('off')
        ax.set_title(f"Speculation Tree (score={result.score:.2f})")
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Speculation tree saved to {output_file}")
    
    def _token_to_str(self, token_id: int) -> str:
        """Convert token ID to string representation"""
        if self.tokenizer:
            try:
                # 使用tokenizer进行真正的detokenization
                decoded_text = self.tokenizer.decode([token_id])
                # 清理文本：移除换行和多余的空格，限制长度
                cleaned_text = decoded_text.replace('\n', '\\n').replace('\t', '\\t').strip()
                if len(cleaned_text) > 20:  # 避免token太长
                    cleaned_text = cleaned_text[:17] + "..."
                # 如果解码后是空白或特殊字符，用引号包围以便识别
                if not cleaned_text or cleaned_text.isspace():
                    return f'"{cleaned_text}"'
                elif any(c in cleaned_text for c in [' ', '\n', '\t']):
                    return f'"{cleaned_text}"'
                else:
                    return cleaned_text
            except Exception as e:
                # tokenizer失败时显示错误信息
                return f"T{token_id}(decode_err)"
        return f"T{token_id}"
    
    def _get_node_tokens(self, node) -> List[int]:
        """Get the token sequence represented by a compressed node (C++ binding limitation)"""
        # This method cannot work with C++ binding as internal _seqs is not accessible
        # Return empty list as fallback
        return []
    
    def _get_compressed_tokens(self, node, tree_info) -> List[int]:
        """从tree_info中获取压缩节点的具体token内容"""
        try:
            seq_id = node.get('seq_id', -1)
            start = node.get('start', 0)
            length = node.get('length', 0)
            
            if seq_id < 0 or length <= 1:
                return []
            
            # 从tree_info的sequences中获取序列数据
            sequences = tree_info.get('sequences', {})
            if seq_id not in sequences:
                return []
            
            sequence = sequences[seq_id]
            if start < 0 or start >= len(sequence):
                return []
            
            # 提取压缩的token段
            end_idx = min(start + length, len(sequence))
            compressed_tokens = sequence[start:end_idx]
            
            return compressed_tokens
        except Exception as e:
            # 如果获取失败，返回空列表
            return []
    
    def _generate_test_patterns(self, max_depth: int = 5) -> List[List[int]]:
        """Generate test patterns to explore tree behavior"""
        patterns = []
        
        # Single token patterns
        for token in range(100, 150):  # Common token range
            patterns.append([token])
        
        # Short patterns
        for i in range(100, 120):
            for j in range(100, 110):
                patterns.append([i, j])
                if len(patterns) < 50:  # Add some 3-token patterns
                    for k in range(100, 105):
                        patterns.append([i, j, k])
        
        # If we have a tokenizer, try some common word patterns
        if self.tokenizer:
            try:
                # Common words/phrases
                test_phrases = ["the", "and", "is", "in", "to", "of", "a"]
                for phrase in test_phrases:
                    tokens = self.tokenizer.encode(phrase)
                    if len(tokens) <= max_depth:
                        patterns.append(tokens)
            except:
                pass  # Ignore tokenizer errors
        
        return patterns[:50]  # Limit to 50 patterns
    
    def debug_statistics(self):
        """Print debugging statistics about the tree (adapted for C++ binding)"""
        print("\n" + "="*80)
        print("SUFFIX TREE STATISTICS")
        print("="*80)
        
        print("C++ Suffix Tree Statistics:")
        print(f"Number of sequences: {self.suffix_tree.num_seqs()}")
        
        # Test patterns to get behavior statistics
        test_patterns = self._generate_test_patterns(10)
        successful_specs = 0
        total_spec_tokens = 0
        total_score = 0.0
        
        print(f"\nTesting {len(test_patterns)} patterns for statistics:")
        
        for pattern in test_patterns:
            try:
                result = self.suffix_tree.speculate(
                    pattern,
                    max_spec_tokens=10,
                    max_spec_factor=2.0,
                    min_token_prob=0.1,
                    use_tree_spec=True
                )
                if result.token_ids:
                    successful_specs += 1
                    total_spec_tokens += len(result.token_ids)
                    total_score += result.score
            except:
                pass
        
        print(f"Successful speculations: {successful_specs}/{len(test_patterns)} ({successful_specs/len(test_patterns)*100:.1f}%)")
        if successful_specs > 0:
            print(f"Average speculation length: {total_spec_tokens/successful_specs:.1f} tokens")
            print(f"Average speculation score: {total_score/successful_specs:.2f}")
        
        print(f"\nNote: Internal tree structure not accessible through C++ binding.")


def main():
    """Command-line interface for suffix tree visualization"""
    parser = argparse.ArgumentParser(description="Visualize suffix tree for debugging")
    parser.add_argument('--cache-file', type=str, default='suffix_cache.pkl',
                      help='Path to suffix cache pickle file')
    parser.add_argument('--max-depth', type=int, default=5,
                      help='Maximum depth for text visualization')
    parser.add_argument('--pattern', type=str,
                      help='Pattern to analyze (comma-separated token IDs)')
    parser.add_argument('--graph', action='store_true',
                      help='Generate graphical visualization')
    parser.add_argument('--stats', action='store_true',
                      help='Show tree statistics')
    
    args = parser.parse_args()
    
    # Create visualizer
    try:
        viz = SuffixTreeVisualizer(cache_file=args.cache_file)
    except Exception as e:
        print(f"Error loading cache file: {e}")
        return
    
    # Run requested visualizations
    if args.stats:
        viz.debug_statistics()
    
    if args.pattern:
        # Parse pattern
        try:
            pattern = [int(x.strip()) for x in args.pattern.split(',')]
            viz.analyze_pattern_matches(pattern)
            viz.visualize_speculation_tree(pattern)
        except ValueError:
            print("Error: Pattern must be comma-separated integers")
    
    if args.graph:
        viz.visualize_tree_graph()
    
    # Always show tree structure
    viz.print_tree_structure(max_depth=args.max_depth)


if __name__ == "__main__":
    main()