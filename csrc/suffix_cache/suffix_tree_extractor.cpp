// Copyright 2025 Snowflake Inc.
// SPDX-License-Identifier: Apache-2.0

#ifdef DEBUG_VISUALIZATION

#include "suffix_tree_extractor.h"
#include <queue>
#include <set>

// 私有萃取方法实现
const std::unordered_map<int, std::vector<int>>& SuffixTreeExtractor::get_seqs(const SuffixTree& tree) {
    return tree._seqs;
}

Node* SuffixTreeExtractor::get_root(const SuffixTree& tree) {
    return tree._root.get();
}

int SuffixTreeExtractor::get_max_depth(const SuffixTree& tree) {
    return tree._max_depth;
}

NodeDebugInfo SuffixTreeExtractor::create_node_info(Node* node) {
    NodeDebugInfo info;
    if (!node) return info;
    
    info.count = node->count;
    info.node_ptr = reinterpret_cast<uintptr_t>(node);
    info.parent_ptr = reinterpret_cast<uintptr_t>(node->parent);
    info.seq_id = node->seq_id;
    info.start = node->start;
    info.length = node->length;
    
    // 收集子节点信息
    for (const auto& [token_id, child_ptr] : node->children) {
        info.children.emplace_back(token_id, reinterpret_cast<uintptr_t>(child_ptr.get()));
    }
    
    return info;
}

void SuffixTreeExtractor::traverse_node(Node* node, std::vector<NodeDebugInfo>& nodes, int depth) {
    if (!node) return;
    
    // 添加当前节点
    nodes.push_back(create_node_info(node));
    
    // 递归遍历子节点
    for (const auto& [token_id, child_ptr] : node->children) {
        traverse_node(child_ptr.get(), nodes, depth + 1);
    }
}

// 公共接口实现
TreeDebugInfo SuffixTreeExtractor::extract_tree_info(const SuffixTree& tree) {
    TreeDebugInfo info;
    
    // 萃取序列数据
    const auto& seqs = get_seqs(tree);
    for (const auto& [seq_id, seq] : seqs) {
        info.sequences[seq_id] = seq;
    }
    
    // 萃取树结构
    Node* root = get_root(tree);
    if (root) {
        info.root_ptr = reinterpret_cast<uintptr_t>(root);
        traverse_node(root, info.nodes);
        info.total_nodes = static_cast<int>(info.nodes.size());
    }
    
    info.max_depth = get_max_depth(tree);
    
    return info;
}

std::map<int, std::vector<int>> SuffixTreeExtractor::extract_sequences(const SuffixTree& tree) {
    std::map<int, std::vector<int>> result;
    const auto& seqs = get_seqs(tree);
    
    for (const auto& [seq_id, seq] : seqs) {
        result[seq_id] = seq;
    }
    
    return result;
}

NodeDebugInfo SuffixTreeExtractor::extract_node_info(const SuffixTree& tree, Node* node) {
    return create_node_info(node);
}

Node* SuffixTreeExtractor::extract_root_node(const SuffixTree& tree) {
    return get_root(tree);
}

void SuffixTreeExtractor::traverse_tree(const SuffixTree& tree, 
                                       std::function<void(const NodeDebugInfo&, int)> visitor,
                                       int max_depth) {
    Node* root = get_root(tree);
    if (!root) return;
    
    // BFS遍历
    std::queue<std::pair<Node*, int>> queue;
    queue.push({root, 0});
    
    while (!queue.empty()) {
        auto [node, depth] = queue.front();
        queue.pop();
        
        if (max_depth >= 0 && depth > max_depth) continue;
        
        // 访问当前节点
        NodeDebugInfo info = create_node_info(node);
        visitor(info, depth);
        
        // 加入子节点
        for (const auto& [token_id, child_ptr] : node->children) {
            queue.push({child_ptr.get(), depth + 1});
        }
    }
}

std::pair<Node*, int> SuffixTreeExtractor::extract_match_pattern(SuffixTree& tree,
                                                                const std::vector<int>& pattern,
                                                                int start_idx) {
    // 调用私有方法
    return tree._match_pattern(pattern, start_idx);
}

SuffixTreeExtractor::TreeStats SuffixTreeExtractor::extract_statistics(const SuffixTree& tree) {
    TreeStats stats;
    Node* root = get_root(tree);
    
    if (!root) return stats;
    
    // 遍历收集统计信息
    std::queue<std::pair<Node*, int>> queue;
    queue.push({root, 0});
    
    int total_compression = 0;
    int max_depth = 0;
    
    while (!queue.empty()) {
        auto [node, depth] = queue.front();
        queue.pop();
        
        stats.total_nodes++;
        max_depth = std::max(max_depth, depth);
        stats.depth_distribution[depth]++;
        
        if (node->length > 1) {
            stats.compressed_nodes++;
            total_compression += (node->length - 1);
        }
        
        // 加入子节点
        for (const auto& [token_id, child_ptr] : node->children) {
            queue.push({child_ptr.get(), depth + 1});
        }
    }
    
    stats.max_depth_actual = max_depth;
    
    if (stats.compressed_nodes > 0) {
        stats.avg_compression_ratio = static_cast<double>(total_compression) / stats.compressed_nodes;
    } else {
        stats.avg_compression_ratio = 0.0;
    }
    
    return stats;
}

bool SuffixTreeExtractor::validate_tree_integrity(const SuffixTree& tree) {
    Node* root = get_root(tree);
    if (!root) return false;
    
    std::set<Node*> visited;
    std::queue<Node*> queue;
    queue.push(root);
    
    while (!queue.empty()) {
        Node* node = queue.front();
        queue.pop();
        
        if (visited.count(node)) {
            // 检测到循环，树结构有问题
            return false;
        }
        visited.insert(node);
        
        // 验证父子关系
        for (const auto& [token_id, child_ptr] : node->children) {
            Node* child = child_ptr.get();
            if (child->parent != node) {
                // 父子关系不一致
                return false;
            }
            queue.push(child);
        }
    }
    
    return true;
}

#endif // DEBUG_VISUALIZATION
