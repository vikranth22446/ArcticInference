// Copyright 2025 Snowflake Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#ifdef DEBUG_VISUALIZATION

#include "suffix_tree.h"
#include <vector>
#include <map>
#include <memory>
#include <functional>

struct NodeDebugInfo {
    int count;
    uintptr_t node_ptr;        // 用于唯一标识节点
    uintptr_t parent_ptr;      // 父节点指针
    std::vector<std::pair<int, uintptr_t>> children;  // token_id -> child_ptr
    int seq_id;
    int start;
    int length;
    
    NodeDebugInfo() : count(0), node_ptr(0), parent_ptr(0), seq_id(-1), start(0), length(0) {}
};

struct TreeDebugInfo {
    std::map<int, std::vector<int>> sequences;      // 序列数据
    std::vector<NodeDebugInfo> nodes;               // 所有节点信息
    uintptr_t root_ptr;                            // 根节点指针
    int max_depth;
    int total_nodes;
    
    TreeDebugInfo() : root_ptr(0), max_depth(0), total_nodes(0) {}
};

class SuffixTreeExtractor {
private:
    // 私有萃取方法 - 直接访问私有成员
    static const std::unordered_map<int, std::vector<int>>& get_seqs(const SuffixTree& tree);
    static Node* get_root(const SuffixTree& tree);
    static int get_max_depth(const SuffixTree& tree);
    
    // 递归遍历节点
    static void traverse_node(Node* node, std::vector<NodeDebugInfo>& nodes, int depth = 0);
    
    // 创建节点调试信息
    static NodeDebugInfo create_node_info(Node* node);

public:
    // 公共接口
    
    /**
     * 萃取完整的树调试信息
     */
    static TreeDebugInfo extract_tree_info(const SuffixTree& tree);
    
    /**
     * 萃取序列数据
     */
    static std::map<int, std::vector<int>> extract_sequences(const SuffixTree& tree);
    
    /**
     * 萃取指定节点的信息
     */
    static NodeDebugInfo extract_node_info(const SuffixTree& tree, Node* node);
    
    /**
     * 获取根节点指针（用于遍历）
     */
    static Node* extract_root_node(const SuffixTree& tree);
    
    /**
     * 遍历整个树结构
     * @param visitor 访问函数: void(NodeDebugInfo, int depth)
     * @param max_depth 最大遍历深度，-1表示无限制
     */
    static void traverse_tree(const SuffixTree& tree, 
                             std::function<void(const NodeDebugInfo&, int)> visitor,
                             int max_depth = -1);
    
    /**
     * 萃取匹配模式的调试信息
     */
    static std::pair<Node*, int> extract_match_pattern(SuffixTree& tree,
                                                      const std::vector<int>& pattern,
                                                      int start_idx = 0);
    
    /**
     * 生成树的统计信息
     */
    struct TreeStats {
        int total_nodes;
        int compressed_nodes;
        int max_depth_actual;
        std::map<int, int> depth_distribution;  // depth -> count
        double avg_compression_ratio;
    };
    
    static TreeStats extract_statistics(const SuffixTree& tree);
    
    /**
     * 验证树的完整性（用于调试）
     */
    static bool validate_tree_integrity(const SuffixTree& tree);
};

#endif // DEBUG_VISUALIZATION
