// Copyright 2025 Snowflake Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "suffix_tree.h"

#ifdef DEBUG_VISUALIZATION
#include "suffix_tree_extractor.h"
#endif

namespace py = pybind11;


PYBIND11_MODULE(_C, m) {
    py::class_<Candidate>(m, "Candidate")
        .def_readwrite("token_ids", &Candidate::token_ids)
        .def_readwrite("parents", &Candidate::parents)
        .def_readwrite("probs", &Candidate::probs)
        .def_readwrite("score", &Candidate::score)
        .def_readwrite("match_len", &Candidate::match_len);

    py::class_<SuffixTree>(m, "SuffixTree")
        .def(py::init<int>())
        .def("num_seqs", &SuffixTree::num_seqs)
        .def("append", &SuffixTree::append)
        .def("extend", &SuffixTree::extend)
        .def("speculate", &SuffixTree::speculate,py::call_guard<py::gil_scoped_release>())

#ifdef DEBUG_VISUALIZATION
        // Debug visualization methods
        .def("_debug_extract_sequences", [](SuffixTree& self) {
            return SuffixTreeExtractor::extract_sequences(self);
        }, "Extract sequences from suffix tree (debug only)")
        
        .def("_debug_extract_tree_info", [](SuffixTree& self) {
            TreeDebugInfo info = SuffixTreeExtractor::extract_tree_info(self);
            
            // Convert to Python dict for easier handling
            py::dict result;
            result["sequences"] = info.sequences;
            result["total_nodes"] = info.total_nodes;
            result["max_depth"] = info.max_depth;
            result["root_ptr"] = info.root_ptr;
            
            // Convert nodes to Python list of dicts
            py::list nodes;
            for (const auto& node : info.nodes) {
                py::dict node_dict;
                node_dict["count"] = node.count;
                node_dict["node_ptr"] = node.node_ptr;
                node_dict["parent_ptr"] = node.parent_ptr;
                node_dict["seq_id"] = node.seq_id;
                node_dict["start"] = node.start;
                node_dict["length"] = node.length;
                
                // Convert children to list of pairs
                py::list children;
                for (const auto& [token_id, child_ptr] : node.children) {
                    children.append(py::make_tuple(token_id, child_ptr));
                }
                node_dict["children"] = children;
                
                nodes.append(node_dict);
            }
            result["nodes"] = nodes;
            
            return result;
        }, "Extract complete tree debug info (debug only)")
        
        .def("_debug_extract_statistics", [](SuffixTree& self) {
            auto stats = SuffixTreeExtractor::extract_statistics(self);
            py::dict result;
            result["total_nodes"] = stats.total_nodes;
            result["compressed_nodes"] = stats.compressed_nodes;
            result["max_depth_actual"] = stats.max_depth_actual;
            result["avg_compression_ratio"] = stats.avg_compression_ratio;
            result["depth_distribution"] = stats.depth_distribution;
            return result;
        }, "Extract tree statistics (debug only)")
        
        .def("_debug_validate_integrity", [](SuffixTree& self) {
            return SuffixTreeExtractor::validate_tree_integrity(self);
        }, "Validate tree integrity (debug only)")
#endif
        ;

#ifdef DEBUG_VISUALIZATION
    // Add module-level function to check if debug features are available
    m.def("debug_visualization_available", []() { return true; }, 
          "Check if debug visualization features are compiled in");
#else
    m.def("debug_visualization_available", []() { return false; }, 
          "Check if debug visualization features are compiled in");
#endif
}
