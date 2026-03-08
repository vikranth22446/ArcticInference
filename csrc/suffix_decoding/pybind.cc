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
        .def("append", &SuffixTree::append,
             py::call_guard<py::gil_scoped_release>())
        .def("extend", &SuffixTree::extend,
             py::call_guard<py::gil_scoped_release>())
        .def("remove", &SuffixTree::remove,
             py::call_guard<py::gil_scoped_release>())
        .def("clear", &SuffixTree::clear,
             py::call_guard<py::gil_scoped_release>())
        .def("speculate", &SuffixTree::speculate,
             py::arg("pattern"),
             py::arg("max_spec_tokens"),
             py::arg("max_spec_factor") = 1.0f,
             py::arg("max_spec_offset") = 0.0f,
             py::arg("min_token_prob") = 0.1f,
             py::arg("use_tree_spec") = false,
             py::call_guard<py::gil_scoped_release>())
        .def("append_safe", &SuffixTree::append_safe, 
             py::call_guard<py::gil_scoped_release>(),
             "Thread-safe append with per-object locking and GIL release")
        .def("extend_safe", &SuffixTree::extend_safe, 
             py::call_guard<py::gil_scoped_release>(),
             "Thread-safe extend with per-object locking and GIL release")
        .def("remove_safe", &SuffixTree::remove_safe,
             py::call_guard<py::gil_scoped_release>(),
             "Thread-safe remove with per-object locking and GIL release")
        .def("num_seqs_safe", &SuffixTree::num_seqs_safe,
             py::call_guard<py::gil_scoped_release>(),
             "Thread-safe num_seqs with per-object locking and GIL release")
        .def("speculate_safe", &SuffixTree::speculate_safe,
             py::arg("pattern"),
             py::arg("max_spec_tokens"),
             py::arg("max_spec_factor") = 1.0f,
             py::arg("max_spec_offset") = 0.0f,
             py::arg("min_token_prob") = 0.1f,
             py::arg("use_tree_spec") = false,
             py::call_guard<py::gil_scoped_release>(),
             "Thread-safe speculate with per-object locking and GIL release")
        .def("check_integrity", &SuffixTree::check_integrity)
        .def("estimate_memory", &SuffixTree::estimate_memory);
}