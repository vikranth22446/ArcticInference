# Copyright 2025 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

cmake_minimum_required(VERSION 3.14)
project(SuffixCache CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build." FORCE)
  message(STATUS "Setting build type to 'Release' as none was specified.")
endif()
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g3 -ggdb")

# Debug visualization option
option(DEBUG_VISUALIZATION "Enable debug visualization for suffix trees" OFF)

if(DEBUG_VISUALIZATION)
    message(STATUS "DEBUG_VISUALIZATION enabled")
    add_definitions(-DDEBUG_VISUALIZATION)
endif()

find_package(pybind11 REQUIRED)

# Source files
set(SOURCES pybind.cc suffix_tree.cc)

# Add extractor sources if debug visualization is enabled
if(DEBUG_VISUALIZATION)
    list(APPEND SOURCES suffix_tree_extractor.cpp)
endif()

pybind11_add_module(_C ${SOURCES})

# EXAMPLE_VERSION_INFO is defined by setup.py and passed into the C++ code as a
# define (VERSION_INFO) here.
target_compile_definitions(_C PRIVATE VERSION_INFO=${EXAMPLE_VERSION_INFO})
