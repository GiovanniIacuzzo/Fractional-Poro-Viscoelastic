# reteco-v1/CMakeLists.txt
cmake_minimum_required(VERSION 3.24)

find_package(deal.II 9.5.0 REQUIRED
  HINTS ${DEAL_II_DIR} $ENV{DEAL_II_DIR} $ENV{CONDA_PREFIX}
)
deal_ii_initialize_cached_variables()

project(reteco_v1 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(Python3 COMPONENTS Interpreter Development.Module REQUIRED)
find_package(pybind11 CONFIG REQUIRED)
find_package(Eigen3 3.4 REQUIRED NO_MODULE)

add_library(reteco_cpp_core STATIC
  cpp_core/src/caputo_l1_memory.cpp
  cpp_core/src/invariant_matrices.cpp
  cpp_core/src/initial_compatibility.cpp
  cpp_core/src/biot_visco_solver.cpp
  cpp_core/src/tracer_transport_solver.cpp
  cpp_core/src/energy_mass_monitor.cpp
  cpp_core/src/mms_manufactured.cpp
)

target_include_directories(reteco_cpp_core PUBLIC
  "${CMAKE_CURRENT_SOURCE_DIR}/cpp_core/include"
)

target_link_libraries(reteco_cpp_core Eigen3::Eigen)

deal_ii_setup_target(reteco_cpp_core RELEASE)

pybind11_add_module(_core MODULE
  cpp_core/bindings/pybind_module.cpp
)

target_include_directories(_core PRIVATE
  "${CMAKE_CURRENT_SOURCE_DIR}/cpp_core/include"
)

target_include_directories(_core SYSTEM PRIVATE
  "${EIGEN3_INCLUDE_DIRS}"
  ${DEAL_II_USER_INCLUDE_DIRS}
)

target_compile_definitions(_core PRIVATE
  ${DEAL_II_USER_DEFINITIONS}
  ${DEAL_II_USER_DEFINITIONS_RELEASE}
)

separate_arguments(
  DEALII_CXX_FLAGS_LIST
  NATIVE_COMMAND
  "${DEAL_II_CXX_FLAGS} ${DEAL_II_CXX_FLAGS_RELEASE}"
)

separate_arguments(
  DEALII_LINK_FLAGS_LIST
  NATIVE_COMMAND
  "${DEAL_II_LINKER_FLAGS} ${DEAL_II_LINKER_FLAGS_RELEASE}"
)

target_compile_options(_core PRIVATE
  ${DEALII_CXX_FLAGS_LIST}
)

target_link_options(_core PRIVATE
  ${DEALII_LINK_FLAGS_LIST}
)

target_link_libraries(_core PRIVATE
  reteco_cpp_core
)

set_target_properties(_core PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY
  "${CMAKE_CURRENT_SOURCE_DIR}/python/reteco"
)
