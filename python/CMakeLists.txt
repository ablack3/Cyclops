# Build for the `cyclops` Python extension module.
#
# Everything Python-specific lives here; the C++ core and the language-neutral
# facade are consumed as an ordinary CMake target from ../src/CMakeLists.txt.

cmake_minimum_required(VERSION 3.19...3.31)

project(cyclops_python LANGUAGES CXX)

if(NOT CYCLOPS_ROOT)
    get_filename_component(CYCLOPS_ROOT "${CMAKE_CURRENT_LIST_DIR}/.." ABSOLUTE)
endif()
message(STATUS "Cyclops root: ${CYCLOPS_ROOT}")

if(NOT EXISTS "${CYCLOPS_ROOT}/src/cyclops/api/CyclopsApi.h")
    message(FATAL_ERROR
        "Cannot find the Cyclops C++ facade under ${CYCLOPS_ROOT}/src. "
        "Build from a full checkout of the repository, not from python/ alone.")
endif()

find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 CONFIG REQUIRED)

# The core + facade static library (cyclops::core).
add_subdirectory("${CYCLOPS_ROOT}/src" cyclops_core)

pybind11_add_module(_cyclops MODULE src/bindings.cpp)

target_link_libraries(_cyclops PRIVATE cyclops::core)

target_compile_definitions(_cyclops PRIVATE
    VERSION_INFO="${SKBUILD_PROJECT_VERSION}"
)

install(TARGETS _cyclops DESTINATION cyclops)
