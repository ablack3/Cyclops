# Standalone CMake target for the Cyclops C++ core.
#
# This file is additive: the R build drives compilation from src/Makevars(.win),
# which CMake never sees, so adding it changes nothing for `R CMD INSTALL`.
#
# It exists so that non-R front-ends (Python bindings, the JNI layer, the CLI in
# standalone/) can consume the core as an ordinary CMake target instead of
# re-listing its translation units. The list below mirrors src/Makevars.in.
#
# Consume it with:
#     add_subdirectory(<cyclops>/src cyclops_core)
#     target_link_libraries(my_target PRIVATE cyclops::core)

cmake_minimum_required(VERSION 3.19)

if(NOT DEFINED PROJECT_NAME)
    project(CyclopsCore LANGUAGES CXX)
endif()

# --- version ----------------------------------------------------------------

if(NOT DEFINED CYCLOPS_VERSION)
    file(READ "${CMAKE_CURRENT_LIST_DIR}/../DESCRIPTION" _cyclops_description)
    string(REGEX MATCH "Version:[ \t]*([0-9][0-9.]*)" _ "${_cyclops_description}")
    set(CYCLOPS_VERSION "${CMAKE_MATCH_1}")
    if(NOT CYCLOPS_VERSION)
        message(FATAL_ERROR "Could not read Version from DESCRIPTION")
    endif()
endif()

# --- dependencies -----------------------------------------------------------

find_package(Threads REQUIRED)

# Eigen is header-only and used only for the Fisher-information and asymptotic
# variance matrices in CyclicCoordinateDescent. The R build gets it from the
# RcppEigen package; other front-ends supply their own.
if(NOT TARGET Eigen3::Eigen)
    find_package(Eigen3 3.3 QUIET NO_MODULE)
endif()
if(NOT TARGET Eigen3::Eigen)
    message(STATUS "Eigen3 not found locally; fetching a pinned copy")
    include(FetchContent)
    set(EIGEN_BUILD_DOC OFF CACHE BOOL "" FORCE)
    set(EIGEN_BUILD_TESTING OFF CACHE BOOL "" FORCE)
    set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
    FetchContent_Declare(Eigen3
        URL https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz
        URL_HASH SHA256=8586084f71f9bde545ee7fa6d00288b264a2b7ac3607b974e54d13e7162c1c72
    )
    FetchContent_MakeAvailable(Eigen3)
endif()

# --- sources ----------------------------------------------------------------

set(CYCLOPS_CORE_SOURCES
    cyclops/CcdInterface.cpp
    cyclops/CompressedDataMatrix.cpp
    cyclops/CyclicCoordinateDescent.cpp
    cyclops/ModelData.cpp
    cyclops/Timer.cpp

    cyclops/drivers/AbstractCrossValidationDriver.cpp
    cyclops/drivers/AbstractDriver.cpp
    cyclops/drivers/AbstractSelector.cpp
    cyclops/drivers/AutoSearchCrossValidationDriver.cpp
    cyclops/drivers/BootstrapDriver.cpp
    cyclops/drivers/BootstrapSelector.cpp
    cyclops/drivers/CrossValidationSelector.cpp
    cyclops/drivers/WeightBasedSelector.cpp
    cyclops/drivers/GridSearchCrossValidationDriver.cpp
    cyclops/drivers/HierarchyAutoSearchCrossValidationDriver.cpp
    cyclops/drivers/HierarchyGridSearchCrossValidationDriver.cpp
    cyclops/drivers/ProportionSelector.cpp

    cyclops/priors/CovariatePrior.cpp
    cyclops/engine/AbstractModelSpecifics.cpp

    utils/HParSearch.cpp
)

# cyclops/io/InputReader.cpp is deliberately omitted: it implements the
# text-file readers used only by the CLI, and every other translation unit
# references it solely from commented-out includes.

if(WIN32)
    # Thread.h only routes through TinyThread on Windows (see USE_TTHREAD).
    list(APPEND CYCLOPS_CORE_SOURCES tinythread/tinythread.cpp)
endif()

# The stable, language-neutral facade.
set(CYCLOPS_API_SOURCES
    cyclops/api/CyclopsApi.cpp
)

# --- target -----------------------------------------------------------------

add_library(cyclops_core STATIC ${CYCLOPS_CORE_SOURCES} ${CYCLOPS_API_SOURCES})
add_library(cyclops::core ALIAS cyclops_core)

target_include_directories(cyclops_core
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}>
        # Several core headers include their siblings unqualified
        # (e.g. #include "ModelData.h"), so cyclops/ is on the path too.
        $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/cyclops>
)

target_compile_definitions(cyclops_core
    PUBLIC
        DOUBLE_PRECISION
        CYCLOPS_VERSION="${CYCLOPS_VERSION}"
)

target_compile_features(cyclops_core PUBLIC cxx_std_17)

target_link_libraries(cyclops_core
    PUBLIC
        Eigen3::Eigen
        Threads::Threads
)

set_target_properties(cyclops_core PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    CXX_EXTENSIONS OFF
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON
)

if(MSVC)
    # The core relies on the conforming preprocessor and on <cmath> constants.
    target_compile_definitions(cyclops_core PUBLIC _USE_MATH_DEFINES NOMINMAX)
    # /Zc:__cplusplus is required, not cosmetic: without it MSVC reports
    # __cplusplus as 199711L regardless of /std:c++17, so Types.h compiles its
    # C++98 make_unique emulation *as well as* pulling in std::make_unique
    # through ADL -- every call site is then ambiguous. PUBLIC because that
    # header is part of the interface.
    target_compile_options(cyclops_core PUBLIC /Zc:__cplusplus)
    target_compile_options(cyclops_core PRIVATE /bigobj /EHsc)
else()
    # -g1 mirrors src/Makevars.in: the templated kernels in engine/ generate
    # enormous debug records otherwise.
    target_compile_options(cyclops_core PRIVATE
        $<$<CONFIG:Release>:-g1>
        -Wno-unknown-pragmas
    )
endif()
