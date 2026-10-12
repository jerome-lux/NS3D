cmake_minimum_required(VERSION 3.16)

project(navierstokes3D LANGUAGES CXX)

# --- general options ----------------------------------------------------------
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release)
endif()

# --- directory layout ---------------------------------------------------------
set(SRC_DIR   ${CMAKE_SOURCE_DIR}/src)
set(DATA_DIR  ${CMAKE_SOURCE_DIR}/data)
set(TESTS_DIR ${CMAKE_SOURCE_DIR}/tests)

# --- dependencies --------------------------------------------------------------
# Eigen3 (header only): the code uses the #include <eigen3/...> form,
# so we must point to the PARENT directory of the eigen3/ folder.
# 'eigen3/Eigen/Core' exists both in Eigen 3.3 and 3.4.
find_path(EIGEN3_INCLUDE_DIR
          NAMES eigen3/Eigen/Core
          DOC "Directory containing the eigen3/ include tree")
if(NOT EIGEN3_INCLUDE_DIR)
    message(FATAL_ERROR
        "Eigen3 not found (an 'eigen3/...' include tree is expected).\n"
        "On Debian/Ubuntu: sudo apt install libeigen3-dev")
endif()
message(STATUS "Eigen3: ${EIGEN3_INCLUDE_DIR}")

# Blitz++ (3D arrays)
find_path(BLITZ_INCLUDE_DIR
          NAMES blitz/array.h
          DOC "Include directory of Blitz++")
find_library(BLITZ_LIBRARY
             NAMES blitz
             DOC "Blitz++ library")
if(NOT BLITZ_INCLUDE_DIR OR NOT BLITZ_LIBRARY)
    message(FATAL_ERROR
        "Blitz++ not found.\n"
        "On Debian/Ubuntu: sudo apt install libblitz-dev")
endif()
message(STATUS "Blitz++: ${BLITZ_INCLUDE_DIR} / ${BLITZ_LIBRARY}")

# VTK (reading/writing .vti images)
# Request only the components we actually use so that the Qt-dependent
# GUI modules (vtkGUISupportQt, ...) are NOT pulled in. On Ubuntu the
# VTK 9 packages are built with Qt enabled and a bare find_package(VTK)
# triggers a search for Qt6.
find_package(VTK REQUIRED COMPONENTS
    CommonCore
    CommonDataModel
    CommonMath
    ImagingCore
    IOCore
    IOLegacy
    IOXML
)
# Imported targets linked below: VTK::CommonCore VTK::CommonDataModel ...
set(VTK_TARGETS
    VTK::CommonCore
    VTK::CommonDataModel
    VTK::CommonMath
    VTK::ImagingCore
    VTK::IOCore
    VTK::IOLegacy
    VTK::IOXML
)
message(STATUS "VTK ${VTK_VERSION} found")

# OpenMP (optional: the code also runs in serial)
find_package(OpenMP QUIET)

# Common compile settings shared by every target.
set(NS3D_INCLUDES ${SRC_DIR} ${EIGEN3_INCLUDE_DIR} ${BLITZ_INCLUDE_DIR})
set(NS3D_LIBS ${BLITZ_LIBRARY} ${VTK_TARGETS})

# --- targets --------------------------------------------------------------------
# Main solver executable.
add_executable(NS3D
    ${SRC_DIR}/navierstokes3D.cxx
    ${SRC_DIR}/NSBase.cxx
    ${SRC_DIR}/SIMPLEC.cxx
    ${SRC_DIR}/SIMPLER.cxx
    ${SRC_DIR}/SIMPLERR.cxx
    ${SRC_DIR}/IDEAL.cxx
    ${SRC_DIR}/vtkio.cxx
    ${SRC_DIR}/NSBase.h
    ${SRC_DIR}/SIMPLEC.h
    ${SRC_DIR}/SIMPLER.h
    ${SRC_DIR}/SIMPLERR.h
    ${SRC_DIR}/IDEAL.h
    ${SRC_DIR}/vtkio.h
)

# Test-image generator (spheres, non-overlapping). Only needs the VTK I/O.
add_executable(gen_spheres
    ${SRC_DIR}/gen_spheres.cxx
    ${SRC_DIR}/vtkio.cxx
    ${SRC_DIR}/vtkio.h
)

foreach(tgt NS3D gen_spheres)
    target_include_directories(${tgt} PRIVATE ${NS3D_INCLUDES})
    target_link_libraries(${tgt} PRIVATE ${NS3D_LIBS})

    if(OpenMP_CXX_FOUND)
        target_link_libraries(${tgt} PRIVATE OpenMP::OpenMP_CXX)
    endif()
endforeach()

if(OpenMP_CXX_FOUND)
    message(STATUS "OpenMP found: building with parallel loops")
else()
    message(STATUS "OpenMP not found: building in serial")
endif()

# --- testing --------------------------------------------------------------------
enable_testing()
add_subdirectory(tests)

# --- installation -------------------------------------------------------------------
install(TARGETS NS3D RUNTIME DESTINATION bin)
