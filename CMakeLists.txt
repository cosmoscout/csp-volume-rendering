# ------------------------------------------------------------------------------------------------ #
#                                This file is part of CosmoScout VR                                #
#       and may be used under the terms of the MIT license. See the LICENSE file for details.      #
#                         Copyright: (c) 2020 German Aerospace Center (DLR)                        #
# ------------------------------------------------------------------------------------------------ #

option(CSP_VOLUME_RENDERING "Enable compilation of this plugin" ON)

if (NOT CSP_VOLUME_RENDERING)
  return()
endif()

# dependencies -------------------------------------------------------------------------------------

if (DEFINED ENV{VIRACOCHA_ROOT_DIR})
  SET(VIRACOCHA_ROOT_DIR "$ENV{VIRACOCHA_ROOT_DIR}")
else()
  SET(VIRACOCHA_ROOT_DIR ${COSMOSCOUT_EXTERNALS_DIR})
endif()

if (DEFINED ENV{VTK_CMAKE_CONFIG_DIR})
  SET(VTK_DIR "$ENV{VTK_CMAKE_CONFIG_DIR}")
else()
  SET(VTK_DIR ${COSMOSCOUT_EXTERNALS_DIR}/lib/cmake/vtk-8.2)
endif()

if (DEFINED ENV{OSPRAY_CMAKE_CONFIG_DIR})
  SET(ospray_DIR "$ENV{OSPRAY_CMAKE_CONFIG_DIR}")
else()
  SET(ospray_DIR ${COSMOSCOUT_EXTERNALS_DIR}/lib/cmake/ospray-2.1.1)
endif()

if (DEFINED ENV{EMBREE_CMAKE_CONFIG_DIR})
  SET(embree_DIR "$ENV{EMBREE_CMAKE_CONFIG_DIR}")
else()
  SET(embree_DIR ${COSMOSCOUT_EXTERNALS_DIR})
endif()

if (DEFINED ENV{OSPCOMMON_CMAKE_CONFIG_DIR})
  SET(ospcommon_DIR "$ENV{OSPCOMMON_CMAKE_CONFIG_DIR}")
else()
  SET(ospcommon_DIR ${COSMOSCOUT_EXTERNALS_DIR}/lib/cmake/ospcommon-1.3.1)
endif()

if (DEFINED ENV{OPENVKL_CMAKE_CONFIG_DIR})
  SET(openvkl_DIR "$ENV{OPENVKL_CMAKE_CONFIG_DIR}")
else()
  SET(openvkl_DIR ${COSMOSCOUT_EXTERNALS_DIR}/lib/cmake/openvkl-0.9.0)
endif()

if (DEFINED ENV{OIDN_CMAKE_CONFIG_DIR})
  SET(OpenImageDenoise_DIR "$ENV{OIDN_CMAKE_CONFIG_DIR}")
else()
  SET(OpenImageDenoise_DIR ${COSMOSCOUT_EXTERNALS_DIR}/lib/cmake/OpenImageDenoise)
endif()

if (DEFINED ENV{OSPCOMMON_TBB_ROOT})
  SET(OSPCOMMON_TBB_ROOT "$ENV{OSPCOMMON_TBB_ROOT}")
else()
  SET(OSPCOMMON_TBB_ROOT ${COSMOSCOUT_EXTERNALS_DIR})
endif()

if (DEFINED ENV{ISPC_EXECUTABLE})
  SET(ISPC_EXECUTABLE "$ENV{ISPC_EXECUTABLE}")
else()
  SET(ISPC_EXECUTABLE ${COSMOSCOUT_EXTERNALS_DIR}/bin/ispc.exe)
endif()

find_package(ospray REQUIRED)
find_package(OpenImageDenoise REQUIRED)
find_package(VTK REQUIRED)
find_package(VIRACOCHA REQUIRED)

# build plugin -------------------------------------------------------------------------------------

file(GLOB_RECURSE SOURCE_FILES src/*.cpp)
# Resoucre files and header files are only added in order to make them available in your IDE.
file(GLOB_RECURSE HEADER_FILES src/*.hpp)
file(GLOB_RECURSE RESOUCRE_FILES gui/*)

add_library(csp-volume-rendering SHARED
  ${SOURCE_FILES}
  ${HEADER_FILES}
  ${RESOUCRE_FILES}
)

target_link_libraries(csp-volume-rendering
  PUBLIC
    cs-core
    ospray::ospray
    OpenImageDenoise
    viracocha::viracochabackend
    ${VTK_LIBRARIES}
)

target_include_directories(csp-volume-rendering
  PUBLIC
	  ${VTK_INCLUDE_DIRS}
)

# Add this Plugin to a "plugins" folder in your IDE.
set_property(TARGET csp-volume-rendering PROPERTY FOLDER "plugins")

# We mark all resource files as "header" in order to make sure that no one tries to compile them.
set_source_files_properties(${RESOUCRE_FILES} PROPERTIES HEADER_FILE_ONLY TRUE)

# Make directory structure available in your IDE.
source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}" FILES
  ${SOURCE_FILES} ${HEADER_FILES} ${RESOUCRE_FILES}
)

install(TARGETS csp-volume-rendering DESTINATION "share/plugins")
install(DIRECTORY "gui" DESTINATION "share/resources")

# build ospray module ------------------------------------------------------------------------------

list(APPEND CMAKE_MODULE_PATH ${ospray_DIR})
include(ispc)
include(GenerateExportHeader)

get_target_property(OSPRAY_ISPC_MODULE_INCLUDE_DIRS ospray::ospray_module_ispc INTERFACE_INCLUDE_DIRECTORIES)

ispc_include_directories(${OSPRAY_ISPC_MODULE_INCLUDE_DIRS})

file(GLOB_RECURSE MODULE_SOURCE_FILES ospray_modules/volume_depth/*.cpp ospray_modules/volume_depth/*.ispc)
file(GLOB_RECURSE MODULE_HEADER_FILES ospray_modules/volume_depth/*.hpp ospray_modules/volume_depth/*.ih)

add_library(ospray_module_volume_depth SHARED)

ispc_target_add_sources(ospray_module_volume_depth
  ${MODULE_SOURCE_FILES}
	${MODULE_HEADER_FILES}
)

target_sources(ospray_module_volume_depth PRIVATE ${MODULE_SOURCE_FILES})

generate_export_header(ospray_module_volume_depth)

target_link_libraries(ospray_module_volume_depth
  PUBLIC
	  ospray::ospray_module_ispc
)

source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR}/ospray_modules/volume_depth FILES
  ${MODULE_SOURCE_FILES} ${MODULE_HEADER_FILES}
)

install(TARGETS ospray_module_volume_depth DESTINATION "lib")