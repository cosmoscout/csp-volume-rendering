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

if (DEFINED ENV{VTK_CMAKE_CONFIG_DIR})
  SET(VTK_DIR "$ENV{VTK_CMAKE_CONFIG_DIR}")
else()
  SET(VTK_DIR ${COSMOSCOUT_EXTERNALS_DIR}/lib/cmake/vtk-8.1)
endif()

if (DEFINED ENV{OSPRAY_CMAKE_CONFIG_DIR})
  SET(ospray_DIR "$ENV{OSPRAY_CMAKE_CONFIG_DIR}")
else()
  SET(ospray_DIR ${COSMOSCOUT_EXTERNALS_DIR}/lib/cmake/ospray-2.3.0)
endif()

if (DEFINED ENV{EMBREE_CMAKE_CONFIG_DIR})
  SET(embree_DIR "$ENV{EMBREE_CMAKE_CONFIG_DIR}")
else()
  SET(embree_DIR ${COSMOSCOUT_EXTERNALS_DIR})
endif()

if (DEFINED ENV{RKCOMMON_CMAKE_CONFIG_DIR})
  SET(rkcommon_DIR "$ENV{RKCOMMON_CMAKE_CONFIG_DIR}")
else()
  SET(rkcommon_DIR ${COSMOSCOUT_EXTERNALS_DIR}/lib/cmake/rkcommon-1.5.0)
endif()

if (DEFINED ENV{OPENVKL_CMAKE_CONFIG_DIR})
  SET(openvkl_DIR "$ENV{OPENVKL_CMAKE_CONFIG_DIR}")
else()
  SET(openvkl_DIR ${COSMOSCOUT_EXTERNALS_DIR}/lib/cmake/openvkl-0.10.0)
endif()

if (DEFINED ENV{OIDN_CMAKE_CONFIG_DIR})
  SET(OpenImageDenoise_DIR "$ENV{OIDN_CMAKE_CONFIG_DIR}")
else()
  SET(OpenImageDenoise_DIR ${COSMOSCOUT_EXTERNALS_DIR}/lib/cmake/OpenImageDenoise)
endif()

if (DEFINED ENV{RKCOMMON_TBB_ROOT})
  SET(RKCOMMON_TBB_ROOT "$ENV{RKCOMMON_TBB_ROOT}")
else()
  SET(RKCOMMON_TBB_ROOT ${COSMOSCOUT_EXTERNALS_DIR})
endif()

if (DEFINED ENV{ISPC_EXECUTABLE})
  SET(ISPC_EXECUTABLE "$ENV{ISPC_EXECUTABLE}")
else()
  SET(ISPC_EXECUTABLE ${COSMOSCOUT_EXTERNALS_DIR}/bin/ispc.exe)
endif()

find_package(ospray REQUIRED)
find_package(OpenImageDenoise REQUIRED)
find_package(VTK REQUIRED)

# build plugin -------------------------------------------------------------------------------------

file(GLOB_RECURSE SOURCE_FILES src/*.cpp)
# Resource files and header files are only added in order to make them available in your IDE.
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

add_subdirectory(ospray_modules)
