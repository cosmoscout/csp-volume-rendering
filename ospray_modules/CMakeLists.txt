# build ospray module ------------------------------------------------------------------------------

list(APPEND CMAKE_MODULE_PATH ${ospray_DIR})
include(ispc)
include(GenerateExportHeader)

get_target_property(OSPRAY_ISPC_MODULE_INCLUDE_DIRS ospray::ospray_module_ispc INTERFACE_INCLUDE_DIRECTORIES)

ispc_include_directories(${OSPRAY_ISPC_MODULE_INCLUDE_DIRS})

set(MODULE_SOURCE_FILES
  volume_depth/ospray/moduleInit.cpp
	volume_depth/ospray/render/VolumeDepth.cpp
	volume_depth/ospray/render/VolumeDepth.ispc
	volume_depth/ospray/render/volumes.ispc
	volume_depth/ospray/render/surfaces.ispc
	volume_depth/ospray/render/lightAlpha.ispc
	volume_depth/ospray/math/halton.ispc
	volume_depth/ospray/math/sobol.ispc
)
set(MODULE_HEADER_FILES
  volume_depth/ospray/render/VolumeDepth.hpp
	volume_depth/ospray/render/VolumeDepth.ih
	volume_depth/ospray/render/volumes.ih
	volume_depth/ospray/render/surfaces.ih
)

add_library(ospray_module_volume_depth SHARED)

ispc_target_add_sources(ospray_module_volume_depth
  ${MODULE_SOURCE_FILES}
	${MODULE_HEADER_FILES}
)
target_include_directories(ospray_module_volume_depth PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/volume_depth/ospray)

target_sources(ospray_module_volume_depth PRIVATE ${MODULE_SOURCE_FILES})

generate_export_header(ospray_module_volume_depth)

target_link_libraries(ospray_module_volume_depth
  PUBLIC
    ospray::ospray_module_ispc
    ospray::ospray_sdk
)

source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR}/volume_depth FILES
  ${MODULE_SOURCE_FILES} ${MODULE_HEADER_FILES}
)

install(TARGETS ospray_module_volume_depth DESTINATION "lib")
