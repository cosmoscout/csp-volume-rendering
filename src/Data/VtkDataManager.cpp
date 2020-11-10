////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "VtkDataManager.hpp"

#include "../logger.hpp"

#include <ViracochaBackend/DataManager/VrcGenericDataLoader.h>

namespace csp::volumerendering {

////////////////////////////////////////////////////////////////////////////////////////////////////

VtkDataManager::VtkDataManager(std::string path, std::string filenamePattern)
    : DataManager(path, filenamePattern) {
  initState();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

vtkSmartPointer<vtkDataSet> VtkDataManager::loadDataImpl(int timestep) {
  return VrcGenericDataLoader::LoadVtkDataSet(mTimestepFiles[timestep].c_str());
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace csp::volumerendering
