////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef CSP_VOLUME_RENDERING_NETCDFDATAMANAGER_HPP
#define CSP_VOLUME_RENDERING_NETCDFDATAMANAGER_HPP

#include "../logger.hpp"

#include "FileLoader.hpp"

namespace csp::volumerendering {

class NetCdfFileLoader : public FileLoader {
 public:
 protected:
  vtkSmartPointer<vtkDataSet> loadDataImpl(std::string const& file) override;
};

} // namespace csp::volumerendering

#endif // CSP_VOLUME_RENDERING_NETCDFDATAMANAGER_HPP
