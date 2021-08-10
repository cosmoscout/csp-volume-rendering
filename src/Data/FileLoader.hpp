////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef CSP_VOLUME_RENDERING_FILELOADER_HPP
#define CSP_VOLUME_RENDERING_FILELOADER_HPP

#include "../logger.hpp"

#include <vtkDataSet.h>
#include <vtkSmartPointer.h>

namespace csp::volumerendering {

class FileLoader {
 public:
  virtual vtkSmartPointer<vtkDataSet> loadDataImpl(std::string const& file) = 0;
};

} // namespace csp::volumerendering

#endif // CSP_VOLUME_RENDERING_FILELOADER_HPP
