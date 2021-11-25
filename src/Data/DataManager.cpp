////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "DataManager.hpp"

#include "../logger.hpp"
#include "NetCDFFileLoader.hpp"
#include "VtkFileLoader.hpp"

#include "../../../../src/cs-utils/convert.hpp"
#include "../../../../src/cs-utils/filesystem.hpp"
#include "../../../../src/cs-utils/utils.hpp"

#include <vtkCellData.h>
#include <vtkPointData.h>
#include <vtkStructuredGrid.h>

#include <glm/gtc/type_ptr.hpp>
#include <glm/gtx/norm.hpp>

#include <algorithm>
#include <future>
#include <random>
#include <regex>

namespace csp::volumerendering {

////////////////////////////////////////////////////////////////////////////////////////////////////

const char* DataManagerException::what() const noexcept {
  return "Failed to initialize DataManager.";
}

////////////////////////////////////////////////////////////////////////////////////////////////////

DataManager::DataManager(Settings::Data const& dataSettings)
    : mDataSettings(dataSettings) {
  // Create appropriate file loader
  switch (mDataSettings.mType.get()) {
  case VolumeFileType::eVtk:
    mFileLoader = std::make_unique<VtkFileLoader>();
    break;
  case VolumeFileType::eNetCdf:
    mFileLoader = std::make_unique<VtkFileLoader>();
    break;
  default:
    logger().error("Invalid volume data type given in settings! Should be 'vtk' or 'netcdf'.");
    throw DataManagerException();
    break;
  }

  // Parse regex
  std::regex patternRegex;
  try {
    patternRegex = std::regex(".*" + mDataSettings.mNamePattern.get());
  } catch (const std::regex_error& e) {
    logger().error("Filename pattern '{}' is not a valid regular expression: {}",
        mDataSettings.mNamePattern.get(), e.what());
    throw DataManagerException();
  }

  // Check if regex has capture group for lod
  bool haveLodFiles = false;
  if (patternRegex.mark_count() == 2) {
    haveLodFiles = true;
  } else if (patternRegex.mark_count() != 1) {
    logger().error(
        "Filename pattern '{}' has the wrong amount of capture groups: {}! The pattern should "
        "contain only one capture group capturing the timestep component of the filename.",
        mDataSettings.mNamePattern.get(), patternRegex.mark_count());
    throw DataManagerException();
  }

  // Get all files matching the regex
  std::set<std::string> files;
  try {
    files = cs::utils::filesystem::listFiles(mDataSettings.mPath.get(), patternRegex);
  } catch (const boost::filesystem::filesystem_error& e) {
    logger().error("Loading volume data from '{}' failed: {}", mDataSettings.mPath.get(), e.what());
    throw DataManagerException();
  }

  // Try to get csv data
  std::string csvPattern = mDataSettings.mNamePattern.get();
  cs::utils::replaceString(csvPattern, boost::filesystem::extension(csvPattern), ".csv");
  try {
    std::set<std::string> files =
        cs::utils::filesystem::listFiles(mDataSettings.mPath.get(), std::regex(".*" + csvPattern));
    if (files.size() == 0) {
      logger().error("No csv data found in '{}'!", mDataSettings.mPath.get());
      throw DataManagerException();
    }
    mCsvData = cs::utils::filesystem::loadToString(*files.begin());
  } catch (const boost::filesystem::filesystem_error& e) {
    logger().error("Loading csv data from '{}' failed: {}", mDataSettings.mPath.get(), e.what());
    throw DataManagerException();
  }

  // Sort the data files by timestep and lod
  std::vector<Timestep> timesteps;
  for (std::string file : files) {
    file = std::regex_replace(file, std::regex(R"(\\)"), "/");
    std::smatch match;
    std::regex_search(file, match, patternRegex);

    Timestep timestep;
    Lod      lod;
    try {
      timestep = std::stoi(match[haveLodFiles ? 2 : 1].str());
    } catch (const std::invalid_argument&) {
      logger().error("Capture group in filename pattern '{}' does "
                     "not match an integer for file "
                     "'{}': Match of group is '{}'! A suitable "
                     "capture group could be '([0-9])+'.",
          mDataSettings.mNamePattern.get(), file, match[haveLodFiles ? 2 : 1].str());
      throw DataManagerException();
    }
    try {
      lod = haveLodFiles ? std::stoi(match[1].str()) : 0;
    } catch (const std::invalid_argument&) {
      logger().error("Capture group in filename pattern '{}' does "
                     "not match an integer for file "
                     "'{}': Match of group is '{}'! A suitable "
                     "capture group could be '([0-9])+'.",
          mDataSettings.mNamePattern.get(), file, match[1].str());
      throw DataManagerException();
    }

    timesteps.push_back(timestep);
    mFiles[timestep][lod] = file;
  }
  if (timesteps.size() == 0) {
    logger().error("No files matching '{}' found in '{}'!", mDataSettings.mNamePattern.get(),
        mDataSettings.mPath.get());
    throw DataManagerException();
  }
  pTimesteps.set(timesteps);

  // Initialize state
  Timestep timestep;
  {
    std::scoped_lock lock(mStateMutex);
    timestep = pTimesteps.get()[0];
  }
  setTimestep(timestep);
  mInitScalarsThread = std::thread(&DataManager::initScalars, this);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void DataManager::addPathlines(Settings::Pathlines const& pathlinesSettings) {
  mPathlines = std::make_unique<Pathlines>(pathlinesSettings.mPath.get());
}

////////////////////////////////////////////////////////////////////////////////////////////////////

DataManager::~DataManager() {
  if (mInitScalarsThread.joinable()) {
    mInitScalarsThread.join();
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

bool DataManager::isReady() {
  std::scoped_lock lock(mScalarsMutex);
  return pScalars.get().size() > 0;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

cs::utils::Signal<Scalar const&> const& DataManager::onScalarRangeUpdated() const {
  return mOnScalarRangeUpdated;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::array<double, 2> DataManager::getScalarRange(Scalar const& scalar) {
  std::scoped_lock lock(mStateMutex);
  return mScalarRanges[scalar.getId()];
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::array<double, 2> DataManager::getScalarRange(std::string scalarId) {
  std::scoped_lock lock(mStateMutex);
  return mScalarRanges[scalarId];
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void DataManager::setTimestep(int timestep) {
  {
    std::scoped_lock lock(mStateMutex);
    if (std::find(pTimesteps.get().begin(), pTimesteps.get().end(), timestep) !=
        pTimesteps.get().end()) {
      mCurrentTimestep = timestep;
    } else {
      logger().warn("'{}' is not a timestep in the current "
                    "dataset. '{}' will be used instead.",
          timestep, mCurrentTimestep);
      timestep = mCurrentTimestep;
    }
  }
  mTimestepCv.notify_all();
  getFromCache(timestep);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void DataManager::cacheTimestep(int timestep) {
  if (std::find(pTimesteps.get().begin(), pTimesteps.get().end(), timestep) ==
      pTimesteps.get().end()) {
    logger().warn("'{}' is not a timestep in the current dataset. "
                  "No timestep will be cached.",
        timestep);
    return;
  }
  getFromCache(timestep);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void DataManager::setActiveScalar(std::string const& scalarId) {
  std::scoped_lock lock(mStateMutex, mScalarsMutex);
  auto             scalar = std::find_if(pScalars.get().begin(), pScalars.get().end(),
      [&scalarId](Scalar const& s) { return s.getId() == scalarId; });
  if (scalar != pScalars.get().end()) {
    mActiveScalar = *scalar;
  } else {
    logger().warn("'{}' is not a scalar in the current dataset. "
                  "'{}' will be used instead.",
        scalarId, mActiveScalar.getId());
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::string const& DataManager::getCsvData() {
  return mCsvData;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

DataManager::Metadata DataManager::calculateMetadata() {
  Metadata meta;
  switch (mDataSettings.mStructure.get()) {
  case VolumeStructure::eStructuredSpherical: {
    Metadata::StructuredSpherical      metadata;
    vtkSmartPointer<vtkStructuredGrid> volume = vtkStructuredGrid::SafeDownCast(getData());

    glm::dvec3            origin;
    std::array<double, 6> bounds;
    std::array<int, 3>    dimensions;
    volume->GetPoint(0, 0, 0, glm::value_ptr(origin));
    volume->GetBounds(bounds.data());
    volume->GetDimensions(dimensions.data());

    std::array<glm::dvec3, 3> increments;
    volume->GetPoint(1, 0, 0, glm::value_ptr(increments[0]));
    volume->GetPoint(0, 1, 0, glm::value_ptr(increments[1]));
    volume->GetPoint(0, 0, 1, glm::value_ptr(increments[2]));
    std::array<glm::dvec3, 3> maxPoints;
    volume->GetPoint(dimensions[0] - 1, 0, 0, glm::value_ptr(maxPoints[0]));
    volume->GetPoint(0, dimensions[1] - 1, 0, glm::value_ptr(maxPoints[1]));
    volume->GetPoint(0, 0, dimensions[2] - 1, glm::value_ptr(maxPoints[2]));

    // Get the radial axis in the VTK data.
    // This should be the axis, where each step results in the biggest movement relative to
    // the center of the carthesian coordinates.
    double maxDist = 0.;
    for (int i = 0; i < 3; i++) {
      double dist    = glm::length(increments[i]) - glm::length(origin);
      double distAbs = std::abs(dist);
      if (distAbs > maxDist) {
        maxDist             = distAbs;
        metadata.mAxes.mRad = i;
      }
    }
    metadata.mRanges.mRad = {glm::length(origin), glm::length(maxPoints[metadata.mAxes.mRad])};

    // Check the other two axes to see which is the latitudinal and which is the longitudinal axis.
    for (int axisOffset = 1; axisOffset < 3; axisOffset++) {
      // Grid axis, that is checked in this iteration
      int axis = (metadata.mAxes.mRad + axisOffset) % 3;
      // Grid axis, that is neither the radial, nor the currently checked grid axis
      int otherAxis = (metadata.mAxes.mRad + 3 - axisOffset) % 3;

      std::array<double, 3>                angles = {0., 0., 0.};
      std::array<std::array<double, 3>, 2> maxDiffs;

      // Do the check for two times with different positions along the other axis.
      for (int i = 0; i < 2; i++) {
        maxDiffs[i] = {0, 0, 0};
        glm::dvec3         point;
        glm::dvec3         pointPrev;
        std::array<int, 3> pos = {0, 0, 0};
        pos[otherAxis]         = i;

        // Iterate over all points of the checked axis, with constant values for the other two axes.
        // Keep track of the total angle of rotation along each of the carthesian axes, as well as
        // the maximal translation along each of the carthesian axes between two neighboured points.
        volume->GetPoint(pos[0], pos[1], pos[2], glm::value_ptr(point));
        for (int j = 1; j < dimensions[axis]; j++) {
          pointPrev = point;
          switch (axis) {
          case 0:
            volume->GetPoint(j, pos[1], pos[2], glm::value_ptr(point));
            break;
          case 1:
            volume->GetPoint(pos[0], j, pos[2], glm::value_ptr(point));
            break;
          case 2:
            volume->GetPoint(pos[0], pos[1], j, glm::value_ptr(point));
            break;
          }
          for (int k = 0; k < 3; k++) {
            maxDiffs[i][k] = std::max(maxDiffs[i][k], std::abs(pointPrev[k] - point[k]));
          }
          if (i == 0) {
            angles[0] += glm::acos(glm::dot(glm::normalize(point * glm::dvec3(0, 1, 1)),
                glm::normalize(pointPrev * glm::dvec3(0, 1, 1))));
            angles[1] += glm::acos(glm::dot(glm::normalize(point * glm::dvec3(1, 0, 1)),
                glm::normalize(pointPrev * glm::dvec3(1, 0, 1))));
            angles[2] += glm::acos(glm::dot(glm::normalize(point * glm::dvec3(1, 1, 0)),
                glm::normalize(pointPrev * glm::dvec3(1, 1, 0))));
          }
        }
      }

      // Check, for which carthesian axis there was no translation when iterating over all points.
      // If there was no translation, the points on the currently checked grid axis rotate around
      // the carthesian axis. This carthesian axis is a candidate for the up axis.
      std::array<int, 2> zeroAxes = {-1, -1};
      for (int i = 0; i < 2; i++) {
        for (int k = 0; k < 3; k++) {
          if (maxDiffs[i][k] == 0) {
            zeroAxes[i] = k;
          }
        }
      }

      // If the points rotate around the same carthesian axis for both iterations, the currently
      // checked grid axis probably is the longitudinal axis.
      if (zeroAxes[0] == zeroAxes[1] && zeroAxes[0] != -1) {
        metadata.mAxes.mLon = axis;
        metadata.mAxes.mLat = otherAxis;

        int    upAxis   = -1;
        double maxAngle = 0.;
        for (int i = 0; i < 3; i++) {
          if (angles[i] > maxAngle) {
            maxAngle = angles[i];
            upAxis   = i;
          }
        }
        maxAngle = cs::utils::convert::toDegrees(maxAngle);

        glm::dvec3 up{0., 0., 0.};
        up[upAxis] = 1.;

        // Check the direction of rotation around the up axis.
        bool flipped = glm::dot(glm::cross(origin, increments[metadata.mAxes.mLon]), up) < 0.;
        if (flipped) {
          metadata.mRanges.mLon = {maxAngle, 0.};
        } else {
          metadata.mRanges.mLon = {0., maxAngle};
        }

        // The minimum and maximum latitude is determined by calculating the angle between the down
        // vector and the first and last point on the latitudinal grid axis.
        metadata.mRanges.mLat = {
            cs::utils::convert::toDegrees(glm::acos(glm::dot(glm::normalize(origin), -up))),
            cs::utils::convert::toDegrees(
                glm::acos(glm::dot(glm::normalize(maxPoints[metadata.mAxes.mLat]), -up)))};

        break;
      }
    }

    meta.mStructuredSpherical = metadata;
    break;
  }
  }
  return meta;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

DataManager::Metadata::StructuredSpherical const& DataManager::getMetadata() {
  assert(("getMetadata must not be called when metadata is not set in the plugin settings.",
      mDataSettings.mMetadata.has_value()));
  return mDataSettings.mMetadata->mStructuredSpherical;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::future<std::vector<float>> DataManager::getSample(
    State state, std::chrono::high_resolution_clock::duration duration) {
  return std::async(
      std::launch::async, [this, state = std::move(state), duration = std::move(duration)]() {
        vtkSmartPointer<vtkDataArray> data;
        switch (state.mScalar.mType) {
        case ScalarType::eCellData:
          data = getData(state)->GetCellData()->GetScalars(state.mScalar.mName.c_str());
          break;
        case ScalarType::ePointData:
          data = getData(state)->GetPointData()->GetScalars(state.mScalar.mName.c_str());
          break;
        }

        std::random_device                       rd;
        std::default_random_engine               gen(rd());
        std::uniform_int_distribution<vtkIdType> indices(0, data->GetNumberOfTuples() - 1);

        std::chrono::high_resolution_clock::time_point start =
            std::chrono::high_resolution_clock::now();
        std::vector<float> sample;
        while (std::chrono::high_resolution_clock::now() < start + duration) {
          sample.push_back(data->GetVariantValue(indices(gen)).ToFloat());
        }
        return sample;
      });
}

////////////////////////////////////////////////////////////////////////////////////////////////////

vtkSmartPointer<vtkDataSet> DataManager::getData(
    std::optional<State> optState, std::optional<int> optLod) {
  State state = optState.value_or(getState());

  std::shared_future<vtkSmartPointer<vtkDataSet>> futureData =
      getFromCache(state.mTimestep, optLod);
  vtkSmartPointer<vtkDataSet> dataset = futureData.get();
  if (!dataset) {
    logger().error("Loaded data is null! Is the data type "
                   "correctly set in the settings?");
    throw std::runtime_error("Loaded data is null.");
  }
  if (state.mScalar.mName != "") {
    switch (state.mScalar.mType) {
    case ScalarType::ePointData:
      dataset->GetPointData()->SetActiveScalars(state.mScalar.mName.c_str());
      break;
    case ScalarType::eCellData:
      dataset->GetCellData()->SetActiveScalars(state.mScalar.mName.c_str());
      break;
    }
  }
  return dataset;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

DataManager::State DataManager::getState() {
  std::scoped_lock lock(mStateMutex);
  State            state{mCurrentTimestep, mActiveScalar};
  return state;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

int DataManager::getMaxLod(State state, std::optional<int> max) {
  std::scoped_lock lock(mDataMutex);
  auto             cacheEntry = mCache.find(state.mTimestep);
  if (cacheEntry == mCache.end()) {
    return 0;
  }
  Lod maxLod = 0;
  for (auto const& lod : cacheEntry->second) {
    if (lod.second.wait_for(std::chrono::seconds(0)) == std::future_status::ready &&
        lod.first > maxLod) {
      if (max.has_value() && lod.first > max.value()) {
        continue;
      }
      maxLod = lod.first;
    }
  }
  return maxLod;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

int DataManager::getMinLod(State state) {
  std::scoped_lock lock(mDataMutex);
  auto             cacheEntry = mCache.find(state.mTimestep);
  if (cacheEntry == mCache.end()) {
    return 0;
  }
  Lod minLod = std::numeric_limits<int>::max();
  for (auto const& lod : cacheEntry->second) {
    if (lod.second.wait_for(std::chrono::seconds(0)) == std::future_status::ready &&
        lod.first < minLod) {
      minLod = lod.first;
    }
  }
  return minLod;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

Pathlines const& DataManager::getPathlines() const {
  assert(("getPathlines must not be called when pathlines are "
          "deactivated.",
      mPathlines));
  return *mPathlines;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void DataManager::initState() {
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void DataManager::initScalars() {
  vtkSmartPointer<vtkDataSet> data;
  try {
    data = getData();
  } catch (const std::exception&) {
    logger().error("Could not load initial data for determining "
                   "available scalars! "
                   "Requested data may have no active scalar.");
    return;
  }

  std::vector<Scalar> scalars;
  {
    std::scoped_lock lock(mStateMutex);
    for (int i = 0; i < data->GetPointData()->GetNumberOfArrays(); i++) {
      if (data->GetPointData()->GetAbstractArray(i)->GetNumberOfComponents() == 1) {
        Scalar scalar;
        scalar.mName = data->GetPointData()->GetArrayName(i);
        scalar.mType = ScalarType::ePointData;
        data->GetPointData()
            ->GetScalars(scalar.mName.c_str())
            ->GetRange(mScalarRanges[scalar.getId()].data());
        scalars.push_back(scalar);
      }
    }
    for (int i = 0; i < data->GetCellData()->GetNumberOfArrays(); i++) {
      if (data->GetCellData()->GetAbstractArray(i)->GetNumberOfComponents() == 1) {
        Scalar scalar;
        scalar.mName = data->GetCellData()->GetArrayName(i);
        scalar.mType = ScalarType::eCellData;
        data->GetCellData()
            ->GetScalars(scalar.mName.c_str())
            ->GetRange(mScalarRanges[scalar.getId()].data());
        scalars.push_back(scalar);
      }
    }
    if (scalars.size() == 0) {
      logger().error("No scalars found in volume data! Requested "
                     "data may have no active scalar.");
      return;
    }
    mActiveScalar = scalars[0];
  }
  {
    std::scoped_lock lock(mScalarsMutex);
    pScalars.setWithNoEmit(scalars);
  }
  pScalars.touch();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::shared_future<vtkSmartPointer<vtkDataSet>> DataManager::getFromCache(
    Timestep timestep, std::optional<Lod> optLod) {
  auto lods = mFiles.find(timestep)->second;
  Lod  lod  = optLod.value_or(getMaxLod({timestep, mActiveScalar}));

  std::scoped_lock lock(mDataMutex);
  auto             cacheEntry = mCache.find(timestep);
  if (cacheEntry == mCache.end()) {
    Lod lod = lods.lower_bound(0)->first;
    loadData(timestep, lod);
    return mCache[timestep][lod];
  }
  std::thread loadNext([this, lod, lods, timestep]() {
    {
      std::unique_lock<std::mutex> cvLock(mStateMutex);
      if (mTimestepCv.wait_for(cvLock, std::chrono::seconds(2),
              [this, timestep]() { return mCurrentTimestep != timestep; })) {
        return;
      }
    }
    std::scoped_lock lock(mDataMutex);
    auto             cacheEntry = mCache.find(timestep);
    auto             loadingLod = cacheEntry->second.upper_bound(lod);
    auto             nextLod    = lods.upper_bound(lod);
    if (loadingLod == cacheEntry->second.end() && nextLod != lods.end()) {
      loadData(timestep, nextLod->first);
    }
  });
  loadNext.detach();
  return mCache[timestep][lod];
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void DataManager::loadData(Timestep timestep, Lod lod) {
  logger().info("Loading data for timestep {}, level of detail {}...", timestep, lod);
  auto data = std::async(
      std::launch::async,
      [this](Timestep timestep, Lod lod) {
        std::chrono::high_resolution_clock::time_point timer;
        vtkSmartPointer<vtkDataSet>                    data;
        {
          std::scoped_lock lock(mReadMutex);
          timer = std::chrono::high_resolution_clock::now();
          data  = mFileLoader->loadDataImpl(mFiles[timestep][lod]);
        }

        std::map<Scalar, std::array<std::optional<double>, 2>> updatedScalars;
        std::vector<Scalar>                                    scalars;
        {
          std::scoped_lock lock(mScalarsMutex);
          scalars = pScalars.get();
        }
        for (auto const& scalar : scalars) {
          std::array<double, 2> range;
          switch (scalar.mType) {
          case ScalarType::ePointData:
            data->GetPointData()->GetScalars(scalar.mName.c_str())->GetRange(range.data());
            break;
          case ScalarType::eCellData:
            data->GetCellData()->GetScalars(scalar.mName.c_str())->GetRange(range.data());
            break;
          }
          if (range[0] < mScalarRanges[scalar.getId()][0]) {
            updatedScalars[scalar][0] = range[0];
          }
          if (range[1] > mScalarRanges[scalar.getId()][1]) {
            updatedScalars[scalar][1] = range[1];
          }
        }
        {
          std::scoped_lock lock(mStateMutex);
          for (auto const& [scalar, value] : updatedScalars) {
            if (value[0].has_value()) {
              mScalarRanges[scalar.getId()][0] = value[0].value();
            }
            if (value[1].has_value()) {
              mScalarRanges[scalar.getId()][1] = value[1].value();
            }
          }
        }
        for (auto const& scalar : updatedScalars) {
          mOnScalarRangeUpdated.emit(scalar.first);
        }

        logger().info("Finished loading data for timestep {}, "
                      "level of detail {}. Took {}s.",
            timestep, lod,
            (float)(std::chrono::high_resolution_clock::now() - timer).count() / 1000000000);

        return data;
      },
      timestep, lod);
  mCache[timestep][lod] = std::move(data);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace csp::volumerendering
