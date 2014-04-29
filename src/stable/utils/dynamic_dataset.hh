#ifndef _UTILS_DYNAMIC_DATASET_HH_
#define _UTILS_DYNAMIC_DATASET_HH_


//Local includes
#include "data_image.hh"
#include "dataset.hh"
#include "color_coding.hh"
#include "configuration.hh"

// PCL includes
#include <pcl/common/common.h>

// Eigen includes
#include <eigen3/Eigen/Geometry>

namespace Utils {

class DynamicDataset : public virtual Dataset{
  public:
    DynamicDataset(Configuration conf);
    virtual ~DynamicDataset() {};

    virtual void SetLoadFlags(int load_flags) = 0;
    virtual Utils::DataImage GenerateImage(const pcl::PointCloud<pcl::PointXYZRGB>& cloud, const Eigen::Matrix3f& q) = 0;

    static DynamicDataset* LoadDataset(const std::string& config_filename);
    static DynamicDataset* LoadDataset(Configuration config);

  protected:
    virtual void Load() = 0;

};
}

#endif // _UTILS_DYNAMIC_DATASET_HH_
