#ifndef _UTILS_DATASET_HH_
#define _UTILS_DATASET_HH_

//Local includes
#include "color_coding.hh"

namespace Utils {

enum LoadRequirements {
  RGB =              1 << 0,
  LAB =              1 << 1,
  DEPTH =            1 << 2,
  ANNOTATION =       1 << 3,
  UNARY =            1 << 4,
  UNARY2 =           1 << 5,
  ACCELEROMETER =    1 << 6,
  GRADIENT_COLOR =   1 << 7,
  GRADIENT_DEPTH =   1 << 8,
  GEOMETRIC_FEAT =   1 << 9,
  DEPTH_COVARIANCE = 1 << 10,
  LAB_INTEGRAL     = 1 << 11,
  NORMALS =          1 << 12
}; // If this is changed, don't forget to update the loop printing the things computed! In both dataset types!

enum DatasetType {
  MSRC = 1,
  NYUDEPTH_V1 = 2,
  CITY = 3,
  LEUVEN = 4,
  NYUDEPTH_V2 = 5,
  CAMVID = 6,
  KITTI = 7,
  STRANDS = 8
};

void DisplayLoadFlags(int load_flags);

class Dataset {
  public:
    const ColorCoding* GetColorCoding() const;
  protected:
    ColorCoding*                    m_color_coding;
    int                             m_load_requirement_flags;
};

bool t_sort(std::string i, std::string j);

}
#endif // _UTILS_DATASET_HH_
