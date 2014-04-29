#ifndef COLOR_CODING_HH
#define COLOR_CODING_HH

#include <opencv2/core/core.hpp>

namespace Utils {
class ColorCoding
{
public:
  ColorCoding();

  cv::Mat BgrToLabel(const cv::Mat& bgr) const;
  cv::Mat LabelToBgr(const cv::Mat& label) const;

protected:
  virtual signed char BgrToLabel(const unsigned char *bgr) const =0;
  virtual cv::Vec3b LabelToBgr(signed char label) const =0;

};
}
#endif // COLOR_CODING_HH
