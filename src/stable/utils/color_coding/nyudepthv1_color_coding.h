#ifndef NYUDEPTHV1_COLOR_CODING_HH
#define NYUDEPTHV1_COLOR_CODING_HH

#include "../color_coding.hh"

#include <opencv2/core/core.hpp>

namespace Utils{

class NyudepthV1ColorCoding : public ColorCoding
{
public:
  NyudepthV1ColorCoding();

private:
  signed char BgrToLabel(const unsigned char *bgr) const;
  cv::Vec3b LabelToBgr(signed char label) const;
};
}
#endif // NYUDEPTHV1_COLOR_CODING_HH
