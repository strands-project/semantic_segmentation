#ifndef GENERALCOLORCODING_H
#define GENERALCOLORCODING_H

#include "../color_coding.hh"

#include <opencv2/core/core.hpp>

namespace Utils{
  
class GeneralColorCoding : public ColorCoding{

public:
    GeneralColorCoding();

private:
  signed char BgrToLabel(const unsigned char *bgr) const;
  cv::Vec3b LabelToBgr(signed char label) const;
};
}
#endif // GENERALCOLORCODING_H
