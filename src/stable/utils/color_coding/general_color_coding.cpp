#include "general_color_coding.h"

using namespace Utils;

GeneralColorCoding::GeneralColorCoding(){}

signed char GeneralColorCoding::BgrToLabel(const unsigned char* bgr) const{
  unsigned char r,g,b;
  unsigned char lab = 0;
  b = bgr[0];
  g = bgr[1];
  r = bgr[2];
  for (int i = 0; i < 8; i++) {
    lab = (lab << 3) | (((r >> i) & 1) << 0) | (((g >> i) & 1) << 1) | (((b >> i) & 1) << 2);
  }
  return lab;
}

cv::Vec3b GeneralColorCoding::LabelToBgr(signed char label) const{
  unsigned char r,g,b =0;
  unsigned char lab = label;
  for (int i = 0; lab > 0; i++, lab >>= 3) {
    r |= (unsigned char) (((lab >> 0) & 1) << (7 - i));
    g |= (unsigned char) (((lab >> 1) & 1) << (7 - i));
    b |= (unsigned char) (((lab >> 2) & 1) << (7 - i));
  }
  return cv::Vec3b(b,g,r);
}