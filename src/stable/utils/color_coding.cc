#include "color_coding.hh"
using namespace Utils;

ColorCoding::ColorCoding(){  
}

cv::Mat ColorCoding::BgrToLabel(const cv::Mat& bgr) const{
  cv::Mat result(bgr.rows, bgr.cols, CV_8UC1);
  for(unsigned int y=0; y < bgr.rows; ++y){
    const unsigned char *r = bgr.ptr<unsigned char>(y);
    for(unsigned int x=0; x < bgr.cols; ++x, r += 3){
      result.at<signed char>(y,x) = BgrToLabel(r);
    }
  }
  return result;
}

cv::Mat ColorCoding::LabelToBgr(const cv::Mat& label) const{
  cv::Mat result(label.rows, label.cols, CV_8UC3);
  for(unsigned int y=0; y < label.rows; ++y){
    for(unsigned int x=0; x < label.cols; ++x){
      result.at<cv::Vec3b>(y,x) = LabelToBgr(label.at<signed char>(y,x));
    }
  }
  return result;
}
