#ifndef CV_UTIL_HH
#define CV_UTIL_HH

#include <opencv2/core/core.hpp>
#include <string>

namespace Utils{
void SaveMat(const std::string& filename, const cv::Mat& data);
void ReadMat(const std::string& filename, cv::Mat& data);

void ShowCvMat(const cv::Mat& m);
}
#endif // CV_UTIL_HH
