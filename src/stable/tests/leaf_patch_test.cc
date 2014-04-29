// Random forests includes
#include <randomized_decision_forest/tree.hh>
#include <randomized_decision_forest/datasample.hh>

// OpenCV includes
#include <opencv2/core/core.hpp>

int main(int argc, char *argv[]) {
  // Setup a basic tree node. Just use this, it will be a leaf
  // and the other methods are easy to test.
  Rdfs::Tree t;
  t.InitAverageLeafPatches(1);

  //Setup three basic images
  cv::Mat a(3,3, CV_8UC3, cv::Scalar(0));
  a.at<cv::Vec3b>(0,0) = cv::Vec3b(255, 255, 255);
  a.at<cv::Vec3b>(0,1) = cv::Vec3b(255, 255, 255);
  a.at<cv::Vec3b>(0,2) = cv::Vec3b(255, 255, 255);
  cv::Mat b(3,3, CV_8UC3, cv::Scalar(0));
  b.at<cv::Vec3b>(1,0) = cv::Vec3b(255, 255, 255);
  b.at<cv::Vec3b>(1,1) = cv::Vec3b(255, 255, 255);
  b.at<cv::Vec3b>(1,2) = cv::Vec3b(255, 255, 255);
  cv::Mat c(3,3, CV_8UC3, cv::Scalar(0));
  c.at<cv::Vec3b>(2,0) = cv::Vec3b(255, 255, 255);
  c.at<cv::Vec3b>(2,1) = cv::Vec3b(255, 255, 255);
  c.at<cv::Vec3b>(2,2) = cv::Vec3b(255, 255, 255);
  cv::Mat d(3,3, CV_8UC3, cv::Scalar(255, 255, 255));
  

  //Setup a some basic datasamples, needed for determining the middle points.
  Rdfs::DataSample x(0, 1, 1, 0, 0.0, 0);
  Rdfs::DataSample y(0, 0, 0, 0, 0.0, 0);
  
  //Add the images
  t.AddImageToLeafPatch(a, &x);
  t.AddImageToLeafPatch(b, &x);
  t.AddImageToLeafPatch(c, &x);
  t.AddImageToLeafPatch(d, &y);

  //Extract average
  cv::Mat res = t.FinalizeLeafPatch();
  
  cv::Vec3b gray(static_cast<unsigned char>(255.0f/3.0f), static_cast<unsigned char>(255.0f/3.0f), static_cast<unsigned char>(255.0f/3.0f));
  cv::Vec3b dark_gray(static_cast<unsigned char>(510.0f/4.0f), static_cast<unsigned char>(510.0f/4.0f), static_cast<unsigned char>(510.0f/4.0f));
  for(int y = 0; y < 3; y++){
    for(int x = 0; x < 3; x++){
      if((y==0 && res.at<cv::Vec3b>(y,x) != gray) ||
         (x==0 && res.at<cv::Vec3b>(y,x) != gray) ||
         (x!=0 && y!=0 && res.at<cv::Vec3b>(y,x) != dark_gray)){
        std::cerr << "The leaf patches are not working!" << std::endl;
        std::cerr << x << " " << y << " "  <<  res.at<cv::Vec3b>(y,x) << std::endl;
        return EXIT_FAILURE;
      }
    }
  }
  std::cout << "The leaf patches are working just fine!" << std::endl;
  return EXIT_SUCCESS;
}
