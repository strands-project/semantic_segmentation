// STL includes
#include <vector>
#include <string>

// ROS includes
#include <ros/ros.h>
#include <tf/transform_listener.h>
#include <tf/message_filter.h>
#include <tf_conversions/tf_eigen.h>
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <sensor_msgs/CameraInfo.h>

// PCL includes
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl_ros/point_cloud.h>
#include <pcl/conversions.h>

// C includes
#include <dirent.h>
#include <cmath>
#include <fstream>

// OpenCV includes
#include <opencv2/highgui/highgui.hpp>

// Semantic includes
#include "utils/data_image.hh"
#include "utils/dynamic_dataset.hh"
#include "inference/densecrf_evaluator.hh"
#include "randomized_decision_forest/forest.hh"

//Do the subscribers.

class PointReader{    
    
  ros::NodeHandle nh;
  tf::TransformListener tf;
  tf::MessageFilter<sensor_msgs::PointCloud2> * tf_filter;
  message_filters::Subscriber<sensor_msgs::PointCloud2> cloud_sub;
  std::string cloud_topic; 
  std::string transform_topic1; 
  std::string transform_topic2; 
  Utils::DynamicDataset* data;
  Utils::Configuration config;
  Rdfs::Forest * forest;
  Inference::DenseCRFEvaluator crf_evaluator;
  ros::Publisher pub;
  ros::Subscriber cam_info_sub;

public:
  PointReader(std::string cloud_topic, std::string transform_topic1, std::string transform_topic2, std::string config_file, std::string forest_file) : tf(), cloud_topic(cloud_topic), transform_topic1(transform_topic1), transform_topic2(transform_topic2){
    cloud_sub.subscribe(nh, cloud_topic, 10);
    tf_filter = new tf::MessageFilter<sensor_msgs::PointCloud2>(cloud_sub, tf, transform_topic1, 10);
    tf_filter->registerCallback( boost::bind(&PointReader::PointCloudCallback, this, _1) );
    
    //Subscribe to camera info callback. 
    cam_info_sub = nh.subscribe("/head_xtion/rgb/camera_info", 1, &PointReader::CameraInfoCallback, this);

    //Load the config.
    std::cout << "Loading from: " << config_file << std::endl;

    //Setup the dataset.
    data = Utils::DynamicDataset::LoadDataset(config_file);
    int load_requirements = 0;

    //Load the forest.
    forest = Rdfs::Forest::LoadByFilename(forest_file);
    load_requirements |= forest->GetFeatureRequirements();
    
    //setup the crf evaluator.
    crf_evaluator = Inference::DenseCRFEvaluator(config_file);
    load_requirements |= crf_evaluator.GetLoadRequirements();
    load_requirements = load_requirements & ~Utils::UNARY; //Remove the unary if it was included, as this cannot be loaded directly here.


    //Set the load requirements for the dataset, specified by the other components.
    data->SetLoadFlags(load_requirements);

    pub = nh.advertise<sensor_msgs::PointCloud2> ("output", 1);
  }

private:
  //Camera info callback. When this is called set the calibration and 
  //shutdown the subscriber. 
  void CameraInfoCallback(const sensor_msgs::CameraInfo::ConstPtr& msg){
    Utils::DataImage::m_fx_rgb = msg->K[0];
    Utils::DataImage::m_fy_rgb = msg->K[4];
    Utils::DataImage::m_fx_rgb_inv = 1.0f/ msg->K[0]; 
    Utils::DataImage::m_fy_rgb_inv = 1.0f/ msg->K[4];
    Utils::DataImage::m_cx_rgb = msg->K[2];
    Utils::DataImage::m_cy_rgb = msg->K[5];
    Utils::DataImage::m_has_calibration = true;
    cam_info_sub.shutdown(); 
    //We ignore the depth calibration as this is not needed here because 
    //everyting is already reprojected.  
  }

  //callback
  void PointCloudCallback(const sensor_msgs::PointCloud2::ConstPtr& sensor_msg){
    //Convert so that we can use the cloud.
    pcl::PointCloud<pcl::PointXYZRGB> cloud;
    pcl::fromROSMsg(*sensor_msg, cloud);
   
    //Needed as somewhere the correct size of the cloud gets lost.
    cloud.width = 640;
    cloud.height = 480; 

    //Parse the transform. 
    tf::StampedTransform t;
    ros::Time now = sensor_msg->header.stamp;
    tf.waitForTransform(transform_topic1, transform_topic2, now, ros::Duration(1.0));
    tf.lookupTransform(transform_topic1, transform_topic2, now, t);
    tf::Quaternion q = t.getRotation();
    Eigen::Quaterniond q_eigen;
    tf::quaternionTFToEigen(q, q_eigen);	
    
    //std::cout <<  q_eigen.matrix() << std::endl;
    //We just want to tilt angle. Here comes the evil magic...
    Eigen::Matrix3d rot = q_eigen.matrix();
    float angle = (asin(rot(1)) + -asin(rot(6)))/2.0;
    Eigen::Matrix3f transformation = Eigen::Matrix3f::Identity();
    transformation(4) = cos(angle);
    transformation(5) = sin(angle);
    transformation(7) = -transformation(5);
    transformation(8) = transformation(4);
    //std::cout << angle << std::endl;
    // std::cout << transformation << std::endl;
    
    //Do the actual computations here. 
    Utils::DataImage current_image; 
    current_image = data->GenerateImage(cloud, transformation);
    cv::Mat result_unary;
    forest->GetUnaryPotential(current_image, &result_unary, true);
    current_image.AddUnaryPotential(result_unary);
    cv::Mat segmentation_result(current_image.Height(), current_image.Width(), CV_8SC1);
    crf_evaluator.Evaluate(current_image, segmentation_result);

    //Show the result directly in a cv window. 
    //cv::Mat result = data->GetColorCoding()->LabelToBgr(segmentation_result);
    //cv::namedWindow( "Display window", cv::WINDOW_AUTOSIZE );// Create a window for display.
    //cv::imshow( "Display window", result );                   // Show our image inside it.
    //cv::waitKey(0);

    //Write out the result. 
    int index=0;
    cv::Mat result_bgr = data->GetColorCoding()->LabelToBgr(segmentation_result);
    for(int y=0; y < cloud.height; ++y){
      uchar* pix = result_bgr.ptr<uchar>(y);
      for(int x=0; x < cloud.width; ++x){
       cloud.points[index].b = *(pix++);
       cloud.points[index].g = *(pix++);
       cloud.points[index].r = *(pix++);
       index++;
      }
    }

    sensor_msgs::PointCloud2 sensor_msg_out;
    pcl::toROSMsg(cloud, sensor_msg_out);
    sensor_msg_out.header = sensor_msg->header;
    pub.publish(sensor_msg_out);
  }
};


int main(int argc, char** argv){
  ros::init(argc, argv, "semantic_segmentation");
  if(argc !=6){
    std::cout << "Wrong number of arguments. Correct usage is: " << argv[0] << " <cloud_topic> <transform_topic1> <transform_topic2> <semantic_config_file> <forest_file>" << std::endl;
   return 1; 
  }

  //Setup subcribers
  PointReader test(argv[1], argv[2], argv[3], argv[4], argv[5]);

  //Spin
  ros::spin(); 
}
