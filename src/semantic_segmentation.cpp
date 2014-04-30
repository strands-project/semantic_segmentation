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
  std::string transform_topic; 
  Utils::DynamicDataset* data;
  Utils::Configuration config;
  Rdfs::Forest * forest;
  Inference::DenseCRFEvaluator crf_evaluator;
  ros::Publisher pub;

public:
  PointReader(std::string cloud_topic, std::string transform_topic, std::string config_file, std::string forest_file) : tf(), cloud_topic(cloud_topic), transform_topic(transform_topic){
    cloud_sub.subscribe(nh, cloud_topic, 10);
    tf_filter = new tf::MessageFilter<sensor_msgs::PointCloud2>(cloud_sub, tf, transform_topic, 10);
    tf_filter->registerCallback( boost::bind(&PointReader::PointCloudCallback, this, _1) );

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
    //TODO Pay attention here! This should be tested with the actual data on the robot! 
    tf.lookupTransform("/base_link", transform_topic, ros::Time(0), t); 
    tf::Quaternion q = t.getRotation();
    Eigen::Quaterniond q_eigen;
    tf::quaternionTFToEigen(q, q_eigen);	
    // std::cout <<  q_eigen.matrix() << std::endl; //Just for testing.
    
    //Do the actual computations here. 
    Utils::DataImage current_image; 
    current_image = data->GenerateImage(cloud, q_eigen.matrix().cast<float>());
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
  if(argc !=5){
    std::cout << "Wrong number of arguments. Correct usage is: " << argv[0] << " <cloud_topic> <transform_topic> <semantic_config_file> <forest_file>" << std::endl; 
  }

  //Setup subcribers
  PointReader test(argv[1], argv[2], argv[3], argv[4]);

  //Spin
  ros::spin(); 
}
