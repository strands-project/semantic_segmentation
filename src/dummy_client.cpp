//Ros includes
#include "ros/ros.h"

//STL includes
#include <string>
#include <vector>

//Service includes
#include "semantic_segmentation/LabelIntegratedPointCloud.h"


int main(int argc, char **argv){
  ros::init(argc, argv, "semantic_segmentation_dummy_client");
  std::string waypoint = "WayPoint1";
  if(argc > 1){
    waypoint = std::string(argv[1]);
  }
  ros::NodeHandle nh("~");
  ros::ServiceClient client = nh.serviceClient<semantic_segmentation::LabelIntegratedPointCloud>("/semantic_segmentation_node/label_integrated_cloud");

  semantic_segmentation::LabelIntegratedPointCloud srv;
  srv.request.waypoint_id = waypoint;
  if (client.call(srv)){
    ROS_INFO("Requested labeled cloud for %s", waypoint.c_str());
    uint C = srv.response.index_to_label_name.size();
    for(uint c = 0; c < C; ++c){
      ROS_INFO("%s : %f", srv.response.index_to_label_name[c].c_str(), srv.response.label_frequencies[c]);
    }
  }else{
    ROS_ERROR("Failed to call service for the labeled Waypoint");
    return 1;
  }

  return 0;
}