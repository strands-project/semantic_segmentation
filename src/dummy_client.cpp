//Ros includes
#include "ros/ros.h"

//STL includes
#include <string>
#include <vector>

//Service includes
#include "semantic_segmentation/LabelIntegratedPointCloud.h"
#include "semantic_map_publisher/ObservationService.h"


int main(int argc, char **argv){
  ros::init(argc, argv, "semantic_segmentation_cloud_fetcher");
  std::string waypoint = "WayPoint1";
  if(argc > 1){
    waypoint = std::string(argv[1]);
  }
  ros::NodeHandle nh("~");
  ros::ServiceClient client_get_cloud = nh.serviceClient<semantic_map_publisher::ObservationService>("/semantic_map_publisher/SemanticMapPublisher/ObservationService");
  ros::ServiceClient client_push_cloud = nh.serviceClient<semantic_segmentation::LabelIntegratedPointCloud>("/semantic_segmentation_node/label_integrated_cloud");

  semantic_map_publisher::ObservationService srv;
  srv.request.waypoint_id = waypoint;
  srv.request.resolution = 0.01; //is this 1 cm?
  if (client_get_cloud.call(srv)){
    ROS_INFO("Received cloud!");
    sensor_msgs::PointCloud2 integrated_cloud = srv.response.cloud;
    semantic_segmentation::LabelIntegratedPointCloud srv_sem_seg;
    srv_sem_seg.request.integrated_cloud = integrated_cloud;
    if (client_push_cloud.call(srv_sem_seg)){
      ROS_INFO("Sent cloud!");

      //TODO debug semantic output here!

    }else{
      ROS_ERROR("Failed to call service to send the waypoint pointcloud");
      return 1;
    }
  }else{
    ROS_ERROR("Failed to call service to get the waypoint pointcloud");
    return 1;
  }

  return 0;
}