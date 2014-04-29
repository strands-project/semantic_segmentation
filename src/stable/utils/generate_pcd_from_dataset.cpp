//#define testing //Uncomment this to do some sanity checks.

// Local includes
#include "configuration.hh"
#include "static_dataset.hh"
#include "data_image.hh"

#ifdef testing
#include "dynamic_dataset.hh"
#endif


// PCL includes
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/visualization/pcl_visualizer.h>

// OpenCV includes
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

int main(int argc, char *argv[]) {
    // Load configuration data
    Utils::Configuration configuration(argv[1]);
    int dataset_type = configuration.read<int>("dataset_type");

    // Load the dataset
    std::cout << "Loading the dataset" << std::endl;
#ifndef testing
    Utils::StaticDataset * dataset = Utils::StaticDataset::LoadDataset(configuration, Utils::StaticDataset::ALL, Utils::DEPTH | Utils::RGB | Utils::ACCELEROMETER);
#else
    Utils::StaticDataset * dataset = Utils::StaticDataset::LoadDataset(configuration, Utils::StaticDataset::ALL, Utils::DEPTH | Utils::RGB | Utils::ACCELEROMETER | Utils::GEOMETRIC_FEAT);
    Utils::DynamicDataset * dataset_test = Utils::DynamicDataset::LoadDataset(configuration);
    dataset_test->SetLoadFlags(Utils::DEPTH | Utils::RGB | Utils::ACCELEROMETER | Utils::GEOMETRIC_FEAT);
#endif

    // Get each image, create an organized PCD and store it.
    for(unsigned int i = 0; i < dataset->ImageCount(); ++i) {
        // Load it
        Utils::DataImage im = dataset->GenerateImage(i);
        continue;

        cv::Mat depth = im.GetDepthImage();
        cv::Mat rgb = im.GetRGBImage();
        const float cx_rgb = im.m_cx_rgb;
        const float cy_rgb = im.m_cy_rgb;
        const float fx_rgb_inv = im.m_fx_rgb_inv;
        const float fy_rgb_inv = im.m_fy_rgb_inv;

        // Convert it.
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZRGB>);
        cloud->width    = im.Width();
        cloud->height   = im.Height();
        cloud->is_dense = false;
        cloud->points.resize (cloud->width * cloud->height);        
        // Eigen::Vector4f     sensor_origin_    // Look at this maybe ?
        // Eigen::Quaternionf  sensor_orientation_
        int index = 0;
        for(int y = 0; y < cloud->height; y++) {
            float* depth_ptr = depth.ptr<float>(y);
            unsigned char* rgb_ptr = rgb.ptr<unsigned char>(y);
            for(int x = 0; x < cloud->width; x++) {
                pcl::PointXYZRGB point;
                point.x = (static_cast<float>(x) - cx_rgb) * fx_rgb_inv *  (*depth_ptr);
                point.y = (static_cast<float>(y) - cy_rgb) * fy_rgb_inv *  (*depth_ptr);
                point.z = *depth_ptr;
                uint32_t point_rgb =  (((uint32_t) *(rgb_ptr+2)) << 16 | ((uint32_t) *(rgb_ptr+1)) << 8 | ((uint32_t) *rgb_ptr));
                point.rgb = *reinterpret_cast<float*>(&point_rgb);
                cloud->points[index] = point;
                index++;
                depth_ptr++;
                rgb_ptr+=3;
            }
        }

        // write it.
        pcl::io::savePCDFileBinary(im.GetPointCloudFilename(), *cloud);
        //pcl::io::savePCDFileASCII(im.GetPcdFilename(), cloud);

//         pcl::visualization::PCLVisualizer viewer("Cloud Viewer");
//         viewer.addPointCloud(cloud, std::string("cloud"));
//         viewer.initCameraParameters();
//         viewer.setCameraPosition(-8.4, 1.3, 1.7, -0.6, 0.0, 1.4, 0.0, -1.0, 0.0);
//         while (!viewer.wasStopped ()){
//           viewer.spinOnce();
//         }


        //For testing purposes.
#ifdef testing
        //Load the Image again from the cloud.
        Utils::DataImage im_test = dataset_test->GenerateImage(cloud, im.GetAccelerometerData());
        cv::Mat depth_diff;
        cv::absdiff(im_test.GetDepthImage(), im.GetDepthImage(), depth_diff);
        cv::Scalar d = cv::sum(depth_diff);        // sum elements per channel
        std::cout << "Depth difference: " << d.val[0] << std::endl;
        cv::Mat color_diff;
        cv::absdiff(im_test.GetRGBImage(), im.GetRGBImage(), depth_diff);
        cv::Scalar c = cv::sum(color_diff);        // sum elements per channel
        std::cout << "Color difference: " << c.val[0] + c.val[1] + c.val[2] << std::endl;
        cv::Mat geometric_feature_diff;
        cv::absdiff(im_test.Get3DFeature(), im.Get3DFeature(), depth_diff);
        cv::Scalar g = cv::sum(geometric_feature_diff);        // sum elements per channel
        std::cout << "Geometric Feature difference: " << g.val[0] + g.val[1] + g.val[2] << std::endl;
        std::cout << "Acceleromter difference:" << std::endl;
        std::cout << (im.GetAccelerometerData()) - (im_test.GetAccelerometerData()) << std::endl;
        
        
#endif





    }
    return EXIT_SUCCESS;
}