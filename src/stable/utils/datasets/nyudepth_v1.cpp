// C includes
#include <dirent.h>

// STL includes
#include <stdexcept>

// Boost includes
#include <boost/progress.hpp>

// Local includes
#include "../configuration.hh"
#include "nyudepth_v1.h"
#include "../color_coding/nyudepthv1_color_coding.h"
#include "../string_util.h"
#include "../cv_util.hh"

#include <pcl/visualization/pcl_visualizer.h>

using namespace Utils;

NYUDepthV1::NYUDepthV1(Configuration conf) : StaticDataset::StaticDataset(conf), DynamicDataset::DynamicDataset(conf) {
  m_color_coding = new NyudepthV1ColorCoding();
  m_configuration = conf;

}

NYUDepthV1::~NYUDepthV1() {
  delete m_color_coding;
}


void NYUDepthV1::LoadSynchronizedFramelist() {
  //Check if the data is preprocessed, in this case the numbers have been changed!
  if(m_is_preprocessed) {
    // We are expecting 3 kinds of files in the folder for each frame.
    //The name is the same namely the index.
    std::vector<std::string> depth_files;
    m_image_filenames = ListInDir(m_raw_data_folder, "png"); //Only list the pgm files.
    //Essentially this will give is the number of images. Now just create new lists.

    if(depth_files.size() == 0) {
      std::cerr << "Did not load any frames! Loading from: " << m_raw_data_folder << std::endl;
    }

    m_color_filenames.resize(depth_files.size());
    m_depth_filenames.resize(depth_files.size());
    m_accel_filenames.resize(depth_files.size());

    for(unsigned int i = 1; i <= m_depth_filenames.size(); ++i) {
      std::stringstream ss;
      ss << m_raw_data_folder << "/" << i;
      m_color_filenames[i - 1] = ss.str()  + ".png";
      m_depth_filenames[i - 1] = ss.str() + ".pgm";
      m_accel_filenames[i - 1] = ss.str()  + ".txt";
    }

  } else {
    //List files
    std::vector<std::string> files = ListInDir(m_raw_data_folder);

    //Split up
    std::vector<std::string> rgb_files;
    std::vector<std::string> depth_files;
    std::vector<std::string> accel_files;

    for(unsigned int i = 0; i < files.size(); ++i) {
      if(files[i].at(0) == 'r') {
        rgb_files.push_back(files[i]);
      }

      if(files[i].at(0) == 'd') {
        depth_files.push_back(files[i]);
      }

      if(files[i].at(0) == 'a') {
        accel_files.push_back(files[i]);
      }
    }

    //Synchronize the data so that each rgb image is connected to the shortest distance depth image.

    //Sort lists
    std::sort(rgb_files.begin(), rgb_files.end(), Utils::t_sort);
    std::sort(depth_files.begin(), depth_files.end(), Utils::t_sort);
    std::sort(accel_files.begin(), accel_files.end(), Utils::t_sort);

    //Iterative over depth files
    double current_color_time = 0;
    double delta_time, delta_time_new = 0;
    double delta_time_of_previous_pair = -1;
    double delta_accel_time, delta_accel_time_new = 0;
    int j = 0;
    int j_accel = 0;

    for(unsigned int i = 0; i < rgb_files.size(); ++i) {
      //Get the current time
      current_color_time = strtod(rgb_files[i].substr(rgb_files[i].find_first_of("-") + 1, rgb_files[i].find_last_of("-") - 2).data(), NULL);

      //Compare it to the current time of the depth file at the same location in the list.
      j = i;

      if(j >= static_cast<int>(depth_files.size())) {
        j = static_cast<int>(depth_files.size() - 1);
      }

      delta_time = current_color_time - strtod(depth_files[j].substr(depth_files[j].find_first_of("-") + 1, depth_files[j].find_last_of("-") - 2).data(), NULL);


      //If the depth image comes after the RGB image, we search backwards in the list.
      if(delta_time < 0) {
        delta_time = 9999999;
        delta_time_new = 0;

        while(j >= 0) {
          delta_time_new = fabs(current_color_time - strtod(depth_files[j].substr(depth_files[j].find_first_of("-") + 1, depth_files[j].find_last_of("-") - 2).data(), NULL));

          if(delta_time_new < delta_time) { //We found a better match
            delta_time = delta_time_new; //Set it
            j--; //Search on.

            if(j < 0) {
              j = 0;
              break;
            }
          } else { //This is worse than the previous match, no better match will be found
            j++;
            break;
          }
        }
      } else {   //Otherwise we search fowards in the list.
        delta_time = 9999999;
        delta_time_new = 0;

        while(j < depth_files.size()) {
          delta_time_new = fabs(current_color_time - strtod(depth_files[j].substr(depth_files[j].find_first_of("-") + 1, depth_files[j].find_last_of("-") - 2).data(), NULL));

          if(delta_time_new < delta_time) { //We found a better match
            delta_time = delta_time_new; //Set it
            j++; //Search on.

            if(j >= depth_files.size()) {
              j = depth_files.size() - 1;
              break;
            }
          } else { //This is worse than the previous match, no better match will be found
            j--;
            break;
          }
        }
      }



      //Compare it to the current time of the accel file at the location x2 in the list.
      j_accel = i * 2;

      if(j_accel >= accel_files.size()) {
        j_accel = accel_files.size() - 1;
      }

      delta_accel_time = current_color_time - strtod(accel_files[j_accel].substr(accel_files[j_accel].find_first_of("-") + 1, accel_files[j_accel].find_last_of("-") - 2).data(), NULL);


      //If the depth image comes after the RGB image, we search backwards in the list.
      if(delta_accel_time < 0) {
        delta_accel_time = 9999999;
        delta_accel_time_new = 0;

        while(j_accel >= 0) {
          delta_accel_time_new = fabs(current_color_time - strtod(accel_files[j_accel].substr(accel_files[j_accel].find_first_of("-") + 1, accel_files[j_accel].find_last_of("-") - 2).data(), NULL));

          if(delta_accel_time_new < delta_accel_time) { //We found a better match
            delta_accel_time = delta_accel_time_new; //Set it
            j_accel--; //Search on.

            if(j_accel < 0) {
              j_accel = 0;
              break;
            }
          } else { //This is worse than the previous match, no better match will be found
            j_accel++;
            break;
          }
        }
      } else {   //Otherwise we search fowards in the list.
        delta_accel_time = 9999999;
        delta_accel_time_new = 0;

        while(j_accel < accel_files.size()) {
          delta_accel_time_new = fabs(current_color_time - strtod(accel_files[j_accel].substr(accel_files[j_accel].find_first_of("-") + 1, accel_files[j_accel].find_last_of("-") - 2).data(), NULL));

          if(delta_accel_time_new < delta_accel_time) { //We found a better match
            delta_accel_time = delta_accel_time_new; //Set it
            j_accel++; //Search on.

            if(j >= accel_files.size()) {
              j_accel = accel_files.size() - 1;
              break;
            }
          } else { //This is worse than the previous match, no better match will be found
            j_accel--;
            break;
          }
        }
      }


      //Best match for this frame is at rgb index j, with a time difference of delta_time.
      //Here we can check if the frame is too unsynchronized and possibly drop it.
      if(delta_time < 0.03f) {
        if(j >= depth_files.size() || j_accel >= accel_files.size()) {
          break; //we are done, as all depth files have been used.
        }

        if(delta_time_of_previous_pair != -1 && m_depth_filenames.back() == std::string(m_raw_data_folder + "/" + depth_files[j])) {
          if(delta_time < delta_time_of_previous_pair) {
            m_color_filenames.back() = m_raw_data_folder + "/" + rgb_files[i];
            m_accel_filenames.back() = m_raw_data_folder + "/" + accel_files[j_accel];
          } else {
            //We can just skip this whole frame, as the previous one matches the color image better!
          }

          //   std::cout << "double frame! " <<  rgb_files[i] << " " << depth_files[j]<< " " << accel_files[j_accel]<<std::endl;
        } else {
          m_depth_filenames.push_back(m_raw_data_folder + "/" + depth_files[j]);
          m_color_filenames.push_back(m_raw_data_folder + "/" + rgb_files[i]);
          m_accel_filenames.push_back(m_raw_data_folder + "/" + accel_files[j_accel]);
          //   std::cout << rgb_files[i] << " " << depth_files[j]<< " " << accel_files[j_accel]<<std::endl;
        }

        delta_time_of_previous_pair = delta_time;

      } else {
        //std::cout << " Dropped a frame as not RGB + Depth + Accelerometer match could be found!!" << std::endl;
      }

    }
  }
}

void NYUDepthV1::Load(DataType data_type, int load_flags) {
  //Check if we are reading images from the simple dataset, or if we are reading images
  //from a sequence.
  m_load_requirement_flags = load_flags;
  m_reading_from_sequence = (data_type == StaticDataset::SEQUENCE);

  m_configuration.add<bool>("loaded_from_sequence", m_reading_from_sequence);

  // TODO init the dataimage class!
  Utils::DataImage::Initialize(m_configuration);

  if(m_reading_from_sequence == false) {
    //Load all filenames from a file specified in the configuration.
    m_image_filenames = List(data_type);
    m_image_count = m_image_filenames.size();
    // Load all needed folder and extensions.
    m_images_extension = m_configuration.read<std::string>("image_extension");
    m_images_folder = m_configuration.read<std::string>("image_directory");

    m_annotations_extension = m_configuration.read<std::string>("annotation_extension");
    m_annotations_folder = m_configuration.read<std::string>("annotation_directory");

    m_depth_images_extension = m_configuration.read<std::string>("depth_extension");
    m_depth_images_folder = m_configuration.read<std::string>("depth_directory");
    m_accelerometer_extension = m_configuration.read<std::string>("accelerometer_extension", "");
    m_accelerometer_folder = m_configuration.read<std::string>("accelerometer_directory", "");

    m_unary_extension = m_configuration.read<std::string>("unary_extension");
    m_unary_folder = m_configuration.read<std::string>("unary_directory");

    m_unary2_extension = m_configuration.read<std::string>("unary2_extension", "");
    m_unary2_folder = m_configuration.read<std::string>("unary2_directory", "");

    m_result_extension = m_configuration.read<std::string>("result_extension");
    m_result_folder = m_configuration.read<std::string>("result_directory");

    m_pcd_extension = m_configuration.read<std::string>("pcd_extension");
    m_pcd_folder = m_configuration.read<std::string>("pcd_directory");

  } else {
    // Load stuff that might be needed for raw computations.
    std::string external_config_filename = m_configuration.read<std::string>("external_config", "");

    if(external_config_filename == std::string("")) {
      //Just load the stuff specified in this config.
      m_raw_data_folder = m_configuration.read<std::string>("data_folder", "");
    } else {
      //Load the alternative config file and get the data from there.
      Configuration config(external_config_filename);
      m_raw_data_folder = config.read<std::string>("data_folder", "");
    }

    LoadSynchronizedFramelist();
    m_image_count = m_color_filenames.size();
    // TODO Check if the stuff is preprocessed or not from the config .
    m_is_preprocessed = m_configuration.read<bool>("is_data_preprocessed");

  }

  m_color_calib_filename = m_configuration.read<std::string>("color_calibration_filename");
  m_depth_calib_filename = m_configuration.read<std::string>("depth_calibration_filename");

}

void NYUDepthV1::Load(){
  //Nothing to load yet for the dynamic case.
  Utils::DataImage::Initialize(m_configuration);
}


Utils::DataImage NYUDepthV1::GenerateImage(const pcl::PointCloud<pcl::PointXYZRGB>& cloud, const Eigen::Matrix3f& q) {
  Utils::DataImage image;

  //Set the cloud.
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_xyz_ptr(new pcl::PointCloud<pcl::PointXYZ>);
  cloud_xyz_ptr->width = cloud.width;
  cloud_xyz_ptr->height = cloud.height;
  cloud_xyz_ptr->is_dense = cloud.is_dense;
  cloud_xyz_ptr->resize(cloud.width * cloud.height);
  for(size_t i = 0; i < cloud.points.size(); i++) {
    Eigen::Vector3f p(cloud.points[i].x, cloud.points[i].z, cloud.points[i].y);
    Eigen::Vector3f rect = q*p; 
    
    cloud_xyz_ptr->points[i].x = p(0);
    cloud_xyz_ptr->points[i].y = p(2);
    cloud_xyz_ptr->points[i].z = p(1);
  }
  

  image.AddPointCloud(cloud_xyz_ptr);

//   pcl::PointCloud<pcl::PointXYZRGB>::Ptr rect_ptr(new pcl::PointCloud<pcl::PointXYZRGB>);
//   rect_ptr->width = cloud.width;
//   rect_ptr->height = cloud.height;
//   rect_ptr->is_dense = cloud.is_dense;
//   rect_ptr->resize(cloud.width * cloud.height);
//   for(size_t i = 0; i < cloud.points.size(); i++) {
//    Eigen::Vector3f p( cloud.points[i].x, cloud.points[i].y, cloud.points[i].z);
//    Eigen::Vector3f rect_p = q*p;
//     
//     rect_ptr->points[i].x = p(0);
//     rect_ptr->points[i].y = p(1);
//     rect_ptr->points[i].z = p(2);
//     if(rect_ptr->points[i].z  < -10)  rect_ptr->points[i].z=0;
//     rect_ptr->points[i].r = cloud.points[i].r;
//     rect_ptr->points[i].g = cloud.points[i].g;
//     rect_ptr->points[i].b = cloud.points[i].b;
//   }
// 
//     pcl::visualization::PCLVisualizer viewer("Cloud Viewer");
// 
//     viewer.addPointCloud(rect_ptr, "new");
//     while (!viewer.wasStopped ()){
//        viewer.spinOnce (100);
//     }
  
  
  //Extract RGB and set it.
  if(m_load_requirement_flags & Utils::LAB) {
    //First add the rgb image, this sets the width and the height.
    int index = 0;
    cv::Mat bgr = cv::Mat(cloud.height, cloud.width, CV_8UC3);

    for(int y = 0; y < cloud.height; ++y) {
      unsigned char* color_ptr = bgr.ptr<unsigned char>(y);

      for(int x = 0; x < cloud.width; ++x) {
        *color_ptr++ = cloud.points[index].b;
        *color_ptr++ = cloud.points[index].g;
        *color_ptr++ = cloud.points[index].r;
        index++;
      }
    }

    image.AddRGBImage(bgr);
    std::cout << cloud.width << " " << cloud.height << std::endl;

    //Compute the LAB image, but only keep the RGB image if it is needed.
    image.ComputeLABImage(m_load_requirement_flags & Utils::RGB);

    if(m_load_requirement_flags & Utils::LAB_INTEGRAL) {
      //Also compute an integral image!
      image.ComputeColorIntegralImage();
    }

  } else {
    //Just set the RGB image if it is needed.
    if(m_load_requirement_flags & Utils::RGB) {
      int index = 0;
      cv::Mat bgr = cv::Mat(cloud.height, cloud.width, CV_8UC3);

      for(int y = 0; y < cloud.height; ++y) {
        unsigned char* color_ptr = bgr.ptr<unsigned char>(y);

        for(int x = 0; x < cloud.width; ++x) {
          *color_ptr++ = cloud.points[index].b;
          *color_ptr++ = cloud.points[index].g;
          *color_ptr++ = cloud.points[index].r;
          index++;
        }
      }
      image.AddRGBImage(bgr);
    }
  }

  //Extract Depth and set it.
  if(m_load_requirement_flags & Utils::DEPTH) {
    //Load the depth image.
    int index = 0;
    cv::Mat depth = cv::Mat(cloud.height, cloud.width, CV_32FC1, std::numeric_limits<float>::quiet_NaN());

    for(int y = 0; y < cloud.height; ++y) {
      float* depth_ptr = depth.ptr<float>(y);
      for(int x = 0; x < cloud.width; ++x) {
        *depth_ptr++ = cloud.points[index].z;
        index++;
      }
    }

    image.AddDepthImage(depth);
    if(m_load_requirement_flags & Utils::ACCELEROMETER) {
      //Load the accelerometer data.
      //image.AddAccelerometerData(q);
      image.AddAccelerometerData(Eigen::Matrix3f::Identity());
    }

    if(m_load_requirement_flags & Utils::DEPTH_COVARIANCE) {
      throw std::runtime_error("Cannot compute the depth covariance for data loaded from point clouds, the calibration is needed!");
    }

  }

  //Do everything else.
  if(m_load_requirement_flags & Utils::UNARY || m_load_requirement_flags & Utils::UNARY2 ||
     m_load_requirement_flags & Utils::ANNOTATION) {
    throw std::runtime_error("Images loaded from a point cloud do not have unary potentials or annotations!");
  }


  if(m_load_requirement_flags & Utils::GEOMETRIC_FEAT) {
    image.Calculate3DFeature();
  }

  if(m_load_requirement_flags & Utils::GRADIENT_COLOR) {
    image.ComputeLabGradientImages();
  }

  if(m_load_requirement_flags & Utils::GRADIENT_DEPTH) {
    image.ComputeDepthGradientImages();
  }

  if(m_load_requirement_flags & Utils::NORMALS) {
    image.ComputeNormals();
  }
  
  return image;
}


Utils::DataImage NYUDepthV1::GenerateImage(int index) const {
  //Create an empty image.
  Utils::DataImage image;

  if(!m_reading_from_sequence) {
    //Just read from the dataset.
    //Load color data.
    if(m_load_requirement_flags & Utils::LAB) {
      //First add the rgb image, this sets the width and the height.
      std::string rgb_fn = m_images_folder + "/" + m_image_filenames[index];
      image.AddRGBImage(rgb_fn);

      //Compute the LAB image, but only keep the RGB image if it is needed.
      image.ComputeLABImage(m_load_requirement_flags & Utils::RGB);

      if(m_load_requirement_flags & Utils::LAB_INTEGRAL) {
        //Also compute an integral image!
        image.ComputeColorIntegralImage();
      }

    } else {
      //Just set the RGB image if it is needed.
      if(m_load_requirement_flags & Utils::RGB) {
        std::string rgb_fn = m_images_folder + "/" + m_image_filenames[index];
        image.AddRGBImage(rgb_fn);
      }
    }

    //Load depth data.
    if(m_load_requirement_flags & Utils::DEPTH) {
      //Load the depth image.
      std::string depth_fn = m_depth_images_folder + "/" +  m_image_filenames[index];
      SwapExtension(depth_fn, m_images_extension, m_depth_images_extension);
      image.AddDepthImage(depth_fn);

      if(m_load_requirement_flags & Utils::ACCELEROMETER) {
        //Load the accelerometer data.
        std::string accel_fn = m_accelerometer_folder + "/" +  m_image_filenames[index];
        SwapExtension(accel_fn, m_images_extension, m_accelerometer_extension);
        image.AddAccelerometerData(accel_fn);
      }
    }

    //Load unary potentials and set the filename.
    std::string unary1_fn = m_unary_folder + "/" + m_image_filenames[index];
    SwapExtension(unary1_fn, m_images_extension, m_unary_extension);
    image.SetUnaryFilename(unary1_fn);

    if(m_load_requirement_flags & Utils::UNARY) {
      image.AddUnaryPotential(unary1_fn);
    }

    if(m_load_requirement_flags & Utils::UNARY2) {
      std::string unary2_fn = m_unary2_folder + "/" + m_image_filenames[index];
      SwapExtension(unary2_fn, m_images_extension, m_unary2_extension);
      image.AddUnaryPotential(unary2_fn);
    }

    //Load the ground truth annotation.
    std::string annotation_fn = m_annotations_folder + "/" + m_image_filenames[index];
    SwapExtension(annotation_fn, m_images_extension, m_annotations_extension);
    image.SetAnnotationFilename(annotation_fn);

    if(m_load_requirement_flags & Utils::ANNOTATION) {
      image.AddAnnotation(annotation_fn, m_color_coding);
    }

    //Set the result filename.
    std::string result_fn = m_result_folder + "/" + m_image_filenames[index];
    SwapExtension(result_fn, m_images_extension, m_result_extension);
    image.SetResultFilename(result_fn);

    //Set the pcd filename.
    std::string pcd_fn = m_pcd_folder + "/" + m_image_filenames[index];
    SwapExtension(pcd_fn, m_images_extension, m_pcd_extension);
    image.SetPointCloudFilename(pcd_fn);

  } else {
    //We are reading images from a sequence.
    if(m_is_preprocessed) {
      //We can assume the images are preprocessed. Proceed as above.
      if(m_load_requirement_flags & Utils::LAB) {
        //First add the rgb image, this sets the width and the height.
        std::string rgb_fn = m_color_filenames[index];
        image.AddRGBImage(rgb_fn);

        if(m_load_requirement_flags & Utils::LAB_INTEGRAL) {
          //Also compute an integral image!
          image.ComputeColorIntegralImage();
        }

        //Compute the LAB image, but only keep the RGB image if it is needed.
        image.ComputeLABImage(m_load_requirement_flags & Utils::RGB);
      } else {
        //Just set the RGB image if it is needed.
        if(m_load_requirement_flags & Utils::RGB) {
          image.AddRGBImage(m_color_filenames[index]);
        }
      }

      //Load depth data.
      if(m_load_requirement_flags & Utils::DEPTH) {
        //Load the depth image.
        image.AddDepthImage(m_depth_filenames[index]);

        if(m_load_requirement_flags & Utils::ACCELEROMETER) {
          //Load the accelerometer data.
          image.AddAccelerometerData(m_accel_filenames[index]);
        }
      }
    } else {
      //We need to undistort images. This is only done if we load depth, accelerometer and RGB data.
      if((m_load_requirement_flags & Utils::DEPTH) && ((m_load_requirement_flags & Utils::RGB)
          || (m_load_requirement_flags & Utils::LAB))) {
        image.LoadAndUndistortDepthPlusRGB(m_color_filenames[index], m_depth_filenames[index],
                                           m_accel_filenames[index]);

        //        image.AddRGBImage(m_color_filenames[index]);
        //        image.AddDepthImage(m_depth_filenames[index], true);
        //        image.AddAccelerometerData(m_accel_filenames[index]);

        if(m_load_requirement_flags & Utils::LAB) {
          image.ComputeLABImage(m_load_requirement_flags & Utils::RGB);

          if(m_load_requirement_flags & Utils::LAB_INTEGRAL) {
            //Also compute an integral image!
            image.ComputeColorIntegralImage();
          }
        }
      } else {
        if(m_load_requirement_flags == 0) {
          //We are skipping everything, apparently we just need some filenames!
        } else {
          throw std::runtime_error("Error: You tried to load a still distorted image, but did not require depth, color and accelerometer data.");
        }
      }
    }

    if(m_load_requirement_flags & Utils::DEPTH_COVARIANCE) {
      //Compute the covariance for each pixel.
      image.Compute3DCovariance();
    }

    //Set unary and result filenames
    std::string result_fn = m_color_filenames[index];
    SwapExtension(result_fn, "r-", "label-");
    image.SetResultFilename(result_fn);
    std::string unary_fn = m_color_filenames[index];
    SwapExtension(unary_fn, "r-", "unr-");
    image.SetUnaryFilename(unary_fn);
  }

  if(m_load_requirement_flags & Utils::GEOMETRIC_FEAT) {
    image.Calculate3DFeature();
  }

  if(m_load_requirement_flags & Utils::GRADIENT_COLOR) {
    image.ComputeLabGradientImages();
  }

  if(m_load_requirement_flags & Utils::GRADIENT_DEPTH) {
    image.ComputeDepthGradientImages();
  }

  if(m_load_requirement_flags & Utils::NORMALS) {
    image.ComputeNormals();
  }

  return image;
}

void NYUDepthV1::SetLoadFlags(int load_flags){
  m_load_requirement_flags = load_flags;
}

