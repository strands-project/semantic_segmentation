// Local includes
#include "utils/calibration.h"
#include "utils/config.h"
#include "utils/commandline_parser.h"
#include "utils/cv_util.h"
#include "utils/rgb_label_conversion.h"
#include "utils/data_loader.h"
#include "voxel.h"

// Eigen includes
#include <Eigen/Core>

// libforest
#include "libforest/libforest.h"

// OpenCV includes
#include <opencv2/opencv.hpp>

// PCL includes
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/visualization/pcl_visualizer.h>

// STL includes
#include <map>
#include <memory>

#include "densecrf.h"


std::string usage(char * exe){
  std::string u("\n");
  u += std::string(exe) + " --conf <config file> --opt1 <val1> ... \n";
  return u;
}

int main (int argc, char ** argv) {
  if(argc <= 2){
    throw std::runtime_error("No parameters given. Usage: " + usage(argv[0]));
  }

  //Parse all parameters
  std::map<std::string, std::string> parameter_name_value_map;
  bool parse_succeeded = Utils::parseParamters(argc, argv, parameter_name_value_map);
  if(!parse_succeeded){
    throw std::runtime_error("Mangled command line arguments. " + usage(argv[0]));
  }

  //check if we found a config file.
  if(parameter_name_value_map.count("conf") == 0){
    throw std::runtime_error("No config file was given" + usage(argv[0]));
  }

  std::string config_file = parameter_name_value_map["conf"];
  parameter_name_value_map.erase("conf");
  Utils::Config conf(config_file, parameter_name_value_map);


  Utils::DataLoader dl(conf);
  Utils::RgbLabelConversion label_converter = dl.getLabelConverter();

  //Load the rf data.
  libf::RandomForest<libf::DecisionTree>* forest = new libf::RandomForest<libf::DecisionTree>();
  std::filebuf fb;
  if (fb.open (conf.getPath("model_path"),std::ios::in)){
    std::istream is(&fb);
    forest->read(is);
  }
  fb.close();

  std::vector<std::string> image_names = dl.getImageList("test_images");

  for(int i = 0; i < image_names.size(); ++i) {
    //construct the cloud
    cv::Mat color = dl.loadColor(image_names[i]);
    cv::Mat depth = dl.loadDepth(image_names[i]);
    Utils::Calibration calibration = dl.loadCalibration(image_names[i]);

    pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud;
    pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud_unrectified;
    dl.create_cloud(depth, color, calibration, cloud, cloud_unrectified);

    //voxelize
    std::map<int, std::shared_ptr<Voxel> > voxels;
    if(conf.get<bool>("vccs_rectification")){
      dl.extractVoxels(cloud, cloud_unrectified, voxels);
    }else{
      pcl::PointCloud<pcl::PointXYZRGBA>::Ptr voxelized_cloud;
      dl.extractVoxels(cloud, voxels, voxelized_cloud);
    }
    //Testing
    int valid_points = 0;
    for(auto v : voxels){
      valid_points += v.second->getSize();
    }

    DenseCRF crf(valid_points, label_converter.getValidLabelCount());
    Eigen::MatrixXf unary(label_converter.getValidLabelCount(), valid_points);
    Eigen::MatrixXf feature(6, valid_points);
    Eigen::MatrixXf feature2(3, valid_points);

    cv::Mat result_image(color.rows, color.cols, CV_32SC1, cv::Scalar(-1));
    std::vector<float> probs;
    uint index = 0;
    for(auto v : voxels){
      const libf::DataPoint& feat = v.second->getFeatures();
      forest->classLogPosterior(feat, probs);
      std::vector<float>::iterator result = std::max_element(probs.begin(), probs.end());
      v.second->drawValueIntoImage<int>(result_image, std::distance(probs.begin(), result));
      v.second->addDataToCrfMats(unary, feature, feature2, index, probs);
    }
    crf.setUnaryEnergy( unary );
    //crf.addPairwiseEnergy( feature, new PottsCompatibility( 10 ) );
    crf.addPairwiseEnergy( feature2, new PottsCompatibility( 3 ) );

    Matrix<short,Dynamic,1> map = crf.map(10);
    cv::Mat result_image_crf(color.rows, color.cols, CV_32SC1, cv::Scalar(-1));
    int* r_ptr = result_image_crf.ptr<int>(0);
    index = 0;
    for(auto v : voxels){
      for(int j : v.second->getIndices()){
        r_ptr[j] = map(index);
        index++;
      }
    }
//     Utils::ShowCvMat(label_converter.labelToRgb(result_image), "res", false);
//     Utils::ShowCvMat(label_converter.labelToRgb(labels[i]), "gt");
    cv::imwrite(dl.getResultName(image_names[i]), label_converter.labelToRgb(result_image));
    cv::imwrite(dl.getResultName(image_names[i]+ "crf"), label_converter.labelToRgb(result_image_crf));
//     pcl::io::savePCDFileBinary<pcl::PointXYZRGBA>("/data/work/test" + image_names[i] + ".pcd", *clouds[i]);
  }

  return 0;
}
