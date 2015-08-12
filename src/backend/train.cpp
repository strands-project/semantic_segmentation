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
#include <pcl/common/time.h>
#include <pcl/console/parse.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/png_io.h>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/segmentation/supervoxel_clustering.h>

// STL includes
#include <map>
#include <memory>



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

  Utils::RgbLabelConversion label_converter(conf.getJsonValueAsString(conf.get<std::string>("color_coding_key")));

  Utils::DataLoader dl(conf);

  libf::DataStorage::ptr d = dl.loadAllTrainingData("train_images", conf.get<bool>("vccs_rectification"));
  std::map<int,int> dist;
  for(int a = 0; a < d->getSize(); a++){
    dist[d->getClassLabel(a)]++;
  }
  for(auto b : dist){
    std::cout <<b.first << " "<< b.second << std::endl;
  }

  std::cout << d->getSize() << std::endl;

  libf::RandomForestLearner<libf::DecisionTreeLearner> forestLearner;
  forestLearner.getTreeLearner().setNumFeatures(ceil(sqrt(d->getDimensionality())));
  forestLearner.getTreeLearner().setMinSplitExamples(conf.get<int>("min_split_samples"));
  forestLearner.getTreeLearner().setNumBootstrapExamples(d->getSize());
  forestLearner.getTreeLearner().setUseBootstrap(true);
  //forestLearner.getTreeLearner().setUseClassFrequencies(true);
  forestLearner.getTreeLearner().setMaxDepth(conf.get<int>("max_depth"));

  forestLearner.setNumTrees(conf.get<int>("number_trees"));
  forestLearner.setNumThreads(8);

  auto state = forestLearner.createState();
  libf::ConsoleGUI<decltype(state)> gui(state);

  auto forest = forestLearner.learn(d, state);

  std::filebuf fb;
  if (fb.open (conf.getPath("model_path"),std::ios::out)){
    std::ostream os(&fb);
    forest->write(os);
  }
  fb.close();

  gui.join();

  return 0;
}
