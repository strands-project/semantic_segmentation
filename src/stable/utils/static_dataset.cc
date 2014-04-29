// STL includes
#include <stdexcept>
#include <fstream>

// C includes
#include <dirent.h>

// Local includes
#include "static_dataset.hh"
#include "configuration.hh"
#include "datasets/nyudepth_v1.h"
#include "datasets/strands.h"

using namespace Utils;

std::vector<std::string> const& StaticDataset::Filenames() const{
  return m_image_filenames;
}

StaticDataset::StaticDataset(Configuration conf){
  m_training_split = conf.read<std::string>("training_split");
  m_validation_split = conf.read<std::string>("validation_split");
  m_test_split = conf.read<std::string>("test_split");
}

int StaticDataset::ImageCount() const{
  return m_image_count;
}

std::vector<std::string> StaticDataset::LoadFromList(const std::string &image_list_filename) {
  std::vector<std::string> image_filenames;
  std::string line;
  std::ifstream ifs(image_list_filename.c_str(), std::ios_base::in);
  while (getline(ifs, line, '\n')) {
    image_filenames.push_back(line);
  }
  return image_filenames;
}

StaticDataset *StaticDataset::LoadDataset(const std::string& config_filename, DataType data_type, int load_flags){

  Configuration conf(config_filename);
  return LoadDataset(conf, data_type, load_flags);
}

StaticDataset *StaticDataset::LoadDataset(Configuration conf, DataType data_type, int load_flags){

  DisplayLoadFlags(load_flags);

  int type = conf.read<int>("dataset_type");
  StaticDataset * temp;
  switch (type) {
    case Utils::NYUDEPTH_V1:
    temp = new NYUDepthV1(conf);
    break;
    case Utils::STRANDS:
    temp = new StrandsDataset(conf);
    break;
    //  case Utils::NYUDEPTH_V2:
    //    break;
    //  case Utils::CITY:
    //    break;
  default:
    throw std::runtime_error("Error: Not a supported dataset type!");
  }
  temp->Load(data_type, load_flags);
  return temp;
}

std::vector<std::string> StaticDataset::List(int type) {
  std::vector<std::string> files;

  files.push_back(m_training_split);
  files.push_back(m_validation_split);
  files.push_back(m_test_split);

  std::vector<std::string> names;
  int types[] = {TRAIN, VALID, TEST};

  std::string line;
  for (int i = 0; i < 3; i++) {
    if (type & types[i]) {
      std::ifstream in(files[i].c_str());
      if (!in.is_open()) {
        std::cerr << "Failed to open file " << files[i] << std::endl;
        exit(EXIT_FAILURE);
      }
      while (getline(in, line, '\n')) {
        names.push_back(line); // If the file is in dos format convert it to unix format
      }
    }
  }
  return names;
}

std::vector<std::string> StaticDataset::ListInDir(std::string dirname, std::string extension) {
  std::vector<std::string> result;

  DIR *dp;
  struct dirent *dirp;
  if ((dp = opendir(dirname.c_str())) == NULL) {
    std::cerr << "Failed to open folder " << dirname << std::endl;
    exit (EXIT_FAILURE);
  }
  while ((dirp = readdir(dp)) != NULL) {
    std::string temp = std::string(dirp->d_name);
    if (extension == std::string("") || temp.substr(temp.find_last_of(".") + 1)== extension) {
      result.push_back(temp);
    }
  }
  closedir(dp);
  return result;
}
