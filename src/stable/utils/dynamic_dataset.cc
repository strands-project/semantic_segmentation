// Local includes
#include "dynamic_dataset.hh"
#include "datasets/nyudepth_v1.h"
#include "datasets/strands.h"


using namespace Utils;

DynamicDataset::DynamicDataset(Configuration conf){
}


DynamicDataset *DynamicDataset::LoadDataset(const std::string& config_filename){
  Configuration conf(config_filename);
  return LoadDataset(conf);
}

DynamicDataset *DynamicDataset::LoadDataset(Configuration conf){
  int type = conf.read<int>("dataset_type");
  DynamicDataset * temp;
  switch (type) {
    case Utils::NYUDEPTH_V1:
    temp = new NYUDepthV1(conf);
    break;
    case Utils::STRANDS:
    temp = new StrandsDataset(conf);
    break;
  default:
    throw std::runtime_error("Error: Not a supported dataset type!");
  }
  temp->Load();
  return temp;
}