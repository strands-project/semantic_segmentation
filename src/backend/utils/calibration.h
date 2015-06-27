#pragma once

// Eigen includes
#include <Eigen/Core>
#include <Eigen/Geometry>

// STL includes
#include <iostream>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <boost/concept_check.hpp>

// Local includes
#include "json/json.h"

namespace Utils{

class Calibration {
public:

  Calibration(){
  }

  Calibration(std::string filename) : _filename(filename) {
    std::ifstream calib_file(_filename);
    if(!calib_file){
      throw std::runtime_error("Failed to open calibration file " + _filename);
    }

    //Try to parse it.
    Json::Reader reader;
    if(! reader.parse(calib_file, _calib)){
      throw std::runtime_error("Failed to parse calibration file " + _filename + "\n" + reader.getFormatedErrorMessages());
    }
    if(!_calib.isMember("intrinsic") || !_calib.isMember("translation") || !_calib.isMember("rotation")){
      throw std::runtime_error("Calibration file " + _filename + " is not complete!");
    }

    //intrinsics
    for(unsigned int index = 0; index < 9; index++) {
      _intrinsic(index) = _calib["intrinsic"][index].asFloat();
    }
    _intrinsic.transposeInPlace();
    _intrinsic_inverse = _intrinsic.inverse();

    //Rotation
    if(_calib["rotation"]["format"].asString() == "q3"){
      float qx, qy, qz, qw;
      qx = _calib["rotation"]["data"][0].asFloat();
      qy = _calib["rotation"]["data"][1].asFloat();
      qz = _calib["rotation"]["data"][2].asFloat();
      qw = sqrt(1.0 - qx*qx - qy*qy - qz*qz);
      Eigen::Quaternion<float> q;
      q.w() = qw;
      q.z() = qz;
      q.y() = qy;
      q.x() = qx;
      _extrinsic.linear() = q.matrix();
    }else if(_calib["rotation"]["format"].asString() == "q4"){
      Eigen::Quaternion<float> q;
      q.x() = _calib["rotation"]["data"][0].asFloat();
      q.y() = _calib["rotation"]["data"][1].asFloat();
      q.z() = _calib["rotation"]["data"][2].asFloat();
      q.w() = _calib["rotation"]["data"][3].asFloat();
      _extrinsic.linear() = q.matrix();
    }else if(_calib["rotation"]["format"].asString() == "m33"){
      Eigen::Matrix3f r;
      for(unsigned int index = 0; index < 9; index++) {
        r(index) = _calib["rotation"]["data"][index].asFloat();
      }
      _extrinsic.linear() = r;
    }else{
      throw std::runtime_error("Unknown format for _calibration matrix: " + _calib["rotation"]["format"].asString());
    }


    //Translation
    //Quick fix! For now I'll just ignore the offset in "x" and "y" direction (x,y, as in robotics view).
    Eigen::Vector3f t;
    t(0) = 0;//_calib["translation"][0].asFloat();
    t(1) = 0;//_calib["translation"][1].asFloat();
    t(2) = _calib["translation"][2].asFloat();
    _extrinsic.translation() = t;

  }

  void setExtrinsics(Eigen::Matrix3f rotation, Eigen::Vector3f translation){
    _extrinsic.linear() = rotation;
    _extrinsic.translation() = translation;
  }

  void save(std::string filename){

    Json::Value vec(Json::arrayValue);
    for(unsigned int index = 0; index < 9; index++) {
      vec.append(Json::Value(_extrinsic.rotation()(index)));
    }
    _calib["rotation"]["data"] = vec;
    _calib["rotation"]["format"] = "m33";
    _calib["translation"][0] = _extrinsic.translation()[0];
    _calib["translation"][1] = _extrinsic.translation()[1];
    _calib["translation"][2] = _extrinsic.translation()[2];

    std::ofstream file;
    file.open(filename);
    Json::StyledWriter w;
    file << w.write(_calib);
    file.close();

  }

  Json::Value _calib;
  std::string _filename;
  Eigen::Matrix3f _intrinsic;
  Eigen::Matrix3f _intrinsic_inverse;
  Eigen::Isometry3f _extrinsic;
};
}