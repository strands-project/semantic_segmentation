#include "cv_util.hh"

#include <stdexcept>
#include <string>
#include <fstream>
#include <ios>
#include <iostream>

#include <opencv2/highgui/highgui.hpp>

namespace Utils{

void SaveMat(const std::string& filename, const cv::Mat& data){
  if (data.empty()){
    throw std::runtime_error(std::string("No data was provided for saving to file: ") +filename);
  }

  std::ofstream out(filename.c_str(), std::ios::out|std::ios::binary);
  if (!out){
     throw std::runtime_error(std::string("Could not create file: ") +filename);
  }

  int cols = data.cols;
  int rows = data.rows;
  int chan = data.channels();
  int eSiz = (data.dataend-data.datastart)/(cols*rows*chan);
  int type = data.type();

  // Write header
  out.write((char*)&cols,sizeof(cols));
  out.write((char*)&rows,sizeof(rows));
  out.write((char*)&chan,sizeof(chan));
  out.write((char*)&eSiz,sizeof(eSiz));
  out.write((char*)&type,sizeof(type));

  // Write data.
  if (data.isContinuous()){
     out.write((char *)data.data,cols*rows*chan*eSiz);
  }
  else{
     throw std::runtime_error(std::string("Cannot write non-continuous data to file: ") +filename);
  }
  out.close();
}

void ReadMat(const std::string& filename, cv::Mat& data){
  std::ifstream in(filename.c_str(), std::ios::in|std::ios::binary);
  if (!in){
    throw std::runtime_error(std::string("Could not open file: ") +filename);
  }
  int cols;
  int rows;
  int chan;
  int eSiz;
  int type;

  // Read header
  in.read((char*)&cols,sizeof(cols));
  in.read((char*)&rows,sizeof(rows));
  in.read((char*)&chan,sizeof(chan));
  in.read((char*)&eSiz,sizeof(eSiz));
  in.read((char*)&type,sizeof(type));

  // Alocate Matrix.
  data = cv::Mat(rows,cols,type);

  // Read data.
  if (data.isContinuous()){
     in.read((char *)data.data,cols*rows*chan*eSiz);
  }else{
     throw std::runtime_error(std::string("Could not create a continuous cv::Mat to read from file: ") +filename);
  }
  in.close();
}

void ShowCvMat(const cv::Mat& m){
  cv::namedWindow( "OpenCV mat", cv::WINDOW_AUTOSIZE );
  cv::imshow( "OpenCV mat", m );
  cv::waitKey(0);
}
}