#include "nyudepthv1_color_coding.h"

#include <stdexcept>
#include <iostream>

using namespace Utils;

NyudepthV1ColorCoding::NyudepthV1ColorCoding(){
}

signed char NyudepthV1ColorCoding::BgrToLabel(const unsigned char* bgr) const{
  if (bgr[0] == 0 && bgr[1] == 0 && bgr[2] == 128) {            // Bed
    return 0;
  } else if (bgr[0] == 0 && bgr[1] == 128 && bgr[2] == 0) {     // Blind
    return 1;
  } else if (bgr[0] == 0 && bgr[1] == 128 && bgr[2] == 128) {   // Bookshelf
    return 2;
  } else if (bgr[0] == 128 && bgr[1] == 0 && bgr[2] == 0) {     // Cabinet
    return 3;
  } else if (bgr[0] == 128 && bgr[1] == 128 && bgr[2] == 0) {   // Ceiling
    return 4;
  } else if (bgr[0] == 128 && bgr[1] == 128 && bgr[2] == 128) { // Floor
    return 5;
  } else if (bgr[0] == 0 && bgr[1] == 0 && bgr[2] == 192) {     // Picture
    return 6;
  } else if (bgr[0] == 0 && bgr[1] == 128 && bgr[2] == 64) {    // Sofa
    return 7;
  } else if (bgr[0] == 0 && bgr[1] == 128 && bgr[2] == 192) {   // Table
    return 8;
  } else if (bgr[0] == 128 && bgr[1] == 0 && bgr[2] == 64) {    // Television
    return 9;
  } else if (bgr[0] == 128 && bgr[1] == 0 && bgr[2] == 192) {   // Wall
    return 10;
  } else if (bgr[0] == 128 && bgr[1] == 128 && bgr[2] == 64) {  // Window
    return 11;
  } else if (bgr[0] == 128 && bgr[1] == 128 && bgr[2] == 192) {  // Background
    return 12;
  } else if (bgr[0] == 0 && bgr[1] == 0 && bgr[2] == 0) {     // Void
    return -1;
  } else if (bgr[0] == 10 && bgr[1] == 10 && bgr[2] == 10) {     // No depth
    return -2;
  } else if (bgr[0] == 200 && bgr[1] == 255 && bgr[2] == 200) {   // Depth but no label.
    return -3;
  } else {
    std::cout << "[r,g,b] = [" << (int)bgr[2] << ", "<< (int)bgr[1] << ", "<< (int)bgr[0] << "] "<< std::endl;
    throw std::runtime_error("Unknown color found in the annotation image. Cannot convert to label!");
  }
}

cv::Vec3b NyudepthV1ColorCoding::LabelToBgr(signed char label) const{
  if (label == 0) {         // Bed
    return cv::Vec3b(0,0,128);
  } else if (label == 1) {  // Blind
    return cv::Vec3b(0,128,0);
  } else if (label == 2) {  // Bookshelf
    return cv::Vec3b(0,128,128);
  } else if (label == 3) {  // Cabinet
    return cv::Vec3b(128,0,0);
  } else if (label == 4) {  // Ceiling
    return cv::Vec3b(128,128,0);
  } else if (label == 5) {  // Floor
    return cv::Vec3b(128,128,128);
  } else if (label == 6) {  // Picture
    return cv::Vec3b(0,0,192);
  } else if (label == 7) {  // Sofa
    return cv::Vec3b(0,128,64);
  } else if (label == 8) {  // Table
    return cv::Vec3b(0,128,192);
  } else if (label == 9) { // Television
    return cv::Vec3b(128,0,64);
  } else if (label == 10) { // Wall
    return cv::Vec3b(128,0,192);
  } else if (label == 11) { // Window
    return cv::Vec3b(128,128,64);
  } else if (label == 12) { // Background
    return cv::Vec3b(128,128, 192);
  } else if (label == -1) { // Void
    return cv::Vec3b(0,0,0);
  } else if (label == -2) { // No depth
    return cv::Vec3b(10,10,10);
  } else if (label == -3) { // Depth, but no label
    return cv::Vec3b(200,255,200);
  } else {
    throw std::runtime_error("Unknown label found in the image. Cannot convert to color!");
  }
}


