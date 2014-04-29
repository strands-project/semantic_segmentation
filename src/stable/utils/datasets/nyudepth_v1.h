#ifndef _UTILS_NYUDEPTH_V1_HH_
#define _UTILS_NYUDEPTH_V1_HH_

// STL includes
#include <string>

//Local includes
#include "../data_image.hh"
#include "../configuration.hh"
#include "../static_dataset.hh"
#include "../dynamic_dataset.hh"

// Eigen includes
#include <eigen3/Eigen/Geometry>

namespace Utils {
  class NYUDepthV1 : public StaticDataset, public DynamicDataset{
  public:
    Utils::DataImage GenerateImage(int index) const;
    Utils::DataImage GenerateImage(const pcl::PointCloud<pcl::PointXYZRGB>& cloud, const Eigen::Matrix3f& q);
    NYUDepthV1(Configuration conf);

    ~NYUDepthV1();

    void SetLoadFlags(int load_flags);

  protected:
    void Load(DataType data_type, int load_flags);
    void Load();

  private:
    void LoadSynchronizedFramelist();


  private:
    Utils::Configuration m_configuration;
    //Storage for folders and extensions
    std::string   m_images_extension;
    std::string   m_annotations_extension;
    std::string   m_depth_images_extension;
    std::string   m_unary_extension;
    std::string   m_unary2_extension;
    std::string   m_result_extension;
    std::string   m_accelerometer_extension;
    std::string   m_pcd_extension;


    std::string   m_images_folder;
    std::string   m_annotations_folder;
    std::string   m_depth_images_folder;
    std::string   m_unary_folder;
    std::string   m_unary2_folder;
    std::string   m_result_folder;
    std::string   m_accelerometer_folder;
    std::string   m_pcd_folder;

    //Further things needed in order to read from a continous sequence of images.
    bool          m_reading_from_sequence;
    bool          m_is_preprocessed;
    std::string   m_raw_data_folder;
    std::vector<std::string> m_depth_filenames;
    std::vector<std::string> m_color_filenames;
    std::vector<std::string> m_accel_filenames;

    //Calibration locations
    std::string m_color_calib_filename;
    std::string m_depth_calib_filename;

  };
}

#endif // _UTILS_NYUDEPTH_V1_HH_
