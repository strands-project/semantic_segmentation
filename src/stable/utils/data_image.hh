/**
 * @file    data_image.hh
 * @author  Alexander Hermans <alexander.hermans0@gmail.com>
 * @date    Fri Aug 23 2013
 *
 * @brief   The data image class.
 *
 * @par     An object of this class contains all the information needed about a data image.
 *          It still is rather tailored to a kinect image, but it should be general enough
 *          to work with other things when edited a little.
 *
 * (c) Copyright RWTH Aachen University 2013
 */

#ifndef DATA_IMAGE_HH
#define DATA_IMAGE_HH

// STL includes
#include <vector>
#include <string>

// Eigen include
#include <Eigen/Core>

// Local include
#include "configuration.hh"
#include "color_coding.hh"

//OpenCV includes
#include <opencv2/core/core.hpp>

//TEMP
#include <ctime>

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>

namespace Utils {

/// These are two typedefs copied from a libfreenect header. We only need these to load the dump files.
typedef enum {
    TILT_STATUS_STOPPED = 0x00, /**< Tilt motor is stopped */
    TILT_STATUS_LIMIT   = 0x01, /**< Tilt motor has reached movement limit */
    TILT_STATUS_MOVING  = 0x04, /**< Tilt motor is currently moving to new position */
} freenect_tilt_status_code;
typedef struct {
    int16_t                   accelerometer_x; /**< Raw accelerometer data for X-axis, see FREENECT_COUNTS_PER_G for conversion */
    int16_t                   accelerometer_y; /**< Raw accelerometer data for Y-axis, see FREENECT_COUNTS_PER_G for conversion */
    int16_t                   accelerometer_z; /**< Raw accelerometer data for Z-axis, see FREENECT_COUNTS_PER_G for conversion */
    int8_t                    tilt_angle;      /**< Raw tilt motor angle encoder information */
    freenect_tilt_status_code tilt_status;     /**< State of the tilt motor (stopped, moving, etc...) */
} freenect_raw_tilt_state;


/**
 * @class DataImage
 * @brief Image class containing all necessary data about an image from a dataset.
 */
class DataImage
{

public:
    /**
     * @brief  Basic constructor
     * @return An basic image without any data. This needs to be added with further method calls.
     */
    DataImage();

    /**
     * @brief Adds an RGB image to the data image and currently sets the width and height for the image.
     * @param rgb_image_filename Filename used to load the RGB image.
     */
    void AddRGBImage(const std::string &rgb_image_filename);

    /**
     * @brief Adds an RGB image to the data image loaded eslwhere.
     * @param rgb_mat The actual rgb image.
     */
    void AddRGBImage(const cv::Mat& rgb_mat);
    
    /**
     * @brief Computes the LAB image given a previously loaded RGB image.
     * @param store_both If this this true both RGB and LAB will be stored,
     *                   otherwise the RGB image will be discarded.
     */
    void ComputeLABImage(bool store_both);

    /**
     * @brief Adds the depth image to the current data image.
     * @param depth_image_filename Filename used to load the depth image. Currently only .pgm files are supported.
     * @param is_raw Boolean specifying if the data was preprocessed or not. Depending on this flag the data is
     *               handled in a different way.
     */
    void AddDepthImage(const std::string &depth_image_filename, bool is_raw = false);

    /**
     * @brief Adds the depth image to the current data image.
     * @param depth_mat cv::Mat holding depth values for each pixel (or NaN values if not avaliable).
     */
    void AddDepthImage(cv::Mat& depth_mat);

    /**
     * @brief Adds the accelerometer data to the current data image.
     * @param accel_filename Filename used to load the accelerometer data. This can be a .txt with 4 values
     *                       or a raw .dump file recorded with libfreenect.
     */
    void AddAccelerometerData(const std::string& accel_filename);

     /**
     * @brief Adds the accelerometer data to the current data image.
     * @param rotation Rotation matrix of the camera the image is recorded with.
     */
    void AddAccelerometerData(const Eigen::Matrix3f& rotation);

    /**
     * @brief Loads and undistorts a pair of raw rgb and depth files. They will be undistorted and the depth
     *        image will be reprojected onto the rgb camera plane, thus only requiring the rgb camera paramters
     *        afterwards. Take care as the depth image contains many NaN values and the RGB image has a white border
     *        caused due to the undistorting.
     * @param rgb_image_filename Filename used to load the RGB image.
     * @param depth_filename Filename used to load the depth image. Currently only .pgm files are supported.
     * @param accel_filename Filename used to load the accelerometer data. This can be a .txt with 4 values
     *                       or a raw .dump file recorded with libfreenect.
     */
    void LoadAndUndistortDepthPlusRGB(const std::string &rgb_filename, const std::string &depth_filename, const std::string &accelerometer_filename);

    /**
     * @brief Adds a ground truth annotation image to the current data image.
     * @param filename Filename used to load the annotation image.
     * @param dataset_type Parameter specifying the type of the dataset. This is needed to determine the class
     *                     colorscheme.
     */
    void AddAnnotation(const std::string &filename, const Utils::ColorCoding *cc);

    /**
     * @brief Computes three integral images for the L, a and b channel of the LAB image.
     *        These are stored for later use during the feature evluation.
     */
    void ComputeColorIntegralImage();

    void ComputeLabGradientImages();

    void ComputeDepthGradientImages();

    void Calculate3DFeature();

    void Compute3DCovariance();

    void ComputeNormals();

    ~DataImage();

    inline int Width() const {
        return m_width;
    }
    inline int Height() const {
        return m_height;
    }

    const cv::Mat& GetLABImage() const;

    const cv::Mat& GetRGBImage() const;

    std::string GetRGBFilename() const;

    const cv::Mat &GetDepthImage() const ;

    const cv::Mat& GetAnnotation() const;
    std::string const& GetAnnotationFilename() const;
    void  SetAnnotationFilename(const std::string &filename);

    Eigen::Matrix3f GetAccelerometerData() const;

    const cv::Mat & Get3DFeature() const;
    const cv::Mat &Get3DCovariance() const;
    const cv::Mat &GetGradient3dData() const;

    const cv::Mat &GetGradientData() const;
    const cv::Mat &GetColorIntegralData() const;
    const cv::Mat &GetNormals() const;

    void AddUnaryPotential(const std::string unary_filename, const float weight = 1.0f);
    void AddUnaryPotential(const cv::Mat &unary_potential, const float weight = 1.0f);

    const cv::Mat &GetUnary() const;
    std::string GetUnaryFilename() const;
    void SetUnaryFilename(const std::string & unary_filename);

    std::string GetResultFilename() const;
    void SetResultFilename(const std::string & result_filename);

    std::string GetPointCloudFilename() const;
    void SetPointCloudFilename(const std::string & pcd_filename);

    void AddPointCloud(pcl::PointCloud< pcl::PointXYZ >::Ptr cloud);
    pcl::PointCloud<pcl::PointXYZ>::Ptr GetPointCloud() const;

    pcl::PointCloud<pcl::PointXYZ>::Ptr ComputePointCloudFromDepth(); 

    const cv::Mat &GetValidityMask() const;

    float GetNormalizationFactor(int x_pos) const;
    const float *GetNormalizationData() const;

    float GetColorPatch(int x, int y, int color_channel, int patch_width, int patch_height) const;
    float GetGradientPatch(int x, int y, int gradient_band, int patch_radius =7) const;
    float GetGradientPatch(int x, int y, int gradient_band, int patch_width, int patch_height) const;
    float GetGradient3dPatch(int x, int y, int gradient_band, int patch_width, int patch_height) const;
    //void SaveGradientImages(std::string filename, bool from_depth) const;

    static void SetRGBIntrinsic(std::string rgb_camera_calibration_filename);
    static void SetRGBAndDepthIntrinsic(std::string rgb_camera_calibration_filename, std::string depth_camera_calibration_filename);

    //Call this before doing anything with the class. It loads the calibration files and sets certain parameters which specify how the class will act during runtime.
    static void Initialize(Utils::Configuration configuration);

    static void GetSequenceRgbParameters(double &fx, double &fy, double &cx, double &cy, int &width, int &height);

    cv::Mat ProcessRawDepthImage(cv::Mat raw_depth);

private:
    Eigen::Matrix3f LoadAccelerometerRotation(std::string filename);

    void ComputeDepthNormalization();
    void SetImageSize(int cols, int rows);


private:
    int m_width;                                /// @brief The width of this image.
    int m_height;                               /// @brief The height of this image.
    bool m_is_initialized;                           /// @brief True if any kind of that has been loaded to set the width and height.

    cv::Mat m_rgb_image;                        /// @brief The actual RGB data.
    bool m_has_rgb;                             /// @brief True if RGB data is stored for this image.
    std::string m_rgb_filename;                 /// @brief Stores the rgb image filename.

    cv::Mat m_lab_image;                        /// @brief The actual LAB data.
    bool m_has_lab;                             /// @brief True if LAB data is stored for this image.
    cv::Mat m_lab_integral_image;               /// @brief The LAB data integral images.
    bool m_has_lab_integral;                    /// @brief True if LAB integral images are stored for this image.

    cv::Mat m_depth_image;                 /// @brief The actual depth information.
    bool m_has_depth;                           /// @brief True if depth is stored for this image.

    Eigen::Matrix3f m_accelerometer_rotation;     /// @brief Accelerometer for this image.
    bool m_has_accelerometer_data;                /// @brief True if Accelerometer data was loaded.
    std::vector<float> m_normalization_depth_factor;  /// @brief Stores a column-wise normalization factor, which is the maximum depth in that image column.

    cv::Mat m_annotation_image;                 /// @brief Annotation image that might be needed for this image during training or evaluation.
    bool m_has_annotation;                      /// @brief True if the annotation is available.
    std::string m_annotation_filename;          /// @brief Name of the annotation image.


    //Images needed to store the integral gradient histogram bins.
    cv::Mat m_gradient_histogram;
    bool m_has_gradient_histogram;

    cv::Mat m_gradient_3d_histogram;
    bool m_has_3d_gradient_histogram;

    //Image which stores 3D feature response
    cv::Mat m_3d_feature;
    bool m_has_3d_features;

    //Image which stores the covariance of the 3D locations.
    cv::Mat m_3d_covariance;
    bool m_has_3d_covariance;

    //Image which stores the normals.
    cv::Mat m_normals;
    bool m_has_normals;


    //For test images this is used to store the unary potential which is given to the Dense Crf.
    cv::Mat m_unary_potential;
    bool m_has_unary; // is set to false as long as it is empty. True otherwise


    //This is set to true if the pixel is a valid pixel. False if it is not.
    //Main use is to mask the borders caused by undistortion and depth holes.
    cv::Mat m_validity_mask;


    std::string m_unary_filename;
    std::string m_result_filename;

    std::string m_pcd_filename;
    pcl::PointCloud<pcl::PointXYZ>::Ptr m_cloud;
    bool m_has_point_cloud; 

public:
    // Some static members which are set for all the images. These are public for easy acces.
    static bool m_is_preprocessed;
    static bool  m_is_loaded_from_sequence;
    static float m_sequence_width;
    static float m_sequence_height;

    static bool m_use_rectification_instead_accelerometer;      /// @brief If this is true an alternative rectification algorithm is used instead of the accelerometer data.
// static std::vector<std::string> m_segmentation_directories; /// @brief used to store directories which contain segmentations of the images.


    // Color calibration
    static float m_fx_rgb;
    static float m_fy_rgb;
    static float m_fx_rgb_inv;
    static float m_fy_rgb_inv;
    static float m_cx_rgb;
    static float m_cy_rgb;

    // Depth calibration
    static float m_fx_depth;
    static float m_fy_depth;
    static float m_fx_depth_inv;
    static float m_fy_depth_inv;
    static float m_cx_depth;
    static float m_cy_depth;

    static float m_undistort_k1;
    static float m_undistort_k2;
    static float m_undistort_k3;
    static float m_undistort_p1;
    static float m_undistort_p2;

    // General kinect calibration
    static float m_baseline_focal;
    static float m_kinect_disparity_offset;
    static Eigen::Matrix3f m_kinect_basic_covariance;
    static Eigen::Matrix4f m_depth_to_rgb_transform;

    // Gradient parameters
    static bool m_alternative_gradient_distribution_type;
    static int m_num_normal_neighbors;
    static int m_num_gradient_bins;


    //TEMP!
    float m_clock_cycles;
    float m_image_count;


};
}
#endif // DATA_IMAGE_HH
