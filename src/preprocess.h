#include <pcl_conversions/pcl_conversions.h>
#include <sensor_msgs/msg/point_cloud2.hpp>

#ifdef HAVE_LIVOX_ROS_DRIVER2
#include <livox_ros_driver2/msg/custom_msg.hpp>
#endif

#ifdef HAVE_LIVOX_ROS_DRIVER2
using LivoxCustomMsg = livox_ros_driver2::msg::CustomMsg;
#endif

using namespace std;

typedef pcl::PointXYZINormal PointType;
typedef pcl::PointCloud<PointType> PointCloudXYZI;

enum LID_TYPE
{
    AVIA = 1,
    VELO16,
    L515,
    OUST64,
    XT32,
    KITTI
}; //{1, 2, 3}

namespace velodyne_ros
{
    struct EIGEN_ALIGN16 Point
    {
        PCL_ADD_POINT4D;
        float intensity;
        float time;
        uint16_t ring;

        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    };
} // namespace velodyne_ros
POINT_CLOUD_REGISTER_POINT_STRUCT(velodyne_ros::Point,
                                  (float, x, x)(float, y, y)(float, z, z)(
                                      float, intensity,
                                      intensity)(float, time, time)(uint16_t,
                                                                    ring, ring))

namespace ouster_ros
{
    struct EIGEN_ALIGN16 Point
    {
        PCL_ADD_POINT4D;
        float intensity;
        uint32_t t;
        uint16_t reflectivity;
        uint8_t ring;
        uint16_t ambient;
        uint32_t range;
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    };
} // namespace ouster_ros

// clang-format off
POINT_CLOUD_REGISTER_POINT_STRUCT(ouster_ros::Point,
    (float, x, x)
    (float, y, y)
    (float, z, z)
    (float, intensity, intensity)
    // use std::uint32_t to avoid conflicting with pcl::uint32_t
    (std::uint32_t, t, t)
    (std::uint16_t, reflectivity, reflectivity)
    (std::uint8_t, ring, ring)
    (std::uint16_t, ambient, ambient)
    (std::uint32_t, range, range)
)

/*** Hesai_XT32 ***/
namespace xt32_ros
{
struct EIGEN_ALIGN16 Point
{
    PCL_ADD_POINT4D;
    float    intensity;
    double   timestamp;
    uint16_t ring;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};
} // namespace xt32_ros
POINT_CLOUD_REGISTER_POINT_STRUCT( xt32_ros::Point,
                                   ( float, x, x )( float, y, y )( float, z, z )( float, intensity, intensity )( double, timestamp,
                                                                                                                 timestamp )( uint16_t, ring, ring ) )
/*****************/
                                                                        
class Preprocess {
public:
    Preprocess();

    ~Preprocess();

#ifdef HAVE_LIVOX_ROS_DRIVER2
    void process(const LivoxCustomMsg::ConstSharedPtr &msg, PointCloudXYZI::Ptr &pcl_surf_out);
#endif

    void process(const sensor_msgs::msg::PointCloud2::ConstSharedPtr &msg, PointCloudXYZI::Ptr &pcl_out);

    PointCloudXYZI pl_full, pl_corn, pl_surf;
    PointCloudXYZI pl_buff[128]; //maximum 128 line lidar
    int lidar_type, point_filter_num, N_SCANS;;
    double blind;
    bool kitti_vertical_filter = false;
private:
#ifdef HAVE_LIVOX_ROS_DRIVER2
    void avia_handler(const LivoxCustomMsg::ConstSharedPtr &msg);
#endif

    void l515_handler(const sensor_msgs::msg::PointCloud2::ConstSharedPtr &msg);

    void velodyne_handler(const sensor_msgs::msg::PointCloud2::ConstSharedPtr &msg);

    void kitti_handler(const sensor_msgs::msg::PointCloud2::ConstSharedPtr &msg);

    void oust64_handler(const sensor_msgs::msg::PointCloud2::ConstSharedPtr &msg);

    void xt32_handler(const sensor_msgs::msg::PointCloud2::ConstSharedPtr &msg);
};
