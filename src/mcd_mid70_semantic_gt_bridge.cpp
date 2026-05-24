#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <limits>
#include <memory>
#include <set>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include <pcl/kdtree/kdtree_flann.h>
#include <livox_ros_driver2/msg/custom_msg.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/PCLPointCloud2.h>
#include <pcl/io/pcd_io.h>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/msg/point_field.hpp>
#include <std_msgs/msg/header.hpp>

namespace
{
using LivoxCustomMsg = livox_ros_driver2::msg::CustomMsg;
using PointCloud2Msg = sensor_msgs::msg::PointCloud2;
using PointFieldMsg = sensor_msgs::msg::PointField;

struct LabelPoint
{
    float x = 0.0f;
    float y = 0.0f;
    float z = 0.0f;
    float intensity = 0.0f;
    uint8_t raw_label = 0;
    uint32_t hybrid_label = 0;
};

struct CoordinateKey
{
    int64_t x = 0;
    int64_t y = 0;
    int64_t z = 0;

    bool operator==(const CoordinateKey &other) const
    {
        return x == other.x && y == other.y && z == other.z;
    }
};

struct CoordinateKeyHash
{
    size_t operator()(const CoordinateKey &key) const
    {
        const auto hx = std::hash<int64_t>{}(key.x);
        const auto hy = std::hash<int64_t>{}(key.y);
        const auto hz = std::hash<int64_t>{}(key.z);
        return hx ^ (hy << 1) ^ (hz << 2);
    }
};

CoordinateKey makeCoordinateKey(const double x, const double y, const double z, const double resolution)
{
    const double safe_resolution = std::max(1e-6, resolution);
    return {static_cast<int64_t>(std::llround(x / safe_resolution)),
            static_cast<int64_t>(std::llround(y / safe_resolution)),
            static_cast<int64_t>(std::llround(z / safe_resolution))};
}

struct LabelFrame
{
    std::vector<LabelPoint> labels;
    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud{new pcl::PointCloud<pcl::PointXYZI>()};
    pcl::KdTreeFLANN<pcl::PointXYZI> kdtree;
    std::unordered_map<CoordinateKey, std::vector<size_t>, CoordinateKeyHash> coordinate_index;
    bool kdtree_ready = false;
};

struct OutputPoint
{
    float x = 0.0f;
    float y = 0.0f;
    float z = 0.0f;
    float intensity = 0.0f;
    float time = 0.0f;
    uint16_t ring = 0;
    uint32_t semantic = 0;
};

std::set<int> parseLabelSet(const std::string &text)
{
    std::set<int> labels;
    std::stringstream ss(text);
    std::string token;
    while (std::getline(ss, token, ','))
    {
        token.erase(std::remove_if(token.begin(), token.end(), ::isspace), token.end());
        if (token.empty())
        {
            continue;
        }
        try
        {
            labels.insert(std::stoi(token));
        }
        catch (const std::exception &)
        {
        }
    }
    return labels;
}

template <typename T>
T readValue(const std::vector<uint8_t> &data, const size_t offset)
{
    T value{};
    std::memcpy(&value, data.data() + offset, sizeof(T));
    return value;
}

template <typename T>
void writeValue(std::vector<uint8_t> &data, const size_t offset, const T &value)
{
    std::memcpy(data.data() + offset, &value, sizeof(T));
}

PointFieldMsg makeField(const std::string &name, const uint32_t offset, const uint8_t datatype)
{
    PointFieldMsg field;
    field.name = name;
    field.offset = offset;
    field.datatype = datatype;
    field.count = 1;
    return field;
}

std::string labelPath(const std::string &label_dir, const int frame_index)
{
    std::ostringstream oss;
    oss << label_dir << "/cloud_" << std::setw(4) << std::setfill('0') << frame_index << ".pcd";
    return oss.str();
}
} // namespace

class McdMid70SemanticGtBridge : public rclcpp::Node
{
public:
    McdMid70SemanticGtBridge()
        : Node("mcd_mid70_semantic_gt_bridge")
    {
        source_topic_ = declare_parameter<std::string>("source_topic", "/livox/lidar");
        output_topic_ = declare_parameter<std::string>("output_topic", "/livox/lidar_semantic_gt");
        label_dir_ = declare_parameter<std::string>(
            "label_dir", "data/MCD/semantic_labels/ntu_day_10/inL_labelled");
        frame_id_ = declare_parameter<std::string>("frame_id", "camera_init");
        frame_index_offset_ = declare_parameter<int>("frame_index_offset", 0);
        scan_line_ = declare_parameter<int>("scan_line", 1);
        point_filter_num_ = std::max(1, static_cast<int>(declare_parameter<int>("point_filter_num", 1)));
        pcd_label_start_index_ = std::max(0, static_cast<int>(declare_parameter<int>("pcd_label_start_index", 1)));
        match_xy_threshold_ = declare_parameter<double>("match_xy_threshold", 0.20);
        nearest_match_xyz_threshold_ = declare_parameter<double>("nearest_match_xyz_threshold", 0.50);
        coordinate_match_tolerance_ = declare_parameter<double>("coordinate_match_tolerance", 0.05);
        coordinate_hash_resolution_ = declare_parameter<double>("coordinate_hash_resolution", 0.01);
        intensity_tolerance_ = declare_parameter<double>("intensity_tolerance", 8.0);
        strict_order_labels_ = declare_parameter<bool>("strict_order_labels", true);
        coordinate_label_matching_ = declare_parameter<bool>("coordinate_label_matching", false);
        nearest_label_matching_ = declare_parameter<bool>("nearest_label_matching", true);
        publish_label_cloud_as_input_ = declare_parameter<bool>("publish_label_cloud_as_input", false);
        label_cloud_time_span_ = declare_parameter<double>("label_cloud_time_span", 0.10);
        skip_unlabeled_frames_ = declare_parameter<bool>("skip_unlabeled_frames", true);
        input_points_in_body_frame_ = declare_parameter<bool>("input_points_in_body_frame", true);
        output_points_in_body_frame_ = declare_parameter<bool>("output_points_in_body_frame", true);
        unlabeled_semantic_ = static_cast<uint32_t>(
            std::max(0, static_cast<int>(declare_parameter<int>("unlabeled_semantic", 0))));
        log_interval_ = std::max(1, static_cast<int>(declare_parameter<int>("log_interval", 100)));

        // The MCD semantic PCDs under inL_labelled are not byte-identical to the
        // raw CustomMsg points. They are in the annotated Livox label frame. Query
        // labels by transforming raw Mid70 bag points into that frame, while the
        // output cloud keeps the original raw point coordinates and offset_time.
        const std::vector<double> default_body_to_mid70_R{
            0.9998581316050806, -0.0002258196228773494, -0.016842377168758943,
            0.00017111407692153792, 0.999994705851753, -0.0032494597148797887,
            0.016843021794484745, 0.003246116751423288, 0.9998528768488223};
        const std::vector<double> default_body_to_mid70_T{
            -0.0305152992417423, 0.0089897848417584, -0.0003564686383346};
        const auto body_to_mid70_R = declare_parameter<std::vector<double>>(
            "body_to_mid70_R", default_body_to_mid70_R);
        const auto body_to_mid70_T = declare_parameter<std::vector<double>>(
            "body_to_mid70_T", default_body_to_mid70_T);
        if (body_to_mid70_R.size() == 9)
        {
            std::copy(body_to_mid70_R.begin(), body_to_mid70_R.end(), body_to_mid70_R_.begin());
        }
        else
        {
            RCLCPP_WARN(get_logger(), "body_to_mid70_R must have 9 values; using built-in raw-Mid70 to MCD label-frame default.");
            std::copy(default_body_to_mid70_R.begin(), default_body_to_mid70_R.end(), body_to_mid70_R_.begin());
        }
        if (body_to_mid70_T.size() == 3)
        {
            std::copy(body_to_mid70_T.begin(), body_to_mid70_T.end(), body_to_mid70_T_.begin());
        }
        else
        {
            RCLCPP_WARN(get_logger(), "body_to_mid70_T must have 3 values; using built-in raw-Mid70 to MCD label-frame default.");
            std::copy(default_body_to_mid70_T.begin(), default_body_to_mid70_T.end(), body_to_mid70_T_.begin());
        }

        const std::vector<double> default_output_body_R{
            0.9998581316050806, -0.0002258196228773494, -0.016842377168758943,
            -0.00017111407692153792, -0.999994705851753, 0.0032494597148797887,
            -0.016843021794484745, -0.003246116751423288, -0.9998528768488223};
        const std::vector<double> default_output_body_T{
            -0.010514839241742317, -0.008989784841758377, 0.03735646863833463};
        const auto output_body_R = declare_parameter<std::vector<double>>(
            "output_body_R", default_output_body_R);
        const auto output_body_T = declare_parameter<std::vector<double>>(
            "output_body_T", default_output_body_T);
        if (output_body_R.size() == 9)
        {
            std::copy(output_body_R.begin(), output_body_R.end(), output_body_R_.begin());
        }
        else
        {
            RCLCPP_WARN(get_logger(), "output_body_R must have 9 values; using built-in raw-Mid70 to VN100/body default.");
            std::copy(default_output_body_R.begin(), default_output_body_R.end(), output_body_R_.begin());
        }
        if (output_body_T.size() == 3)
        {
            std::copy(output_body_T.begin(), output_body_T.end(), output_body_T_.begin());
        }
        else
        {
            RCLCPP_WARN(get_logger(), "output_body_T must have 3 values; using built-in raw-Mid70 to VN100/body default.");
            std::copy(default_output_body_T.begin(), default_output_body_T.end(), output_body_T_.begin());
        }

        const std::string plane_label_text = declare_parameter<std::string>(
            "plane_labels", "6,10,13,16,18");
        const std::string object_label_text = declare_parameter<std::string>(
            "object_labels", "1,14,26,27,28");
        const std::string unknown_label_text = declare_parameter<std::string>("unknown_labels", "11");
        plane_labels_ = parseLabelSet(plane_label_text);
        object_labels_ = parseLabelSet(object_label_text);
        unknown_labels_ = parseLabelSet(unknown_label_text);

        fields_.push_back(makeField("x", 0, PointFieldMsg::FLOAT32));
        fields_.push_back(makeField("y", 4, PointFieldMsg::FLOAT32));
        fields_.push_back(makeField("z", 8, PointFieldMsg::FLOAT32));
        fields_.push_back(makeField("intensity", 16, PointFieldMsg::FLOAT32));
        fields_.push_back(makeField("time", 20, PointFieldMsg::FLOAT32));
        fields_.push_back(makeField("ring", 24, PointFieldMsg::UINT16));
        fields_.push_back(makeField("semantic", 28, PointFieldMsg::UINT32));

        publisher_ = create_publisher<PointCloud2Msg>(output_topic_, rclcpp::SensorDataQoS().keep_last(512));
        subscriber_ = create_subscription<LivoxCustomMsg>(
            source_topic_, rclcpp::SensorDataQoS().keep_last(512),
            std::bind(&McdMid70SemanticGtBridge::lidarCallback, this, std::placeholders::_1));

        RCLCPP_INFO(get_logger(),
                    "MCD Mid70 semantic GT bridge: %s -> %s, labels=%s, plane={%s}, object={%s}",
                    source_topic_.c_str(), output_topic_.c_str(), label_dir_.c_str(),
                    plane_label_text.c_str(), object_label_text.c_str());
        RCLCPP_INFO(get_logger(),
                    "MCD GT matching: coordinate=%s, coord_tol=%.3f, coord_res=%.3f, nearest=%s, "
                    "skip_unlabeled=%s, apply_query_transform=%s, output_body=%s, nearest_xyz=%.3f, "
                    "strict_order=%s, pcd_label_start_index=%d, publish_label_cloud=%s.",
                    coordinate_label_matching_ ? "true" : "false",
                    coordinate_match_tolerance_,
                    coordinate_hash_resolution_,
                    nearest_label_matching_ ? "true" : "false",
                    skip_unlabeled_frames_ ? "true" : "false",
                    input_points_in_body_frame_ ? "true" : "false",
                    output_points_in_body_frame_ ? "true" : "false",
                    nearest_match_xyz_threshold_,
                    strict_order_labels_ ? "true" : "false",
                    pcd_label_start_index_,
                    publish_label_cloud_as_input_ ? "true" : "false");
    }

private:
    uint32_t toHybridLabel(const uint8_t raw_label) const
    {
        const int label = static_cast<int>(raw_label);
        if (unknown_labels_.find(label) != unknown_labels_.end())
        {
            return 0;
        }
        if (plane_labels_.find(label) != plane_labels_.end())
        {
            return 1;
        }
        if (object_labels_.find(label) != object_labels_.end())
        {
            return 3;
        }
        return 2;
    }

    const LabelFrame &loadLabels(const int frame_index)
    {
        const auto cached = label_cache_.find(frame_index);
        if (cached != label_cache_.end())
        {
            return *cached->second;
        }

        auto frame = std::make_shared<LabelFrame>();
        pcl::PCLPointCloud2 cloud;
        const std::string path = labelPath(label_dir_, frame_index);
        if (pcl::io::loadPCDFile(path, cloud) != 0)
        {
            auto inserted = label_cache_.emplace(frame_index, frame);
            return *inserted.first->second;
        }

        int x_offset = -1;
        int y_offset = -1;
        int z_offset = -1;
        int intensity_offset = -1;
        int label_offset = -1;
        for (const auto &field : cloud.fields)
        {
            if (field.name == "x")
            {
                x_offset = static_cast<int>(field.offset);
            }
            else if (field.name == "y")
            {
                y_offset = static_cast<int>(field.offset);
            }
            else if (field.name == "z")
            {
                z_offset = static_cast<int>(field.offset);
            }
            else if (field.name == "intensity")
            {
                intensity_offset = static_cast<int>(field.offset);
            }
            else if (field.name == "label")
            {
                label_offset = static_cast<int>(field.offset);
            }
        }

        if (x_offset < 0 || y_offset < 0 || z_offset < 0 || intensity_offset < 0 || label_offset < 0)
        {
            RCLCPP_WARN(get_logger(), "Label PCD missing required fields: %s", path.c_str());
            auto inserted = label_cache_.emplace(frame_index, frame);
            return *inserted.first->second;
        }

        const size_t point_num = static_cast<size_t>(cloud.width) * static_cast<size_t>(cloud.height);
        frame->labels.reserve(point_num);
        frame->cloud->reserve(point_num);
        for (size_t i = 0; i < point_num; ++i)
        {
            const size_t base = i * cloud.point_step;
            LabelPoint point;
            point.x = readValue<float>(cloud.data, base + static_cast<size_t>(x_offset));
            point.y = readValue<float>(cloud.data, base + static_cast<size_t>(y_offset));
            point.z = readValue<float>(cloud.data, base + static_cast<size_t>(z_offset));
            point.intensity = readValue<float>(cloud.data, base + static_cast<size_t>(intensity_offset));
            point.raw_label = readValue<uint8_t>(cloud.data, base + static_cast<size_t>(label_offset));
            point.hybrid_label = toHybridLabel(point.raw_label);
            frame->labels.push_back(point);
            const size_t label_index = frame->labels.size() - 1;
            frame->coordinate_index[
                makeCoordinateKey(point.x, point.y, point.z, coordinate_hash_resolution_)]
                .push_back(label_index);

            pcl::PointXYZI pcl_point;
            pcl_point.x = point.x;
            pcl_point.y = point.y;
            pcl_point.z = point.z;
            pcl_point.intensity = point.intensity;
            frame->cloud->push_back(pcl_point);
        }
        if (!frame->cloud->empty())
        {
            frame->cloud->width = static_cast<uint32_t>(frame->cloud->size());
            frame->cloud->height = 1;
            frame->cloud->is_dense = true;
            frame->kdtree.setInputCloud(frame->cloud);
            frame->kdtree_ready = true;
        }

        auto inserted = label_cache_.emplace(frame_index, frame);
        return *inserted.first->second;
    }

    bool matchesLabel(const livox_ros_driver2::msg::CustomPoint &point, const LabelPoint &label) const
    {
        // Labelled MCD PCDs are in the Mid70 scan order. The body-frame bag used by
        // the SLAM run is already transformed to VN100/body, where y is flipped
        // relative to the labelled PCD. Matching in x/y is enough and avoids
        // coupling this bridge to the SLAM extrinsic parameters.
        const double dx = static_cast<double>(point.x) - static_cast<double>(label.x);
        const double dy = -static_cast<double>(point.y) - static_cast<double>(label.y);
        const double dxy = std::sqrt(dx * dx + dy * dy);
        const double dintensity =
            std::fabs(static_cast<double>(point.reflectivity) - static_cast<double>(label.intensity));
        return dxy <= match_xy_threshold_ && dintensity <= intensity_tolerance_;
    }

    pcl::PointXYZI labelQueryPoint(const livox_ros_driver2::msg::CustomPoint &point) const
    {
        pcl::PointXYZI query;
        if (input_points_in_body_frame_)
        {
            query.x = static_cast<float>(
                body_to_mid70_R_[0] * point.x + body_to_mid70_R_[1] * point.y +
                body_to_mid70_R_[2] * point.z + body_to_mid70_T_[0]);
            query.y = static_cast<float>(
                body_to_mid70_R_[3] * point.x + body_to_mid70_R_[4] * point.y +
                body_to_mid70_R_[5] * point.z + body_to_mid70_T_[1]);
            query.z = static_cast<float>(
                body_to_mid70_R_[6] * point.x + body_to_mid70_R_[7] * point.y +
                body_to_mid70_R_[8] * point.z + body_to_mid70_T_[2]);
        }
        else
        {
            query.x = point.x;
            query.y = point.y;
            query.z = point.z;
        }
        query.intensity = static_cast<float>(point.reflectivity);
        return query;
    }

    std::array<float, 3> labelPointToBody(const LabelPoint &point) const
    {
        const double lx = static_cast<double>(point.x) - body_to_mid70_T_[0];
        const double ly = static_cast<double>(point.y) - body_to_mid70_T_[1];
        const double lz = static_cast<double>(point.z) - body_to_mid70_T_[2];
        return {
            static_cast<float>(body_to_mid70_R_[0] * lx + body_to_mid70_R_[3] * ly +
                               body_to_mid70_R_[6] * lz),
            static_cast<float>(body_to_mid70_R_[1] * lx + body_to_mid70_R_[4] * ly +
                               body_to_mid70_R_[7] * lz),
            static_cast<float>(body_to_mid70_R_[2] * lx + body_to_mid70_R_[5] * ly +
                               body_to_mid70_R_[8] * lz)};
    }

    std::array<float, 3> outputPoint(const livox_ros_driver2::msg::CustomPoint &point) const
    {
        if (!output_points_in_body_frame_)
        {
            return {point.x, point.y, point.z};
        }
        return {
            static_cast<float>(output_body_R_[0] * point.x + output_body_R_[1] * point.y +
                               output_body_R_[2] * point.z + output_body_T_[0]),
            static_cast<float>(output_body_R_[3] * point.x + output_body_R_[4] * point.y +
                               output_body_R_[5] * point.z + output_body_T_[1]),
            static_cast<float>(output_body_R_[6] * point.x + output_body_R_[7] * point.y +
                               output_body_R_[8] * point.z + output_body_T_[2])};
    }

    bool assignNearestLabel(const livox_ros_driver2::msg::CustomPoint &point,
                            const LabelFrame &frame,
                            uint32_t &semantic) const
    {
        if (!frame.kdtree_ready)
        {
            return false;
        }

        const pcl::PointXYZI query = labelQueryPoint(point);
        std::vector<int> indices(1);
        std::vector<float> distances2(1);
        if (frame.kdtree.nearestKSearch(query, 1, indices, distances2) <= 0)
        {
            return false;
        }
        const int label_index = indices.front();
        if (label_index < 0 || static_cast<size_t>(label_index) >= frame.labels.size())
        {
            return false;
        }
        if (distances2.front() >
            static_cast<float>(nearest_match_xyz_threshold_ * nearest_match_xyz_threshold_))
        {
            return false;
        }
        const auto &label = frame.labels[static_cast<size_t>(label_index)];
        const double dintensity =
            std::fabs(static_cast<double>(point.reflectivity) - static_cast<double>(label.intensity));
        if (dintensity > intensity_tolerance_)
        {
            return false;
        }
        semantic = label.hybrid_label;
        return true;
    }

    bool assignCoordinateLabel(const livox_ros_driver2::msg::CustomPoint &point,
                               const LabelFrame &frame,
                               uint32_t &semantic) const
    {
        if (frame.coordinate_index.empty())
        {
            return false;
        }

        const pcl::PointXYZI query = labelQueryPoint(point);
        const double resolution = std::max(1e-6, coordinate_hash_resolution_);
        const double tolerance = std::max(0.0, coordinate_match_tolerance_);
        const int neighbor = std::max(0, static_cast<int>(std::ceil(tolerance / resolution)));
        const CoordinateKey center = makeCoordinateKey(query.x, query.y, query.z, resolution);
        const double max_distance2 = tolerance * tolerance;
        double best_distance2 = std::numeric_limits<double>::max();
        size_t best_index = std::numeric_limits<size_t>::max();

        for (int dx = -neighbor; dx <= neighbor; ++dx)
        {
            for (int dy = -neighbor; dy <= neighbor; ++dy)
            {
                for (int dz = -neighbor; dz <= neighbor; ++dz)
                {
                    const CoordinateKey key{center.x + dx, center.y + dy, center.z + dz};
                    const auto iter = frame.coordinate_index.find(key);
                    if (iter == frame.coordinate_index.end())
                    {
                        continue;
                    }
                    for (const size_t label_index : iter->second)
                    {
                        if (label_index >= frame.labels.size())
                        {
                            continue;
                        }
                        const auto &label = frame.labels[label_index];
                        const double ix = static_cast<double>(query.x) - static_cast<double>(label.x);
                        const double iy = static_cast<double>(query.y) - static_cast<double>(label.y);
                        const double iz = static_cast<double>(query.z) - static_cast<double>(label.z);
                        const double distance2 = ix * ix + iy * iy + iz * iz;
                        if (distance2 > max_distance2)
                        {
                            continue;
                        }
                        const double dintensity =
                            std::fabs(static_cast<double>(point.reflectivity) - static_cast<double>(label.intensity));
                        if (intensity_tolerance_ >= 0.0 && dintensity > intensity_tolerance_)
                        {
                            continue;
                        }
                        if (distance2 < best_distance2)
                        {
                            best_distance2 = distance2;
                            best_index = label_index;
                        }
                    }
                }
            }
        }

        if (best_index == std::numeric_limits<size_t>::max())
        {
            return false;
        }
        semantic = frame.labels[best_index].hybrid_label;
        return true;
    }

    void publishLabelCloudAsInput(const LivoxCustomMsg::SharedPtr &msg,
                                  const LabelFrame &label_frame_data,
                                  const int label_frame)
    {
        const auto &labels = label_frame_data.labels;
        std::vector<OutputPoint> points;
        points.reserve(labels.size());
        std::array<size_t, 4> semantic_counts = {0, 0, 0, 0};

        const float denom = labels.size() > 1 ? static_cast<float>(labels.size() - 1) : 1.0f;
        const float time_span = static_cast<float>(std::max(0.0, label_cloud_time_span_));
        for (size_t i = 0; i < labels.size(); ++i)
        {
            const auto &label = labels[i];
            OutputPoint out;
            const auto body_point = labelPointToBody(label);
            out.x = body_point[0];
            out.y = body_point[1];
            out.z = body_point[2];
            out.intensity = label.intensity;
            out.time = (static_cast<float>(i) / denom) * time_span;
            out.ring = 0;
            out.semantic = label.hybrid_label;
            if (out.semantic <= 3)
            {
                ++semantic_counts[out.semantic];
            }
            points.push_back(out);
        }

        publishOutputCloud(msg->header, points);

        if (frame_index_ < 15 || frame_index_ % log_interval_ == 0)
        {
            RCLCPP_INFO(get_logger(),
                        "MCD semantic GT frame %d: publish label PCD as input, labels=%zu output=%zu "
                        "unknown=%zu plane=%zu gaussian=%zu object=%zu",
                        label_frame, labels.size(), points.size(),
                        semantic_counts[0], semantic_counts[1], semantic_counts[2], semantic_counts[3]);
        }
    }

    void publishOutputCloud(const std_msgs::msg::Header &header, const std::vector<OutputPoint> &points)
    {
        PointCloud2Msg cloud;
        cloud.header = header;
        if (!frame_id_.empty())
        {
            cloud.header.frame_id = frame_id_;
        }
        cloud.height = 1;
        cloud.width = static_cast<uint32_t>(points.size());
        cloud.fields = fields_;
        cloud.is_bigendian = false;
        cloud.point_step = 32;
        cloud.row_step = cloud.point_step * cloud.width;
        cloud.is_dense = true;
        cloud.data.assign(static_cast<size_t>(cloud.row_step), 0);

        for (size_t i = 0; i < points.size(); ++i)
        {
            const size_t base = i * cloud.point_step;
            writeValue<float>(cloud.data, base + 0, points[i].x);
            writeValue<float>(cloud.data, base + 4, points[i].y);
            writeValue<float>(cloud.data, base + 8, points[i].z);
            writeValue<float>(cloud.data, base + 16, points[i].intensity);
            writeValue<float>(cloud.data, base + 20, points[i].time);
            writeValue<uint16_t>(cloud.data, base + 24, points[i].ring);
            writeValue<uint32_t>(cloud.data, base + 28, points[i].semantic);
        }

        publisher_->publish(std::move(cloud));
    }

    void lidarCallback(const LivoxCustomMsg::SharedPtr msg)
    {
        const int label_frame = frame_index_ + frame_index_offset_;
        const auto &label_frame_data = loadLabels(label_frame);
        const auto &labels = label_frame_data.labels;
        if (labels.empty() && skip_unlabeled_frames_)
        {
            if (frame_index_ < 15 || frame_index_ % log_interval_ == 0)
            {
                RCLCPP_INFO(get_logger(),
                            "MCD semantic GT frame %d: no label PCD, skip publishing this frame.",
                            label_frame);
            }
            ++frame_index_;
            return;
        }
        if (publish_label_cloud_as_input_)
        {
            publishLabelCloudAsInput(msg, label_frame_data, label_frame);
            ++frame_index_;
            return;
        }
        size_t label_idx = static_cast<size_t>(std::min<int>(pcd_label_start_index_, labels.size()));

        std::vector<OutputPoint> points;
        points.reserve(msg->points.size());
        std::array<size_t, 4> semantic_counts = {0, 0, 0, 0};
        size_t matched = 0;
        size_t tag16_points = 0;

        for (size_t i = 1; i < msg->points.size(); ++i)
        {
            const auto &point = msg->points[i];
            if (static_cast<int>(point.line) >= scan_line_)
            {
                continue;
            }
            const uint8_t tag = point.tag & 0x30;
            if (tag != 0x10 && tag != 0x00)
            {
                continue;
            }
            if (i % static_cast<size_t>(point_filter_num_) != 0)
            {
                continue;
            }
            if (!std::isfinite(point.x) || !std::isfinite(point.y) || !std::isfinite(point.z))
            {
                continue;
            }
            const auto &prev = msg->points[i - 1];
            if (std::fabs(point.x - prev.x) <= 1e-7f &&
                std::fabs(point.y - prev.y) <= 1e-7f &&
                std::fabs(point.z - prev.z) <= 1e-7f)
            {
                continue;
            }

            OutputPoint out;
            const auto xyz = outputPoint(point);
            out.x = xyz[0];
            out.y = xyz[1];
            out.z = xyz[2];
            out.intensity = static_cast<float>(point.reflectivity);
            out.time = static_cast<float>(point.offset_time) * 1e-9f;
            out.ring = static_cast<uint16_t>(point.line);
            out.semantic = unlabeled_semantic_;

            if (tag == 0x10 && !labels.empty())
            {
                ++tag16_points;
                bool label_matched = false;
                if (coordinate_label_matching_)
                {
                    label_matched = assignCoordinateLabel(point, label_frame_data, out.semantic);
                }
                else if (nearest_label_matching_)
                {
                    label_matched = assignNearestLabel(point, label_frame_data, out.semantic);
                }
                else if (strict_order_labels_)
                {
                    if (label_idx < labels.size())
                    {
                        out.semantic = labels[label_idx].hybrid_label;
                        ++label_idx;
                        label_matched = true;
                    }
                }
                else
                {
                    if (label_idx < labels.size() && matchesLabel(point, labels[label_idx]))
                    {
                        out.semantic = labels[label_idx].hybrid_label;
                        ++label_idx;
                        label_matched = true;
                    }
                    else
                    {
                        const size_t search_end = std::min(labels.size(), label_idx + static_cast<size_t>(8));
                        for (size_t candidate = label_idx + 1; candidate < search_end; ++candidate)
                        {
                            if (matchesLabel(point, labels[candidate]))
                            {
                                out.semantic = labels[candidate].hybrid_label;
                                label_idx = candidate + 1;
                                label_matched = true;
                                break;
                            }
                        }
                    }
                }
                if (label_matched)
                {
                    ++matched;
                }
            }

            if (out.semantic <= 3)
            {
                ++semantic_counts[out.semantic];
            }
            points.push_back(out);
        }

        publishOutputCloud(msg->header, points);

        if (frame_index_ < 15 || frame_index_ % log_interval_ == 0)
        {
            RCLCPP_INFO(get_logger(),
                        "MCD semantic GT frame %d: labels=%zu tag16=%zu matched=%zu output=%zu "
                        "unknown=%zu plane=%zu gaussian=%zu object=%zu",
                        label_frame, labels.size(), tag16_points, matched, points.size(),
                        semantic_counts[0], semantic_counts[1], semantic_counts[2], semantic_counts[3]);
        }
        ++frame_index_;
    }

    std::string source_topic_;
    std::string output_topic_;
    std::string label_dir_;
    std::string frame_id_;
    int frame_index_offset_ = 0;
    int frame_index_ = 0;
    int scan_line_ = 1;
    int point_filter_num_ = 1;
    int pcd_label_start_index_ = 1;
    double match_xy_threshold_ = 0.20;
    double nearest_match_xyz_threshold_ = 0.50;
    double coordinate_match_tolerance_ = 0.05;
    double coordinate_hash_resolution_ = 0.01;
    double intensity_tolerance_ = 8.0;
    bool strict_order_labels_ = true;
    bool coordinate_label_matching_ = false;
    bool nearest_label_matching_ = true;
    bool publish_label_cloud_as_input_ = false;
    double label_cloud_time_span_ = 0.10;
    bool skip_unlabeled_frames_ = true;
    bool input_points_in_body_frame_ = true;
    bool output_points_in_body_frame_ = true;
    uint32_t unlabeled_semantic_ = 0;
    int log_interval_ = 100;
    std::array<double, 9> body_to_mid70_R_{{
        0.9998581316050806, -0.0002258196228773494, -0.016842377168758943,
        0.00017111407692153792, 0.999994705851753, -0.0032494597148797887,
        0.016843021794484745, 0.003246116751423288, 0.9998528768488223}};
    std::array<double, 3> body_to_mid70_T_{{-0.0305152992417423, 0.0089897848417584, -0.0003564686383346}};
    std::array<double, 9> output_body_R_{{
        0.9998581316050806, -0.0002258196228773494, -0.016842377168758943,
        -0.00017111407692153792, -0.999994705851753, 0.0032494597148797887,
        -0.016843021794484745, -0.003246116751423288, -0.9998528768488223}};
    std::array<double, 3> output_body_T_{{-0.010514839241742317, -0.008989784841758377, 0.03735646863833463}};
    std::set<int> plane_labels_;
    std::set<int> object_labels_;
    std::set<int> unknown_labels_;
    std::vector<PointFieldMsg> fields_;
    std::unordered_map<int, std::shared_ptr<LabelFrame>> label_cache_;
    rclcpp::Publisher<PointCloud2Msg>::SharedPtr publisher_;
    rclcpp::Subscription<LivoxCustomMsg>::SharedPtr subscriber_;
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<McdMid70SemanticGtBridge>());
    rclcpp::shutdown();
    return 0;
}
