#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include <livox_ros_driver2/msg/custom_msg.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/msg/point_field.hpp>

namespace
{
using LivoxCustomMsg = livox_ros_driver2::msg::CustomMsg;
using PointCloud2Msg = sensor_msgs::msg::PointCloud2;
using PointFieldMsg = sensor_msgs::msg::PointField;

PointFieldMsg makeField(const std::string &name, const uint32_t offset, const uint8_t datatype)
{
    PointFieldMsg field;
    field.name = name;
    field.offset = offset;
    field.datatype = datatype;
    field.count = 1;
    return field;
}

template <typename T>
void writeValue(std::vector<uint8_t> &data, const size_t offset, const T &value)
{
    std::memcpy(data.data() + offset, &value, sizeof(T));
}
} // namespace

class LivoxCustomToPointCloud2 : public rclcpp::Node
{
public:
    LivoxCustomToPointCloud2()
        : Node("livox_custom_to_pointcloud2")
    {
        input_topic_ = declare_parameter<std::string>("input_topic", "/livox/lidar");
        output_topic_ = declare_parameter<std::string>("output_topic", "/livox/lidar_raw_body");
        frame_id_ = declare_parameter<std::string>("frame_id", "camera_init");
        scan_line_ = std::max(1, static_cast<int>(declare_parameter<int>("scan_line", 1)));
        point_filter_num_ = std::max(1, static_cast<int>(declare_parameter<int>("point_filter_num", 1)));
        keep_reserved_semantic_slot_ = declare_parameter<bool>("keep_reserved_semantic_slot", true);

        fields_.push_back(makeField("x", 0, PointFieldMsg::FLOAT32));
        fields_.push_back(makeField("y", 4, PointFieldMsg::FLOAT32));
        fields_.push_back(makeField("z", 8, PointFieldMsg::FLOAT32));
        fields_.push_back(makeField("intensity", 16, PointFieldMsg::FLOAT32));
        fields_.push_back(makeField("time", 20, PointFieldMsg::FLOAT32));
        fields_.push_back(makeField("ring", 24, PointFieldMsg::UINT16));

        publisher_ = create_publisher<PointCloud2Msg>(output_topic_, rclcpp::SensorDataQoS().keep_last(512));
        subscriber_ = create_subscription<LivoxCustomMsg>(
            input_topic_, rclcpp::SensorDataQoS().keep_last(512),
            std::bind(&LivoxCustomToPointCloud2::callback, this, std::placeholders::_1));

        RCLCPP_INFO(get_logger(), "Livox CustomMsg -> PointCloud2 converter: %s -> %s",
                    input_topic_.c_str(), output_topic_.c_str());
    }

private:
    void callback(const LivoxCustomMsg::ConstSharedPtr msg)
    {
        std::vector<size_t> valid_indices;
        valid_indices.reserve(msg->points.size());
        for (size_t i = 1; i < msg->points.size(); ++i)
        {
            const auto &point = msg->points[i];
            if (static_cast<int>(point.line) >= scan_line_)
            {
                continue;
            }
            const int tag = static_cast<int>(point.tag) & 0x30;
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
            if (std::abs(point.x - prev.x) <= 1e-7f &&
                std::abs(point.y - prev.y) <= 1e-7f &&
                std::abs(point.z - prev.z) <= 1e-7f)
            {
                continue;
            }
            valid_indices.push_back(i);
        }

        PointCloud2Msg out;
        out.header = msg->header;
        if (!frame_id_.empty())
        {
            out.header.frame_id = frame_id_;
        }
        out.height = 1;
        out.width = static_cast<uint32_t>(valid_indices.size());
        out.fields = fields_;
        out.is_bigendian = false;
        out.point_step = keep_reserved_semantic_slot_ ? 32 : 26;
        out.row_step = out.point_step * out.width;
        out.is_dense = true;
        out.data.assign(static_cast<size_t>(out.row_step), 0);

        for (size_t out_index = 0; out_index < valid_indices.size(); ++out_index)
        {
            const auto &point = msg->points[valid_indices[out_index]];
            const size_t base = out_index * static_cast<size_t>(out.point_step);
            const float x = point.x;
            const float y = point.y;
            const float z = point.z;
            const float intensity = static_cast<float>(point.reflectivity);
            const float time = static_cast<float>(point.offset_time) * 1e-9f;
            const uint16_t ring = static_cast<uint16_t>(point.line);
            writeValue(out.data, base + 0, x);
            writeValue(out.data, base + 4, y);
            writeValue(out.data, base + 8, z);
            writeValue(out.data, base + 16, intensity);
            writeValue(out.data, base + 20, time);
            writeValue(out.data, base + 24, ring);
        }
        publisher_->publish(out);
    }

    std::string input_topic_;
    std::string output_topic_;
    std::string frame_id_;
    int scan_line_ = 1;
    int point_filter_num_ = 1;
    bool keep_reserved_semantic_slot_ = true;
    std::vector<PointFieldMsg> fields_;
    rclcpp::Publisher<PointCloud2Msg>::SharedPtr publisher_;
    rclcpp::Subscription<LivoxCustomMsg>::SharedPtr subscriber_;
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<LivoxCustomToPointCloud2>());
    rclcpp::shutdown();
    return 0;
}
