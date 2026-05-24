#include <algorithm>
#include <array>
#include <cctype>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <functional>
#include <sstream>
#include <string>
#include <unordered_set>
#include <vector>

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/msg/point_field.hpp>

using PointCloud2Msg = sensor_msgs::msg::PointCloud2;
using PointFieldMsg = sensor_msgs::msg::PointField;

namespace
{
PointFieldMsg makeField(const std::string &name, const uint32_t offset, const uint8_t datatype)
{
    PointFieldMsg field;
    field.name = name;
    field.offset = offset;
    field.datatype = datatype;
    field.count = 1;
    return field;
}

int fieldOffset(const PointCloud2Msg &msg, const std::string &name, const uint8_t datatype)
{
    for (const auto &field : msg.fields)
    {
        if (field.name == name && field.datatype == datatype)
            return static_cast<int>(field.offset);
    }
    return -1;
}

void writeUint32(std::vector<uint8_t> &data, const size_t offset, const uint32_t value)
{
    std::memcpy(data.data() + offset, &value, sizeof(uint32_t));
}

bool readLabels(const std::string &path, std::vector<uint32_t> &labels)
{
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file.is_open())
        return false;
    const std::streamsize size = file.tellg();
    if (size <= 0 || size % static_cast<std::streamsize>(sizeof(uint32_t)) != 0)
        return false;
    labels.resize(static_cast<size_t>(size / sizeof(uint32_t)));
    file.seekg(0, std::ios::beg);
    return static_cast<bool>(file.read(reinterpret_cast<char *>(labels.data()), size));
}

std::unordered_set<uint32_t> parseLabelSet(const std::string &text)
{
    std::unordered_set<uint32_t> labels;
    std::stringstream stream(text);
    std::string token;
    while (std::getline(stream, token, ','))
    {
        token.erase(std::remove_if(token.begin(), token.end(), ::isspace), token.end());
        if (!token.empty())
            labels.insert(static_cast<uint32_t>(std::stoul(token)));
    }
    return labels;
}

uint32_t semanticKittiToHybridCategory(const uint32_t raw_label,
                                       const std::unordered_set<uint32_t> &plane_labels)
{
    const uint32_t semantic_label = raw_label & 0xFFFFu;
    if (plane_labels.find(semantic_label) != plane_labels.end())
        return 1;

    switch (semantic_label)
    {
    case 40:
    case 44:
    case 48:
    case 49:
    case 50:
    case 51:
    case 52:
    case 60:
    case 70:
    case 71:
    case 72:
    case 80:
    case 81:
    case 99:
        return 2;
    case 10:
    case 11:
    case 13:
    case 15:
    case 16:
    case 18:
    case 20:
    case 30:
    case 31:
    case 32:
    case 252:
    case 253:
    case 254:
    case 255:
    case 256:
    case 257:
    case 258:
    case 259:
        return 3;
    default:
        return 0;
    }
}
} // namespace

class KittiGtSemanticBridge : public rclcpp::Node
{
public:
    KittiGtSemanticBridge() : Node("kitti_gt_semantic_bridge_cpp")
    {
        input_topic_ = declare_parameter<std::string>("input_topic", "/velodyne_points_raw");
        output_topic_ = declare_parameter<std::string>("output_topic", "/velodyne_points");
        label_dir_ = declare_parameter<std::string>(
            "label_dir", "data/KITTI/dataset/sequences/08/labels");
        plane_label_text_ = declare_parameter<std::string>("plane_labels", "40,44,48,49,50,51,52,60,72");
        frame_index_ = declare_parameter<int>("start_index", 0);
        log_interval_ = std::max(1, static_cast<int>(declare_parameter<int>("log_interval", 500)));
        const int qos_depth = std::max(1, static_cast<int>(declare_parameter<int>("qos_depth", 512)));
        plane_labels_ = parseLabelSet(plane_label_text_);

        auto qos = rclcpp::QoS(rclcpp::KeepLast(qos_depth)).reliable();
        publisher_ = create_publisher<PointCloud2Msg>(output_topic_, qos);
        subscription_ = create_subscription<PointCloud2Msg>(
            input_topic_, qos, std::bind(&KittiGtSemanticBridge::cloudCallback, this, std::placeholders::_1));

        RCLCPP_INFO(get_logger(), "KITTI GT semantic bridge: %s -> %s, labels=%s, plane={%s}",
                    input_topic_.c_str(), output_topic_.c_str(), label_dir_.c_str(), plane_label_text_.c_str());
    }

private:
    void cloudCallback(const PointCloud2Msg::SharedPtr msg)
    {
        const size_t point_num = static_cast<size_t>(msg->width) * static_cast<size_t>(msg->height);
        std::vector<uint32_t> labels;
        const std::string label_path = label_dir_ + "/" + sixDigit(frame_index_) + ".label";
        const bool labels_ok = readLabels(label_path, labels) && labels.size() == point_num;
        if (!labels_ok)
        {
            RCLCPP_WARN(get_logger(), "label mismatch frame=%d: labels=%zu points=%zu; using unknown",
                        frame_index_, labels.size(), point_num);
        }

        PointCloud2Msg out = *msg;
        int semantic_offset = fieldOffset(out, "semantic", PointFieldMsg::UINT32);
        if (semantic_offset < 0)
            semantic_offset = 28;

        if (semantic_offset + 4 <= static_cast<int>(out.point_step))
        {
            if (fieldOffset(out, "semantic", PointFieldMsg::UINT32) < 0)
                out.fields.push_back(makeField("semantic", static_cast<uint32_t>(semantic_offset), PointFieldMsg::UINT32));
            out.data = msg->data;
        }
        else
        {
            const uint32_t old_step = out.point_step;
            out.fields.push_back(makeField("semantic", old_step, PointFieldMsg::UINT32));
            out.point_step = old_step + 4;
            out.row_step = out.point_step * out.width;
            out.data.assign(point_num * out.point_step, 0);
            for (size_t i = 0; i < point_num; ++i)
            {
                std::memcpy(out.data.data() + i * out.point_step,
                            msg->data.data() + i * old_step,
                            old_step);
            }
            semantic_offset = static_cast<int>(old_step);
        }

        std::array<size_t, 4> counts = {0, 0, 0, 0};
        for (size_t i = 0; i < point_num; ++i)
        {
            const uint32_t category = labels_ok ? semanticKittiToHybridCategory(labels[i], plane_labels_) : 0;
            writeUint32(out.data, i * out.point_step + static_cast<size_t>(semantic_offset), category);
            if (category <= 3)
                ++counts[category];
        }
        publisher_->publish(std::move(out));

        if (frame_index_ == 0 || (frame_index_ + 1) % log_interval_ == 0)
        {
            RCLCPP_INFO(get_logger(),
                        "GT semantic frame %d: unknown=%zu plane=%zu gaussian=%zu object=%zu",
                        frame_index_, counts[0], counts[1], counts[2], counts[3]);
        }
        ++frame_index_;
    }

    std::string sixDigit(const int index) const
    {
        char buffer[32];
        std::snprintf(buffer, sizeof(buffer), "%06d", index);
        return std::string(buffer);
    }

    std::string input_topic_;
    std::string output_topic_;
    std::string label_dir_;
    std::string plane_label_text_;
    std::unordered_set<uint32_t> plane_labels_;
    int frame_index_ = 0;
    int log_interval_ = 500;
    rclcpp::Subscription<PointCloud2Msg>::SharedPtr subscription_;
    rclcpp::Publisher<PointCloud2Msg>::SharedPtr publisher_;
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<KittiGtSemanticBridge>());
    rclcpp::shutdown();
    return 0;
}
