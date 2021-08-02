#include <iostream>
#include <opencv2/opencv.hpp>
#include <algorithm>
#include <numeric>
#include <utility>
#include <dbscan.h>

#include "opencv2/imgproc.hpp"
#include "boost/program_options.hpp"
#include "image_utils.h"
#include "lines.h"
#include "removal_haze.h"

int main(int argc, char *argv[])
{
    namespace bpo = boost::program_options;
    bpo::options_description opts("all options");
    bpo::variables_map vm;
    int threshold = 0, length = 0;
    float distance = 0;
    opts.add_options()
            ("filename,f", bpo::value<std::string>(), "图像路径")
            ("threshold,t", bpo::value<int>(&threshold)->default_value(190), "图像亮度阈值")
            ("distance,d", bpo::value<float>(&distance)->default_value(50.0), "线条距离阈值")
            ("length,l", bpo::value<int>(&length)->default_value(50), "线段长度阈值");
    try
    {
        bpo::store(bpo::parse_command_line(argc, argv, opts), vm);
    }
    catch (...)
    {
        std::cout << opts << std::endl;
        return -1;
    }
    bpo::notify(vm);
    //打开图像
    std::string path = vm["filename"].as<std::string>();
    cv::Mat image = cv::imread(path);
    if (image.empty())
    {
        return -1;
    }
    brightness_contrast_auto(image, image, 1, 256, 0);
    image = dehaze(image);
    //调整图像大小
    cv::Mat smaller_image;
    cv::resize(image, smaller_image, cv::Size(), 0.5, 0.5, cv::INTER_CUBIC);
    cv::Mat target = smaller_image.clone();

    // 对比和亮度调整
    cv::imwrite("/home/fss/CLionProjects/RoadLaneDetection/tmp/ba.jpg", target);

    //阈值图像
    cv::Mat grayscale;
    combined_threshold(target, grayscale, threshold);
    cv::imwrite("/home/fss/CLionProjects/RoadLaneDetection/tmp/ab.jpg", grayscale);
    auto height = grayscale.size().height;
    auto width = grayscale.size().width;

    cv::Mat detected_lines_img = cv::Mat::zeros(target.rows, target.cols, CV_8UC3);
    cv::Mat reduced_lines_img = target.clone();

    //找线段
    auto lines_target = get_lines(grayscale, length);

    //距离、角度过近的线段变成一条线
    std::vector<int> labels;
    int equilavence_classes_count = cv::partition(lines_target, labels, [=](const cv::Vec4i &l1, const cv::Vec4i &l2)
    {
        return extended_bounding_rectangle_line_equivalence(
                l1, l2,
                .1,
                12.0,
                distance);
    });


    cv::RNG rng(31131);
    std::vector<cv::Scalar> colors(equilavence_classes_count);
    for (int i = 0; i < equilavence_classes_count; i++)
    {
        colors[i] = cv::Scalar(rng.uniform(30, 255), rng.uniform(30, 255), rng.uniform(30, 255));
    }


    for (int i = 0; i < lines_target.size(); i++)
    {
        cv::Vec4i &detectedLine = lines_target[i];
        line(detected_lines_img,
             cv::Point(detectedLine[0], detectedLine[1]),
             cv::Point(detectedLine[2], detectedLine[3]), colors[labels[i]], 2);
    }


    // 线到点，用于拟合
    std::vector<std::vector<cv::Point2i>> point_clouds(equilavence_classes_count);
    for (int i = 0; i < lines_target.size(); i++)
    {
        cv::Vec4i &detectedLine = lines_target[i];
        point_clouds[labels[i]].push_back(cv::Point2i(detectedLine[0], detectedLine[1]));
        point_clouds[labels[i]].push_back(cv::Point2i(detectedLine[2], detectedLine[3]));
    }

    // 合并
    std::vector<cv::Vec4i> reducedLines = std::accumulate(point_clouds.begin(), point_clouds.end(),
                                                          std::vector<cv::Vec4i>{},
                                                          [](std::vector<cv::Vec4i> target,
                                                             const std::vector<cv::Point2i> &_pointCloud)
                                                          {
                                                              std::vector<cv::Point2i> pointCloud = _pointCloud;

                                                              cv::Vec4f lineParams;
                                                              fitLine(pointCloud, lineParams, cv::DIST_L2, 0, 0.01,
                                                                      0.01);
                                                              decltype(pointCloud)::iterator minXP, maxXP;
                                                              std::tie(minXP, maxXP) = std::minmax_element(
                                                                      pointCloud.begin(), pointCloud.end(),
                                                                      [](const cv::Point2i &p1, const cv::Point2i &p2)
                                                                      { return p1.x < p2.x; });


                                                              float m = lineParams[1] / lineParams[0];
                                                              int y1 = ((minXP->x - lineParams[2]) * m) + lineParams[3];
                                                              int y2 = ((maxXP->x - lineParams[2]) * m) + lineParams[3];

                                                              target.emplace_back(minXP->x, y1, maxXP->x, y2);
                                                              return target;
                                                          });

    std::vector<ReducedLine> output_lines;
    double K_all = 0.0;
    for (cv::Vec4i reduced: reducedLines)
    {
        auto color = cv::Scalar(rng.uniform(30, 255), rng.uniform(30, 255), rng.uniform(30, 255));

        auto l = ReducedLine(cv::Point(reduced[0], reduced[1]), cv::Point(reduced[2], reduced[3]));

        double degree = l.degree();
        if (degree < 5)
        {
            continue;
        }

        double K = l.K();
//        if (K > 0)
//        {
//            K_all += 1;
//        }
//        else
//        {
//            K_all -= 1;
//        }
        cv::Point2d start_point = getLine(l.point1_, K, 0);
        if (start_point.y > height)
        {
            start_point = getLine(l.point1_, K, -1, height);
        }
        cv::Point2d end_point = getLine(l.point1_, K, width);
        if (end_point.y < 0)
        {
            end_point = getLine(l.point1_, K, -1, 0);
        }
        output_lines.emplace_back(start_point, end_point);

    }

    // 删除角度不对的
//    output_lines.erase(
//            std::remove_if(
//                    output_lines.begin(),
//                    output_lines.end(),
//                    [&](const ReducedLine &p)
//                    {
//                        if (K_all > 0 && p.K() < 0)
//                        {
//                            return true;
//                        }
//                        else if (K_all < 0 && p.K() > 0)
//                        {
//                            return true;
//                        }
//                        else
//                        {
//                            return false;
//                        }
//                    }
//            ),
//            output_lines.end()
//    );

    //计算交点
    std::vector<vec2f> all_points;
    for (int i = 0; i < output_lines.size(); ++i)
    {
        auto &line1 = output_lines.at(i);
        auto color = cv::Scalar(rng.uniform(30, 255), rng.uniform(30, 255), rng.uniform(30, 255));
        for (const auto &line2 : output_lines)
        {
            cv::Point2f r;

            bool is_intersection = intersection(line1, line2, r);
            circle(reduced_lines_img, r, 5, cv::Scalar(255), -1);

            if (is_intersection)
            {
                line1.appendInterPoint(r);
                line1.appendInterPoint(r);
                if (r.y < height / 2)
                    all_points.emplace_back(r.x, r.y);
            }
        }
    }

    // 交点聚类
    auto dbscan = DBSCAN<vec2f, float>();
    dbscan.Run(&all_points, 2, 60, int(all_points.size() / 10) + 1);
    auto noise = dbscan.Noise;
    auto clusters = dbscan.Clusters;

    int inter_on_top = 0; //聚合到高位的次数
    for (auto &c:clusters)
    {
        for (auto &index:c)
        {
            circle(reduced_lines_img, cv::Point2f(all_points[index].data[0], all_points[index].data[1]), 5,
                   cv::Scalar(0, 0, 255), -1);
            auto p_x = all_points[index].data[0];
            auto p_y = all_points[index].data[1];

            for (auto &line:output_lines)
            {
                auto points = line.inter_points;
                for (auto &p:points)
                {
                    if (p.x == p_x && p.y == p_y)
                    {
                        line.is_inter = true;
                    }
                }
            }
        }
    }


    for (auto &line:output_lines)
    {
        if (line.is_inter)
        {
            cv::line(reduced_lines_img, line.point1_, line.point2_, cv::Scalar(inter_on_top * 255, 255, 0), 2);
        }
    }

    cv::imwrite("/home/fss/CLionProjects/RoadLaneDetection/tmp/o1.jpg", detected_lines_img);
    cv::imwrite("/home/fss/CLionProjects/RoadLaneDetection/tmp/o2.jpg", reduced_lines_img);


    return 0;
}
