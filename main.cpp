#include <iostream>
#include <opencv2/opencv.hpp>
#include <algorithm>
#include <numeric>
#include <utility>
#include <dbscan.h>

#include "windowbox.h"
#include "opencv2/imgproc.hpp"
#include "boost/program_options.hpp"
#include "image_utils.h"
#include "lines.h"
#include "removal_haze.h"
#include <cmath>

#define N_WINDOWS 9
#define WINDOW_WIDTH 1

bool intersection(const cv::Point2f &o1, const cv::Point2f &p1, const cv::Point2f &o2, const cv::Point2f &p2,
                  cv::Point2f &r)
{
    cv::Point2f x = o2 - o1;
    cv::Point2f d1 = p1 - o1;
    cv::Point2f d2 = p2 - o2;

    float cross = d1.x * d2.y - d1.y * d2.x;
    if (abs(cross) < /*EPS*/1e-8)
        return false;

    double t1 = (x.x * d2.y - x.y * d2.x) / cross;
    r = o1 + d1 * t1;
    return true;
}

std::vector<cv::Point2i>
get_rotate_points(const cv::Mat &src_image, std::vector<cv::Point2i> Points, const cv::Point &rotate_center,
                  const double angle)
{
    const double pi = 3.1415926;
    std::vector<cv::Point2i> dst_points;
    int x1 = 0, y1 = 0;
    int row = src_image.rows;
    for (size_t i = 0; i < Points.size(); i++)
    {
        x1 = Points.at(i).x;
        y1 = row - Points.at(i).y;
        int x2 = rotate_center.x;
        int y2 = row - rotate_center.y;
        int x = cvRound((x1 - x2) * cos(pi / 180.0 * angle) - (y1 - y2) * sin(pi / 180.0 * angle) + x2);
        int y = cvRound((x1 - x2) * sin(pi / 180.0 * angle) + (y1 - y2) * cos(pi / 180.0 * angle) + y2);
        y = row - y;
        dst_points.emplace_back(x, y);
    }
    return dst_points;
}

//计算直方图中
inline void lane_histogram(cv::Mat const &img, cv::Mat &histogram)
{
    // Histogram
    cv::Mat cropped = img(cv::Rect(0, img.rows / 2, img.cols, img.rows / 2));
    cv::reduce(cropped / 255, histogram, 0, cv::REDUCE_SUM, CV_32S);
}

void find_lane_windows(cv::Mat &binary_img, WindowBox &window_box, std::vector<WindowBox> &wboxes)
{
    bool continue_lane_search = true;
    int contiguous_box_no_line_count = 0;

    // keep searching up the image for a lane lineand append the boxes
    while (continue_lane_search && window_box.y_top > 0)
    {
        if (window_box.has_line())
            wboxes.push_back(window_box);
        window_box = window_box.get_next_windowbox(binary_img);

        // if we've found the lane and can no longer find a box with a line in it
        // then its no longer worth while searching
        if (window_box.has_lane())
            if (window_box.has_line())
            {
                contiguous_box_no_line_count = 0;
            }
            else
            {
                contiguous_box_no_line_count += 1;
                if (contiguous_box_no_line_count >= 4)
                    continue_lane_search = false;
            }
    }
}

void calc_lane_windows(cv::Mat &binary_img, int nwindows, int width,
                       std::vector<WindowBox> &left_boxes, std::vector<WindowBox> &right_boxes, int x)
{
    // calc height of each window
    int ytop = binary_img.rows;
    int height = ytop / nwindows;


    // Initialise left and right window boxes
    WindowBox wbl(binary_img, x, ytop, width, height);
    find_lane_windows(binary_img, wbl, right_boxes);
}

void polyfit(const cv::Mat &src_x, const cv::Mat &src_y, cv::Mat &dst, int order)
{
    CV_Assert((src_x.rows > 0) && (src_y.rows > 0) && (src_x.cols == 1) && (src_y.cols == 1)
              && (dst.cols == 1) && (dst.rows == (order + 1)) && (order >= 1));
    cv::Mat X;
    X = cv::Mat::zeros(src_x.rows, order + 1, CV_32FC1);
    cv::Mat copy;
    for (int i = 0; i <= order; i++)
    {
        copy = src_x.clone();
        pow(copy, i, copy);
        cv::Mat M1 = X.col(i);
        copy.col(0).copyTo(M1);
    }
    cv::Mat X_t, X_inv;
    transpose(X, X_t);
    cv::Mat temp = X_t * X;
    cv::Mat temp2;
    invert(temp, temp2);
    cv::Mat temp3 = temp2 * X_t;
    cv::Mat W = temp3 * src_y;
    W.copyTo(dst);
}

void poly_fitx(std::vector<double> const &fity, std::vector<double> &fitx, cv::Mat const &line_fit)
{
    for (auto const &y : fity)
    {
        double x = line_fit.at<float>(2, 0) * y * y + line_fit.at<float>(1, 0) * y + line_fit.at<float>(0, 0);
        fitx.push_back(x);
    }



}

template<typename T>
std::vector<double> linspace(T start_in, T end_in, int num_in)
{
    std::vector<double> linspaced;
    double start = static_cast<double>(start_in);
    double end = static_cast<double>(end_in);
    double num = static_cast<double>(num_in);

    if (num == 0)
    { return linspaced; }
    if (num == 1)
    {
        linspaced.push_back(start);
        return linspaced;
    }

    double delta = (end - start) / (num - 1);

    for (int i = 0; i < num - 1; ++i)
    {
        linspaced.push_back(start + delta * i);
    }
    linspaced.push_back(end);

    return linspaced;
}


cv::Mat draw_line_curve(cv::Mat &img, cv::Mat &left_fit)
{
    int y_max = img.rows;
    std::vector<double> fity = linspace<double>(0, y_max - 1, y_max);
    cv::Mat color_warp = cv::Mat::zeros(img.size(), CV_8UC3);

    // Calculate Points
    //拟合点
    std::vector<double> left_fitx;
    poly_fitx(fity, left_fitx, left_fit);
    int npoints = fity.size();

    std::vector<cv::Point> pts_left(npoints),  pts;
    for (int i = 0; i < npoints; i++)
    {
        pts_left[i] = cv::Point(left_fitx[i], fity[i]);
    }
    cv::Mat curve1(pts_left, true);
    curve1.convertTo(curve1, CV_32S); //adapt type for polylines

    cv::Scalar red(0, 255, 255);
    cv::polylines(img, curve1, false, red, 8);
    return img;
//    cv::imwrite("/home/fss/CLionProjects/RoadLaneDetection/tmp/line.jpg",color_warp);
//    cv::Mat new_warp;
//    perspective_warp(color_warp, new_warp, Minv);
//    cv::addWeighted(img, 1, new_warp, 0.3, 0, out_img);
}


cv::Mat calc_fit_from_boxes(std::vector<WindowBox> const &boxes)
{
    int n = boxes.size();
    std::vector<cv::Mat> xmatrices, ymatrices;
    xmatrices.reserve(n);
    ymatrices.reserve(n);

    cv::Mat xtemp, ytemp;
    for (auto const &box : boxes)
    {
        // get matpoints
        box.get_indices(xtemp, ytemp);
        xmatrices.push_back(xtemp);
        ymatrices.push_back(ytemp);
    }
    cv::Mat xs, ys;
    cv::vconcat(xmatrices, xs);
    cv::vconcat(ymatrices, ys);

    // Fit a second order polynomial to each
    cv::Mat fit = cv::Mat::zeros(3, 1, CV_32F);
    polyfit(ys, xs, fit, 2);

    return fit;
}


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
    if (height > width)
    {
        length = height / 20;
    }
    else // width > height
    {
        length = width / 20;
    }
    //找线段
    auto lines_target = get_lines(grayscale, length);

    //超出的一半的直线进行剪裁

    cv::Point2f mid1 = cv::Point2f(0, grayscale.size().height / 2);
    cv::Point2f mid2 = cv::Point2f(grayscale.size().width, grayscale.size().height / 2);
    for (auto &line:lines_target)
    {
//        (line[2] - line[0])
        auto max_y = line[3] < line[1] ? line[3] : line[1];
        if (max_y < grayscale.size().height / 2)
        {
            cv::Point2f v1 = cv::Point2f(line[0], line[1]);
            cv::Point2f v2 = cv::Point2f(line[2], line[3]);
            //计算交点
            cv::Point2f r;
            bool is_intersection = intersection(mid1, mid2, v1, v2, r);
            if (is_intersection)
            {
//                if (line[3] < line[1])
//                {
//                    line[3] = r.y;
//                    line[2] = r.x;
//                }
//                else
//                {
//                    line[1] = r.y;
//                    line[0] = r.x;
//                }
            }

        }
    }
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


    // 计算斜率

    for (auto &line:output_lines)
    {
        if (line.is_inter)
        {
            cv::line(reduced_lines_img, line.point1_, line.point2_, cv::Scalar(inter_on_top * 255, 255, 0), 2);
        }
    }

    int index = 3;
//    for (auto &line:output_lines)
//    {
//        //角度
//        double degree = 90 - line.degree();
//        //开始的位置
//        auto start_point = line.point1_;
//        cv::Size dst_sz(grayscale.cols, grayscale.rows);
//        cv::Point2f center(grayscale.cols / 2., grayscale.rows / 2.);
//        cv::Mat rot_mat = cv::getRotationMatrix2D(center, degree, 1.0);
//
//        cv::Scalar borderColor = cv::Scalar(0, 0, 0);
//        cv::Mat dst,dst2;
//        cv::line(grayscale, line.point1_, line.point2_, cv::Scalar(inter_on_top * 255, 255, 0), 2);
////        cv::imwrite("/home/fss/CLionProjects/RoadLaneDetection/tmp/o3.jpg",target);
//        cv::warpAffine(grayscale, dst, rot_mat, dst_sz, cv::INTER_LINEAR, cv::BORDER_CONSTANT, borderColor);
//        cv::warpAffine(target, dst2, rot_mat, dst_sz, cv::INTER_LINEAR, cv::BORDER_CONSTANT, borderColor);
//
//        cv::Point2i point(start_point.x, start_point.y);
//        std::vector<cv::Point2i> Points;
//        Points.push_back(point);
//        Points.push_back(line.point2_);
//        std::vector<cv::Point2i> dst_points = get_rotate_points(dst, Points, center, degree);
//        if (dst_points[0].y > height)
//        {
//            dst_points[0].y = height;
//        }
//        circle(dst, dst_points[0], 10, cv::Scalar(255, 255, 0), -1);
//        circle(dst2, dst_points[0], 10, cv::Scalar(255, 255, 0), -1);
//
//        std::vector<WindowBox> boxes;
//        calc_lane_windows(grayscale, N_WINDOWS, WINDOW_WIDTH, boxes, boxes, dst_points[0].x);
//        cv::Mat left_fit = calc_fit_from_boxes(boxes);
//        dst2 = draw_line_curve(dst2,left_fit);
//
//        cv::imwrite("/home/fss/CLionProjects/RoadLaneDetection/tmp/i" + std::to_string(index) + ".jpg", dst2);
//        index += 1;
//    }

    cv::imwrite("/home/fss/CLionProjects/RoadLaneDetection/tmp/o1.jpg", detected_lines_img);
    cv::imwrite("/home/fss/CLionProjects/RoadLaneDetection/tmp/o2.jpg", reduced_lines_img);


    return 0;
}
