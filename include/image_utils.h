//
// Created by fss on 2021/7/29.
//

#ifndef ROADLANEDETECTION_IMAGE_UTILS_H
#define ROADLANEDETECTION_IMAGE_UTILS_H
#include "opencv2/opencv.hpp"


/**
 *
 * @param src
 * @param dst
 * @param clipHistPercent 删除的像素比例
 * @param histSize 灰度图的分布
 * @param lowhist 最小灰度
 */
void brightness_contrast_auto(const cv::Mat &src, cv::Mat &dst, float clipHistPercent, int histSize, int lowhist);
/**
 * 提取白色的像素分布
 * @param img
 * @param dst
 */
void combined_threshold(const cv::Mat &img, cv::Mat &dst,int threshold);

/**
 * 检测图像中的直线段
 * @param img
 * @param length_threshold 最短长度
 * @return
 */
std::vector<cv::Vec4i> get_lines(const cv::Mat &gray_img,int length_threshold);


bool extended_bounding_rectangle_line_equivalence(const cv::Vec4i &_l1, const cv::Vec4i &_l2, float extensionLengthFraction,
                                                  float maxAngleDiff, float boundingRectangleThickness);
#endif //ROADLANEDETECTION_IMAGE_UTILS_H
