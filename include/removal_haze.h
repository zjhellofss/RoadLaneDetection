//
// Created by fss on 2021/8/2.
//

#ifndef ROADLANEDETECTION_REMOVAL_HAZE_H
#define ROADLANEDETECTION_REMOVAL_HAZE_H
#include <opencv2/opencv.hpp>
cv::Mat HazeRemoval(cv::Mat &source, cv::Mat &output, int minr = 5, int maxA = 230, double w = 0.98, int guider = 30,
                    double guideeps = 0.001, int L = 0);
cv::Mat dehaze(cv::Mat src);
#endif //ROADLANEDETECTION_REMOVAL_HAZE_H
