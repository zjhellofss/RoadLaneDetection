#include "image_utils.h"
#include "opencv2/ximgproc.hpp"

void brightness_contrast_auto(const cv::Mat &src, cv::Mat &dst, float clipHistPercent, int histSize, int lowhist)
{
    CV_Assert(clipHistPercent >= 0);
    CV_Assert((src.type() == CV_8UC1) || (src.type() == CV_8UC3) || (src.type() == CV_8UC4));

    float alpha, beta;
    double minGray = 0, maxGray = 0;

    cv::Mat gray;
    if (src.type() == CV_8UC1) gray = src;
    else if (src.type() == CV_8UC3) cvtColor(src, gray, cv::COLOR_BGR2GRAY);
    else if (src.type() == CV_8UC4) cvtColor(src, gray, cv::COLOR_BGR2GRAY);
    if (clipHistPercent == 0)
    {
        cv::minMaxLoc(gray, &minGray, &maxGray);
    }
    else
    {
        cv::Mat hist;
        float range[] = {1, 256};
        const float *histRange = {range};
        bool uniform = true;
        bool accumulate = false;
        calcHist(&gray, 1, 0, cv::Mat(), hist, 1, &histSize, &histRange, uniform, accumulate);

        std::vector<float> accumulator(histSize);
        accumulator[0] = hist.at<float>(0);
        for (int i = 1; i < histSize; i++)
        {
            accumulator[i] = accumulator[i - 1] + hist.at<float>(i);
        }

        float max = accumulator.back();

        int clipHistPercent2;
        clipHistPercent2 = clipHistPercent * (max / 100.0);
        clipHistPercent2 /= 2.0;

        minGray = 0;
        while (accumulator[minGray] < clipHistPercent2)
            minGray++;

        maxGray = histSize - 1;
        while (accumulator[maxGray] >= (max - clipHistPercent2))
            maxGray--;
    }


    float inputRange = maxGray - minGray;
    alpha = (histSize - 1) / inputRange;
    beta = -minGray * alpha + lowhist;

    src.convertTo(dst, -1, alpha, beta);
    if (dst.type() == CV_8UC4)
    {
        int from_to[] = {3, 3};
        cv::mixChannels(&src, 4, &dst, 1, from_to, 1);
    }
}

static void sobel_thresh(cv::Mat const &src, cv::Mat &dest, int thresh_min = 0,
                         int thresh_max = 255)
{
    double min, max;
    auto image = src.clone();
    cv::minMaxLoc(image, &min, &max);
    cv::Mat scaled = 255 * (image / max);
    cv::inRange(scaled, cv::Scalar(thresh_min), cv::Scalar(thresh_max), dest);
}

void combined_threshold(const cv::Mat &img, cv::Mat &dst, int threshold)
{
    cv::Mat undist_hls;
    cv::cvtColor(img, undist_hls, cv::COLOR_BGR2HLS);

    cv::Mat hls_channels[3];
    cv::split(undist_hls, hls_channels);

    cv::Mat back_img = hls_channels[1];
    dst = back_img.clone();
    sobel_thresh(dst, dst, threshold, 255);

    cv::medianBlur(dst, dst, 5);
    cv::GaussianBlur(dst, dst, cv::Size(3, 3), 0, 0);

    auto kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
    cv::morphologyEx(dst, dst, cv::MORPH_CLOSE, kernel, cv::Point(-1, -1), 1);

    cv::medianBlur(dst, dst, 5);
    cv::GaussianBlur(dst, dst, cv::Size(3, 3), 0, 0);
}

std::vector<cv::Vec4i> get_lines(const cv::Mat &gray_img, int length_threshold)
{
    using namespace cv::ximgproc;

    float distance_threshold = 1.41421356f;
    double canny_th1 = 50;
    double canny_th2 = 50.0;
    int canny_aperture_size = 3;
    bool do_merge = true;
    cv::Ptr<FastLineDetector> fld = createFastLineDetector(length_threshold,
                                                       distance_threshold, canny_th1, canny_th2, canny_aperture_size,
                                                       do_merge);

    std::vector<cv::Vec4f> lines;
    fld->detect(gray_img, lines);

    std::vector<cv::Vec4i> lines_without_small;
    std::copy_if(lines.begin(), lines.end(), std::back_inserter(lines_without_small), [=](cv::Vec4f line)
    {
        float length = sqrtf((line[2] - line[0]) * (line[2] - line[0])
                             + (line[3] - line[1]) * (line[3] - line[1]));
        return length > length_threshold;
    });
    return lines_without_small;
}



static cv::Vec2d linear_parameters(cv::Vec4i line)
{
    cv::Mat a = (cv::Mat_<double>(2, 2) <<
                                line[0], 1,
            line[2], 1);
    cv::Mat y = (cv::Mat_<double>(2, 1) <<
                                line[1],
            line[3]);
    cv::Vec2d mc;
    solve(a, y, mc);
    return mc;
}

static std::vector<cv::Point2i> bounding_rectangle_contour(cv::Vec4i line, float d)
{


    cv::Vec2f mc = linear_parameters(line);
    float m = mc[0];
    float factor = sqrtf(
            (d * d) / (1 + (1 / (m * m)))
    );

    float x3, y3, x4, y4, x5, y5, x6, y6;

    if (m == 0)
    {
        x3 = line[0];
        y3 = line[1] + d;
        x4 = line[0];
        y4 = line[1] - d;
        x5 = line[2];
        y5 = line[3] + d;
        x6 = line[2];
        y6 = line[3] - d;
    }
    else
    {

        float m_per = -1 / m;


        float c_per1 = line[1] - m_per * line[0];
        float c_per2 = line[3] - m_per * line[2];


        x3 = line[0] + factor;
        y3 = m_per * x3 + c_per1;
        x4 = line[0] - factor;
        y4 = m_per * x4 + c_per1;
        x5 = line[2] + factor;
        y5 = m_per * x5 + c_per2;
        x6 = line[2] - factor;
        y6 = m_per * x6 + c_per2;
    }

    return std::vector<cv::Point2i>{
            cv::Point2i(x3, y3),
            cv::Point2i(x4, y4),
            cv::Point2i(x6, y6),
            cv::Point2i(x5, y5)
    };
}

static cv::Vec4i extended_line(cv::Vec4i line, double d)
{

    cv::Vec4d _line = line[2] - line[0] < 0 ? cv::Vec4d(line[2], line[3], line[0], line[1]) : cv::Vec4d(line[0], line[1], line[2],
                                                                                            line[3]);
    double m = linear_parameters(_line)[0];

    double xd = sqrt(d * d / (m * m + 1));
    double yd = xd * m;
    return cv::Vec4d(_line[0] - xd, _line[1] - yd, _line[2] + xd, _line[3] + yd);
}


bool extended_bounding_rectangle_line_equivalence(const cv::Vec4i &_l1, const cv::Vec4i &_l2, float extensionLengthFraction,
                                                  float maxAngleDiff, float boundingRectangleThickness)
{

    cv::Vec4i l1(_l1), l2(_l2);

    float len1 = sqrtf((l1[2] - l1[0]) * (l1[2] - l1[0]) + (l1[3] - l1[1]) * (l1[3] - l1[1]));
    float len2 = sqrtf((l2[2] - l2[0]) * (l2[2] - l2[0]) + (l2[3] - l2[1]) * (l2[3] - l2[1]));
    cv::Vec4i el1 = extended_line(l1, len1 * extensionLengthFraction);
    cv::Vec4i el2 = extended_line(l2, len2 * extensionLengthFraction);


    float a1 = atan(linear_parameters(el1)[0]);
    float a2 = atan(linear_parameters(el2)[0]);
    if (fabs(a1 - a2) > maxAngleDiff * M_PI / 180.0)
    {
        return false;
    }


    std::vector<cv::Point2i> lineBoundingContour = bounding_rectangle_contour(el1, boundingRectangleThickness / 2);
    return
            pointPolygonTest(lineBoundingContour, cv::Point(el2[0], el2[1]), false) == 1 ||
            pointPolygonTest(lineBoundingContour, cv::Point(el2[2], el2[3]), false) == 1;
}