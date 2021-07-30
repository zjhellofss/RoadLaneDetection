//
// Created by fss on 2021/7/29.
//

#ifndef ROADLANEDETECTION_LINES_H
#define ROADLANEDETECTION_LINES_H

struct ReducedLine
{
    ReducedLine(const cv::Point &point1, const cv::Point &point2) : point1_(point1), point2_(point2)
    {}

    double K() const
    {
        if (point2_.x != point1_.x)
            return (point2_.y - point1_.y) / (point2_.x - point1_.x);
        else
            return std::numeric_limits<double>::min();
    }

    double degree() const
    {
        double K = this->K();
        if (K == -1)
        {
            return std::numeric_limits<double>::max();
        }
        else
        {
            double d = std::abs(std::atan(K));
            d = d / (3.1415926536 / 180);
            return d;
        }
    }

    void setPoint1(const cv::Point2f &point1)
    {
        point1_ = point1;
    }

    void setPoint2(const cv::Point2f &point2)
    {
        point2_ = point2;
    }

    const cv::Point2f &getPoint1() const
    {
        return point1_;
    }

    const cv::Point2f &getPoint2() const
    {
        return point2_;
    }

    void appendInterPoint(const cv::Point2f &point)
    {
        this->inter_points.push_back(point);
    }


    cv::Point2f point1_;
    cv::Point2f point2_;
    std::vector<cv::Point2f> inter_points;
    bool is_inter = false;
};

inline bool intersection(const ReducedLine &l1, const ReducedLine &l2,
                  cv::Point2f &r)
{
    auto o1 = l1.point1_;
    auto p1 = l1.point2_;

    auto o2 = l2.point1_;
    auto p2 = l2.point2_;
    cv::Point2f x = o2 - o1;
    cv::Point2f d1 = p1 - o1;
    cv::Point2f d2 = p2 - o2;

    float cross = d1.x * d2.y - d1.y * d2.x;
    if (abs(cross) < 1e-8)
        return false;

    double t1 = (x.x * d2.y - x.y * d2.x) / cross;
    r = o1 + d1 * t1;
    return true;
}

inline cv::Point2d getLine(const cv::Point2f &point, double k, double x = -1, double y = -1)
{
    if (x == -1 && y == -1)
    {
        return cv::Point2d();
    }
    else
    {
        if (x != -1)
        {
            cv::Point2d point1;
            point1.x = x;
            point1.y = k * (point1.x - point.x) + point.y;
            return point1;
        }
        else if (y != -1)
        {
            cv::Point2d point1;
            point1.y = y;
            point1.x = (point1.y - point.y) / k + point.x;
            return point1;
        }
    }

}

struct vec2f
{
    float data[2]{};

    float operator[](int idx) const
    { return data[idx]; }

    vec2f(float x, float y)
    {
        this->data[0] = x;
        this->data[1] = y;
    }
};


#endif //ROADLANEDETECTION_LINES_H
