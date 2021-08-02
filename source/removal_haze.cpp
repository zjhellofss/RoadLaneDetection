#include <opencv2/opencv.hpp>
#include <cmath>
#include <numeric>

using namespace cv;


void MinFilter(Mat &source, Mat &output, int r)
{
    Mat input;
    source.copyTo(input);

    output.create(source.rows, source.cols, CV_8U);
    for (int i = 0; i <= (input.rows - 1) / r; i++)
    {
        for (int j = 0; j <= (input.cols - 1) / r; j++)
        {
            int w = r;
            int h = r;
            if (i * r + h > input.rows)
            {
                h = input.rows - i * r;
            }
            if (j * r + w > input.cols)
            {
                w = input.cols - j * r;
            }

            Mat ROI = input(Rect(j * r, i * r, w, h));

            double mmin;
            minMaxLoc(ROI, &mmin, 0);

            Mat desROI = output(Rect(j * r, i * r, w, h));
            desROI.setTo(uchar(mmin));
        }
    }
}

void DarkChannel(Mat &source, Mat &output, int r)
{
    Mat input;
    input.create(source.rows, source.cols, CV_8U);

    for (int i = 0; i < source.rows; i++)
    {
        uchar *sourcedata = source.ptr<uchar>(i);
        uchar *indata = input.ptr<uchar>(i);
        for (int j = 0; j < source.cols * source.channels(); j += 3)
        {
            uchar mmin;
            mmin = min(sourcedata[j], sourcedata[j + 1]);
            mmin = min(mmin, sourcedata[j + 2]);

            indata[j / 3] = mmin;
        }
    }

    MinFilter(input, output, r);
}

static void makeDepth32f(Mat &source, Mat &output)
{
    if (source.depth() != CV_32F)
        source.convertTo(output, CV_32F);
    else
        output = source;
}

static void mynorm(Mat &source, Mat &output)
{
    for (int i = 0; i < source.rows; i++)
    {
        float *indata = source.ptr<float>(i);
        float *outdata = output.ptr<float>(i);
        for (int j = 0; j < source.cols * source.channels(); j++)
        {
            outdata[j] = indata[j] / 255.0;
        }
    }
}

void GuideFilter(Mat &source, Mat &guided_image, Mat &output, int radius, double epsilon)
{
    CV_Assert(radius >= 2 && epsilon > 0);
    CV_Assert(source.data != NULL && source.channels() == 1);
    CV_Assert(guided_image.channels() == 1);
    CV_Assert(source.rows == guided_image.rows && source.cols == guided_image.cols);

    Mat guided;
    if (guided_image.data == source.data)
    {
        //make a copy
        guided_image.copyTo(guided);
    }
    else
    {
        guided = guided_image;
    }

    //将输入扩展为32位浮点型，以便以后做乘法
    Mat source_32f, guided_32f;
    makeDepth32f(source, source_32f);
    mynorm(source_32f, source_32f);
    makeDepth32f(guided, guided_32f);
    mynorm(guided_32f, guided_32f);

    //计算I*p和I*I
    Mat mat_Ip, mat_I2;
    multiply(guided_32f, source_32f, mat_Ip);
    multiply(guided_32f, guided_32f, mat_I2);

    //计算各种均值
    Mat mean_p, mean_I, mean_Ip, mean_I2;
    Size win_size(2 * radius + 1, 2 * radius + 1);
    boxFilter(source_32f, mean_p, CV_32F, win_size);
    boxFilter(guided_32f, mean_I, CV_32F, win_size);
    boxFilter(mat_Ip, mean_Ip, CV_32F, win_size);
    boxFilter(mat_I2, mean_I2, CV_32F, win_size);

    //计算Ip的协方差和I的方差
    Mat cov_Ip = mean_Ip - mean_I.mul(mean_p);
    Mat var_I = mean_I2 - mean_I.mul(mean_I);
    var_I += epsilon;

    //求a和b
    Mat a, b;
    divide(cov_Ip, var_I, a);
    b = mean_p - a.mul(mean_I);

    //对包含像素i的所有a、b做平均
    Mat mean_a, mean_b;
    boxFilter(a, mean_a, CV_32F, win_size);
    boxFilter(b, mean_b, CV_32F, win_size);

    //计算输出 (depth == CV_32F)
    Mat tempoutput = mean_a.mul(guided_32f) + mean_b;

    output.create(source.rows, source.cols, CV_8U);

    for (int i = 0; i < source.rows; i++)
    {
        float *data = tempoutput.ptr<float>(i);
        uchar *outdata = output.ptr<uchar>(i);
        for (int j = 0; j < source.cols; j++)
        {
            outdata[j] = saturate_cast<uchar>(data[j] * 255);
        }
    }
}

int getMax(Mat src) {
    int row = src.rows;
    int col = src.cols;
    int temp = 0;
    for (int i = 0; i < row; i++) {
        for (int j = 0; j < col; j++) {
            temp = max((int)src.at<uchar>(i, j), temp);
        }
        if (temp == 255) return temp;
    }
    return temp;
}

Mat dehaze(Mat src) {
    double eps;
    int row = src.rows;
    int col = src.cols;
    Mat M = Mat::zeros(row, col, CV_8UC1);
    Mat M_max = Mat::zeros(row, col, CV_8UC1);
    Mat M_ave = Mat::zeros(row, col, CV_8UC1);
    Mat L = Mat::zeros(row, col, CV_8UC1);
    Mat dst = Mat::zeros(row, col, CV_8UC3);
    double m_av, A;
    //get M
    double sum = 0;
    for (int i = 0; i < row; i++) {
        for (int j = 0; j < col; j++) {
            uchar r, g, b, temp1, temp2;
            b = src.at<Vec3b>(i, j)[0];
            g = src.at<Vec3b>(i, j)[1];
            r = src.at<Vec3b>(i, j)[2];
            temp1 = min(min(r, g), b);
            temp2 = max(max(r, g), b);
            M.at<uchar>(i, j) = temp1;
            M_max.at<uchar>(i, j) = temp2;
            sum += temp1;
        }
    }
    m_av = sum / (row * col * 255);
    eps = 0.85 / m_av;
    boxFilter(M, M_ave, CV_8UC1, Size(51, 51));
    double delta = min(0.9, eps*m_av);
    for (int i = 0; i < row; i++) {
        for (int j = 0; j < col; j++) {
            L.at<uchar>(i, j) = min((int)(delta * M_ave.at<uchar>(i, j)), (int)M.at<uchar>(i, j));
        }
    }
    A = (getMax(M_max) + getMax(M_ave)) * 0.5;
    for (int i = 0; i < row; i++) {
        for (int j = 0; j < col; j++) {
            int temp = L.at<uchar>(i, j);
            for (int k = 0; k < 3; k++) {
                int val = A * (src.at<Vec3b>(i, j)[k] - temp) / (A - temp);
                if (val > 255) val = 255;
                if (val < 0) val = 0;
                dst.at<Vec3b>(i, j)[k] = val;
            }
        }
    }
    return dst;
}

cv::Mat HazeRemoval(Mat &source, Mat &output, int minr , int maxA , double w , int guider ,
                 double guideeps , int L )
{
    Mat input;
    source.copyTo(input);
    Mat dark;
    DarkChannel(input, dark, minr * 2);
    int hash[256];
    memset(hash, 0, sizeof(hash));
    for (int i = 0; i < dark.rows; i++)
    {
        uchar *data = dark.ptr<uchar>(i);
        for (int j = 0; j < dark.cols; j++)
        {
            hash[data[j]]++;
        }
    }

    int num = dark.rows * dark.cols / 1000.0;
    int count = 0;
    uchar thres;
    for (int i = 0; i < 256; i++)
    {
        count += hash[255 - i];
        if (count >= num)
        {
            thres = 255 - i;
            break;
        }
    }
    num = count;
    double b_max = 0, B;
    double g_max = 0, G;
    double r_max = 0, R;
    for (int i = 0; i < dark.rows; i++)
    {
        uchar *data = dark.ptr<uchar>(i);
        uchar *indata = input.ptr<uchar>(i);
        for (int j = 0; j < dark.cols; j++)
        {
            if (data[j] >= thres)
            {
                B = indata[3 * j];
                G = indata[3 * j + 1];
                R = indata[3 * j + 2];
                b_max += B;
                g_max += G;
                r_max += R;
            }
        }
    }
    b_max /= num;
    g_max /= num;
    r_max /= num;
    uchar MMAX = maxA;
    if (b_max > MMAX) b_max = MMAX;
    if (g_max > MMAX) g_max = MMAX;
    if (r_max > MMAX) r_max = MMAX;


    Mat img_t;
    img_t.create(dark.rows, dark.cols, CV_8U);
    Mat temp;
    temp.create(dark.rows, dark.cols, CV_8UC3);
    double b_temp = b_max / 255;
    double g_temp = g_max / 255;
    double r_temp = r_max / 255;
    for (int i = 0; i < dark.rows; i++)
    {
        uchar *data = input.ptr<uchar>(i);
        uchar *tdata = temp.ptr<uchar>(i);
        for (int j = 0; j < dark.cols * 3; j += 3)
        {
            tdata[j] = saturate_cast<uchar>(data[j] / b_temp);
            tdata[j + 1] = saturate_cast<uchar>(data[j] / g_temp);
            tdata[j + 2] = saturate_cast<uchar>(data[j] / r_temp);
        }
    }

    Mat gray;
    cvtColor(temp, gray, cv::COLOR_BGR2GRAY);
    DarkChannel(temp, temp, minr * 2);
    for (int i = 0; i < dark.rows; i++)
    {
        uchar *darkdata = temp.ptr<uchar>(i);
        uchar *tdata = img_t.ptr<uchar>(i);
        for (int j = 0; j < dark.cols; j++)
        {
            tdata[j] = 255 - w * darkdata[j];
        }
    }
    GuideFilter(img_t, gray, img_t, guider, guideeps);

    //还原图像
    output.create(input.rows, input.cols, CV_8UC3);
    for (int i = 0; i < input.rows; i++)
    {
        uchar *tdata = img_t.ptr<uchar>(i);
        uchar *indata = input.ptr<uchar>(i);
        uchar *outdata = output.ptr<uchar>(i);
        for (int j = 0; j < input.cols; j++)
        {
            uchar b = indata[3 * j];
            uchar g = indata[3 * j + 1];
            uchar r = indata[3 * j + 2];
            double t = tdata[j];
            t /= 255;
            if (t < 0.1) t = 0.1;

            outdata[3 * j] = saturate_cast<uchar>((b - b_max) / t + b_max + L);
            outdata[3 * j + 1] = saturate_cast<uchar>((g - g_max) / t + g_max + L);
            outdata[3 * j + 2] = saturate_cast<uchar>((r - r_max) / t + r_max + L);
        }
    }
    return output;
}