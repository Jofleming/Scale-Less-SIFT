#include "Visualizer.h"

cv::Mat Visualizer::warpImage(const cv::Mat& src, const cv::Mat& flow) {
    CV_Assert(!src.empty());
    CV_Assert(flow.type() == CV_32FC2);
    CV_Assert(src.rows == flow.rows && src.cols == flow.cols);

    cv::Mat mapX(src.size(), CV_32FC1);
    cv::Mat mapY(src.size(), CV_32FC1);

    for (int y = 0; y < src.rows; ++y) {
        for (int x = 0; x < src.cols; ++x) {
            cv::Vec2f f = flow.at<cv::Vec2f>(y, x);
            mapX.at<float>(y, x) = static_cast<float>(x) + f[0];
            mapY.at<float>(y, x) = static_cast<float>(y) + f[1];
        }
    }

    cv::Mat warped;
    cv::remap(src, warped, mapX, mapY, cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(0));
    return warped;
}
