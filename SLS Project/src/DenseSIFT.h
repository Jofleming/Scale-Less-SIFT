#pragma once
#include <opencv2/opencv.hpp>

class DenseSIFT {
public:
    DenseSIFT(int step = 4);

    // Input: grayscale 8-bit image
    // Output: H x W x 128 descriptors (CV_32FC(128))
    cv::Mat compute(const cv::Mat& gray);

private:
    int step_;
    cv::Ptr<cv::SIFT> sift_;
};
