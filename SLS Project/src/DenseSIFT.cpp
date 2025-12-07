#include "DenseSIFT.h"
#include <cstring> // for std::memcpy
#include <iostream>

DenseSIFT::DenseSIFT(int step)
    : step_(step)
{
    // Requires OpenCV 4+ where SIFT is in main module
    sift_ = cv::SIFT::create();
}

cv::Mat DenseSIFT::compute(const cv::Mat& gray) {
    CV_Assert(!gray.empty());
    CV_Assert(gray.type() == CV_8UC1); // expect 8-bit grayscale

    int rows = gray.rows;
    int cols = gray.cols;

    std::vector<cv::KeyPoint> keypoints;
    keypoints.reserve((rows / step_ + 1) * (cols / step_ + 1));

    for (int y = 0; y < rows; y += step_) {
        for (int x = 0; x < cols; x += step_) {
            keypoints.emplace_back(cv::KeyPoint((float)x, (float)y, (float)step_));
        }
    }

    cv::Mat descriptors;        // N x 128
    sift_->compute(gray, keypoints, descriptors);

    if (descriptors.empty()) {
        std::cerr << "DenseSIFT: descriptors empty!" << std::endl;
        return cv::Mat();
    }

    int descDim = descriptors.cols; // should be 128
    CV_Assert(descDim > 0);

    // Dense map: H x W x descDim
    cv::Mat dense(rows, cols, CV_32FC(descDim), cv::Scalar(0));

    for (int i = 0; i < (int)keypoints.size(); ++i) {
        int x = cvRound(keypoints[i].pt.x);
        int y = cvRound(keypoints[i].pt.y);
        if (x < 0 || x >= cols || y < 0 || y >= rows)
            continue;

        float* dst = dense.ptr<float>(y, x);
        const float* src = descriptors.ptr<float>(i);
        std::memcpy(dst, src, sizeof(float) * descDim);
    }

    return dense;
}
