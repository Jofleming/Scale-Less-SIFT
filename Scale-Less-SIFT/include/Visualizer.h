#pragma once
#include <opencv2/opencv.hpp>

class Visualizer {
public:
    // Warp src using dense flow (dx,dy per pixel)
    static cv::Mat warpImage(const cv::Mat& src, const cv::Mat& flow);
};
