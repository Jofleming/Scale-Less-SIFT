#pragma once
#include <opencv2/opencv.hpp>
#include <string>

class ImageLoader {
public:
    // Load grayscale 8-bit image
    static cv::Mat loadGray(const std::string& path);

    // Load grayscale float image in [0,1]
    static cv::Mat loadGrayFloat(const std::string& path);
};
