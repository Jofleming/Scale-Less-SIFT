#include "ImageLoader.h"
#include <iostream>

cv::Mat ImageLoader::loadGray(const std::string& path) {
    cv::Mat img = cv::imread(path, cv::IMREAD_GRAYSCALE);
    if (img.empty()) {
        std::cerr << "Failed to load image: " << path << std::endl;
    }
    return img;
}

cv::Mat ImageLoader::loadGrayFloat(const std::string& path) {
    cv::Mat img = loadGray(path);
    if (!img.empty()) {
        img.convertTo(img, CV_32F, 1.0 / 255.0);
    }
    return img;
}
