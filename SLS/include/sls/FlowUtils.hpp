#pragma once
#include <opencv2/opencv.hpp>

namespace sls {

    // Dense local matching: for each pixel in source, search a local window in target
    cv::Mat computeDenseFlowLocal(
        const cv::Mat& sourceDesc,   // H x W x C, CV_32FC(C)
        const cv::Mat& targetDesc,   // H x W x C, CV_32FC(C)
        int windowRadius = 5         // e.g., 5 or 7
    );

    // Color visualization for flow (HSV mapping)
    cv::Mat flowToColor(const cv::Mat& flow); // flow: H x W x 2, CV_32FC2

    // Warp target image into source coordinates using flow
    cv::Mat warpImage(const cv::Mat& target, const cv::Mat& flow);

    // Result struct for evaluation
    struct FlowEvalResult {
        double meanError;
        double medianError;
        double percentBelow2px;
        double percentBelow5px;
        int    numSamples;
    };

    // Evaluate flow against a ground-truth 3x3 homography H (CV_64F)
    FlowEvalResult evaluateFlowAgainstHomography(
        const cv::Mat& flow,  // H x W x 2
        const cv::Mat& H,     // 3x3, CV_64F
        int step = 4          // evaluate every 'step' pixels
    );
}
