#pragma once
#include <opencv2/opencv.hpp>

namespace sls {
    cv::Mat computeDenseFlowLocal(
        const cv::Mat& sourceDesc,
        const cv::Mat& targetDesc,
        int windowRadius = 5
    );

    cv::Mat flowToColor(const cv::Mat& flow);
    cv::Mat warpImage(const cv::Mat& target, const cv::Mat& flow);

    // Result struct for evaluation
    struct FlowEvalResult {
        double meanError;
        double medianError;
        double percentBelow2px;
        double percentBelow5px;
        int    numSamples;
    };

    FlowEvalResult evaluateFlowAgainstHomography(
        const cv::Mat& flow,
        const cv::Mat& H,
        int step = 4
    );
}
