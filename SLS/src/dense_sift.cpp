#include "sls/dense_sift.hpp"
#include <opencv2/opencv.hpp>
#include <cmath>
#include <iostream>

DescriptorGrid generateDescriptors(const cv::Mat& grayImage, const SLSOptions& opts) {
    DescriptorGrid out;

    if (grayImage.empty()) {
        std::cerr << "generateDescriptors: input image is empty\n";
        return out;
    }

    // Pad image
    float NBP = 4.0f;
    float SBP = 3.0f * opts.sigma.back();
    float w = SBP * (NBP + 1.0f);
    int padSize = static_cast<int>(std::ceil(w / 2.0f));

    // grayImage may be CV_32F in [0,1] or CV_8U; we will:
    //   - pad it
    //   - convert the padded version to CV_8U for SIFT
    cv::Mat paddedFloat;
    cv::copyMakeBorder(grayImage, paddedFloat,
        padSize, padSize, padSize, padSize,
        cv::BORDER_REFLECT_101);

    cv::Mat padded;
    if (paddedFloat.depth() != CV_8U) {
        paddedFloat.convertTo(padded, CV_8U, 255.0);
    }
    else {
        padded = paddedFloat;
    }

    int rows = padded.rows;
    int cols = padded.cols;
    int gridSpacing = opts.gridSpacing;

    // Build grid of coordinates inside padded region
    std::vector<cv::Point2f> coords;
    coords.reserve(((rows - 2 * padSize + gridSpacing - 1) / gridSpacing) *
        ((cols - 2 * padSize + gridSpacing - 1) / gridSpacing));

    for (int y = padSize; y < rows - padSize; y += gridSpacing) {
        for (int x = padSize; x < cols - padSize; x += gridSpacing) {
            coords.emplace_back(static_cast<float>(x), static_cast<float>(y));
        }
    }

    int numPoints = static_cast<int>(coords.size());
    int numSigma = static_cast<int>(opts.sigma.size());
    out.numPoints = numPoints;

    // Approximate grid dimensions
    out.s1 = (cols - 2 * padSize + gridSpacing - 1) / gridSpacing;
    out.s2 = (rows - 2 * padSize + gridSpacing - 1) / gridSpacing;

    int D = 128;  // SIFT descriptor length

    // dpMat: D x (numPoints * numSigma), float
    out.dpMat = cv::Mat::zeros(D, numPoints * numSigma, CV_32F);

    // SIFT extractor
    cv::Ptr<cv::SIFT> sift = cv::SIFT::create();

    // For each sigma, set keypoint size and compute descriptors
    for (int si = 0; si < numSigma; ++si) {
        float sigma = opts.sigma[si];
        float patchSize = 3.0f * sigma * (NBP + 1.0f);

        std::vector<cv::KeyPoint> keypoints;
        keypoints.reserve(numPoints);

        for (int i = 0; i < numPoints; ++i) {
            cv::KeyPoint kp;
            kp.pt = coords[i];
            kp.size = patchSize;
            kp.angle = 0.0f;
            keypoints.push_back(kp);
        }

        cv::Mat desc;
        sift->compute(padded, keypoints, desc)

        if (desc.rows != numPoints || desc.cols != D) {
            std::cerr << "Unexpected SIFT descriptor layout: "
                << "rows=" << desc.rows << " (expected " << numPoints << "), "
                << "cols=" << desc.cols << " (expected " << D << ")\n";
        }

        // Copy descriptors into dpMat so that for point i and scale si:
        //   column index = si + i * numSigma
        for (int i = 0; i < numPoints; ++i) {
            cv::Mat srcRow = desc.row(i);        
            cv::Mat dstCol = out.dpMat.col(si + i * numSigma);
            srcRow.reshape(1, D).copyTo(dstCol);
        }
    }

    return out;
}
