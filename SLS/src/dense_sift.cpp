#include "sls/dense_sift.hpp"
#include "sls/sls_options.hpp"

#include <opencv2/opencv.hpp>
#include <cmath>
#include <iostream>

using namespace cv;

// Generate dense SIFT descriptors on a regular grid.
DescriptorGrid generateDescriptors(const Mat& grayImage, const SLSOptions& opts) {
    DescriptorGrid out;

    if (grayImage.empty()) {
        std::cerr << "generateDescriptors: input image is empty\n";
        return out;
    }

    // Padding size similar to MATLAB code
    const float NBP = 4.0f;
    const float SBP = 3.0f * opts.sigma.back();
    const float w = SBP * (NBP + 1.0f);
    const int   padSize = static_cast<int>(std::ceil(w / 2.0f));

    Mat paddedFloat;
    copyMakeBorder(grayImage, paddedFloat,
        padSize, padSize, padSize, padSize,
        BORDER_REFLECT_101);

    Mat padded;
    if (paddedFloat.depth() != CV_8U) {
        paddedFloat.convertTo(padded, CV_8U, 255.0);
    }
    else {
        padded = paddedFloat;
    }

    const int rows = padded.rows;
    const int cols = padded.cols;
    const int gridSpacing = opts.gridSpacing;

    // Build grid of coordinates inside padded region
    std::vector<Point2f> coords;
    coords.reserve(((rows - 2 * padSize + gridSpacing - 1) / gridSpacing) *
        ((cols - 2 * padSize + gridSpacing - 1) / gridSpacing));

    for (int y = padSize; y < rows - padSize; y += gridSpacing) {
        for (int x = padSize; x < cols - padSize; x += gridSpacing) {
            coords.emplace_back(static_cast<float>(x), static_cast<float>(y));
        }
    }

    const int numPoints = static_cast<int>(coords.size());
    const int numSigma = static_cast<int>(opts.sigma.size());
    const int D = 128;

    out.numPoints = numPoints;
    out.s1 = (cols - 2 * padSize + gridSpacing - 1) / gridSpacing;
    out.s2 = (rows - 2 * padSize + gridSpacing - 1) / gridSpacing;

    out.dpMat = Mat::zeros(D, numPoints * numSigma, CV_32F);

    // SIFT extractor
    Ptr<SIFT> sift = SIFT::create();

    // For each scale, build keypoints and compute descriptors
    for (int si = 0; si < numSigma; ++si) {
        float sigma = opts.sigma[si];
        float patchSize = 3.0f * sigma * (NBP + 1.0f);

        std::vector<KeyPoint> keypoints;
        keypoints.reserve(numPoints);

        for (int i = 0; i < numPoints; ++i) {
            KeyPoint kp;
            kp.pt = coords[i];
            kp.size = patchSize;
            kp.angle = 0.0f;
            keypoints.push_back(kp);
        }

        Mat desc;
        sift->compute(padded, keypoints, desc);

        if (desc.rows != numPoints || desc.cols != D) {
            std::cerr << "generateDescriptors: unexpected SIFT size ("
                << desc.rows << "x" << desc.cols << "), expected "
                << numPoints << "x" << D << "\n";
        }

        // Copy descriptors into dpMat.
        for (int i = 0; i < numPoints; ++i) {
            Mat srcRow = desc.row(i);
            Mat dstCol = out.dpMat.col(si + i * numSigma);
            srcRow.reshape(1, D).copyTo(dstCol);
        }
    }

    return out;
}
