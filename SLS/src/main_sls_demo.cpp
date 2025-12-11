// Simple DSIFT + Scale-less SIFT (SLS-like) extraction, matching, and visualization.
// Uses OpenCV SIFT for dense descriptors and a simplified SLS pipeline.
// Author: Jordan Fleming
#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <algorithm>

#include "sls/sls_options.hpp"
#include "sls/sls_extractor.hpp"
#include "sls/dense_sift.hpp"

using namespace cv;
using std::cout;
using std::endl;

// Average descriptors across scales for each grid point
// dp: D x (numPoints * numSigma), column layout = s + i * numSigma
// Returns: D x numPoints
static Mat averageAcrossScales(const Mat& dp,
    int numPoints,
    int numSigma)
{
    CV_Assert(dp.cols == numPoints * numSigma);
    const int D = dp.rows;

    Mat desc(D, numPoints, CV_32F);
    desc.setTo(0);

    float invNumSigma = 1.0f / static_cast<float>(numSigma);

    for (int i = 0; i < numPoints; ++i) {
        Mat outCol = desc.col(i);
        outCol.setTo(0);

        for (int s = 0; s < numSigma; ++s) {
            int colIdx = s + i * numSigma;
            outCol += dp.col(colIdx);
        }
        outCol *= invNumSigma;
    }

    return desc;
}

// build a uniform grid of keypoints given grid size (s1 x s2)
// and image size (cols x rows).
static void buildGridKeypoints(int gridWidth,
    int gridHeight,
    int imgCols,
    int imgRows,
    std::vector<KeyPoint>& keypoints)
{
    keypoints.clear();
    keypoints.reserve(gridWidth * gridHeight);

    float cellW = static_cast<float>(imgCols) / static_cast<float>(gridWidth);
    float cellH = static_cast<float>(imgRows) / static_cast<float>(gridHeight);

    for (int row = 0; row < gridHeight; ++row) {
        for (int col = 0; col < gridWidth; ++col) {
            float x = (col + 0.5f) * cellW;
            float y = (row + 0.5f) * cellH;
            KeyPoint kp;
            kp.pt = Point2f(x, y);
            kp.size = std::min(cellW, cellH); // arbitrary but reasonable
            keypoints.push_back(kp);
        }
    }
}


// summarize match distances (mean, best, worst, count)
static void summarizeMatches(const std::vector<DMatch>& matches,
    const std::string& name)
{
    if (matches.empty()) {
        cout << name << ": no matches.\n";
        return;
    }

    double sum = 0.0;
    double best = matches.front().distance;
    double worst = matches.front().distance;

    for (const auto& m : matches) {
        sum += m.distance;
        if (m.distance < best)  best = m.distance;
        if (m.distance > worst) worst = m.distance;
    }

    double mean = sum / static_cast<double>(matches.size());

    cout << name << " distances:\n";
    cout << "  mean  = " << mean
        << ", best = " << best
        << ", worst = " << worst
        << ", count = " << matches.size() << "\n";
}


int main(int argc, char** argv)
{
    // Can pass in image paths or will default to source.jpg and target.jpg
    std::string srcPath = (argc > 1) ? argv[1] : "data/source.jpg";
    std::string tgtPath = (argc > 2) ? argv[2] : "data/target.jpg";

    // Load images and make grayscale
    Mat img1 = imread(srcPath, IMREAD_GRAYSCALE);
    Mat img2 = imread(tgtPath, IMREAD_GRAYSCALE);

    if (img1.empty() || img2.empty()) {
        std::cerr << "ERROR: could not load source or target image.\n";
        std::cerr << "  source: " << srcPath << "\n";
        std::cerr << "  target: " << tgtPath << "\n";
        return -1;
    }

    cout << "Loaded images:\n";
    cout << "  source: " << srcPath << "  (" << img1.cols << " x " << img1.rows << ")\n";
    cout << "  target: " << tgtPath << "  (" << img2.cols << " x " << img2.rows << ")\n";

    // Downsample for speed. Can change if needed
    double scaleFactor = 0.25;
    if (scaleFactor != 1.0) {
        resize(img1, img1, Size(), scaleFactor, scaleFactor, INTER_AREA);
        resize(img2, img2, Size(), scaleFactor, scaleFactor, INTER_AREA);
        cout << "After downsampling: "
            << img1.cols << " x " << img1.rows << endl;
    }


    // Compute dense SIFT (DSIFT) baseline (multi-scale descriptors)
    SLSOptions opts;

    opts.sigma.clear();
    int numSigma = 3;
    float sigmaStart = 1.0f;
    float sigmaEnd = 4.0f;
    float step = (sigmaEnd - sigmaStart) / (numSigma - 1);

    for (int i = 0; i < numSigma; ++i) {
        float s = sigmaStart + i * step;
        opts.sigma.push_back(s);
    }

    opts.gridSpacing = 8;
    opts.dimReduction = 32;
    opts.dimReductionCov = 20000;
    opts.subsDim = 6;

    TickMeter tm;

    cout << "\n[INFO] Computing DSIFT descriptors...\n";
    tm.start();
    DescriptorGrid ds1 = generateDescriptors(img1, opts);
    DescriptorGrid ds2 = generateDescriptors(img2, opts);
    tm.stop();
    cout << "[INFO] DSIFT done.\n";

    cout << "\nDSIFT baseline:\n";
    cout << "  image1: dpMat = " << ds1.dpMat.rows << " x " << ds1.dpMat.cols
        << ", numPoints = " << ds1.numPoints
        << ", grid = " << ds1.s1 << " x " << ds1.s2 << "\n";
    cout << "  image2: dpMat = " << ds2.dpMat.rows << " x " << ds2.dpMat.cols
        << ", numPoints = " << ds2.numPoints
        << ", grid = " << ds2.s1 << " x " << ds2.s2 << "\n";
    cout << "  DSIFT extraction time: " << tm.getTimeMilli() << " ms\n";

    // Compute Scale-less SIFT (SLS-like) descriptors
    bool usePaperParams = false;

    cout << "\nComputing SLS descriptors...\n";
    tm.reset();
    tm.start();
    SLSOutput slsOut = extractScalelessDescs(img1, img2, usePaperParams);
    tm.stop();
    cout << "SLS done.\n";

    cout << "\nSLS descriptors:\n";
    cout << "  desc1 size: " << slsOut.desc1.size << "\n";
    cout << "  desc2 size: " << slsOut.desc2.size << "\n";
    cout << "  PCA basis size: " << slsOut.pcaBasis.rows
        << " x " << slsOut.pcaBasis.cols << "\n";
    cout << "  SLS extraction time: " << tm.getTimeMilli() << " ms\n";

    // Build per-point DSIFT descriptors (average across scales)
    if (ds1.numPoints == 0 || ds2.numPoints == 0) {
        std::cerr << "ERROR: DSIFT produced no points; cannot match.\n";
        return 0;
    }

    Mat dsift1 = averageAcrossScales(ds1.dpMat, ds1.numPoints, numSigma);
    Mat dsift2 = averageAcrossScales(ds2.dpMat, ds2.numPoints, numSigma);

    Mat dsift1T = dsift1.t();
    Mat dsift2T = dsift2.t();

    // Build keypoints on a uniform grid (for both DSIFT and SLS)
    std::vector<KeyPoint> kp1, kp2;
    buildGridKeypoints(ds1.s1, ds1.s2, img1.cols, img1.rows, kp1);
    buildGridKeypoints(ds2.s1, ds2.s2, img2.cols, img2.rows, kp2);

    if ((int)kp1.size() != ds1.numPoints || (int)kp2.size() != ds2.numPoints) {
        std::cerr << "WARNING: keypoint count and numPoints differ.\n";
    }

    // Match DSIFT descriptors using BFMatcher (L2)
    cout << "\n[INFO] Matching DSIFT descriptors...\n";
    tm.reset();
    tm.start();

    BFMatcher matcherDSIFT(NORM_L2, true);
    std::vector<DMatch> matchesDSIFT;
    matcherDSIFT.match(dsift1T, dsift2T, matchesDSIFT);

    tm.stop();
    cout << "  DSIFT matches found: " << matchesDSIFT.size() << "\n";
    cout << "  DSIFT matching time: " << tm.getTimeMilli() << " ms\n";

    summarizeMatches(matchesDSIFT, "DSIFT");

    // Sort matches by distance and keep best K
    std::sort(matchesDSIFT.begin(), matchesDSIFT.end(),
        [](const DMatch& a, const DMatch& b) {
            return a.distance < b.distance;
        });

    const size_t maxMatchesToShow = 200;
    if (matchesDSIFT.size() > maxMatchesToShow) {
        matchesDSIFT.resize(maxMatchesToShow);
    }

    Mat matchesDSIFTImage;
    drawMatches(img1, kp1, img2, kp2, matchesDSIFT,
        matchesDSIFTImage,
        Scalar::all(-1), Scalar::all(-1),
        std::vector<char>(),
        DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

    imwrite("matches_dsift.jpg", matchesDSIFTImage);
    cout << "  Saved DSIFT matches to matches_dsift.jpg\n";

    namedWindow("DSIFT Matches", WINDOW_NORMAL);
    resizeWindow("DSIFT Matches",
        matchesDSIFTImage.cols / 2,
        matchesDSIFTImage.rows / 2);
    imshow("DSIFT Matches", matchesDSIFTImage);

    // Match SLS descriptors
    if (slsOut.desc1.empty() || slsOut.desc2.empty()) {
        std::cerr << "ERROR: SLS descriptors are empty; skipping SLS matching.\n";
    }
    else {
        cout << "\n[INFO] Matching SLS descriptors...\n";
        Mat sls1T = slsOut.desc1.t();
        Mat sls2T = slsOut.desc2.t();

        tm.reset();
        tm.start();

        BFMatcher matcherSLS(NORM_L2, true);
        std::vector<DMatch> matchesSLS;
        matcherSLS.match(sls1T, sls2T, matchesSLS);

        tm.stop();
        cout << "  SLS matches found: " << matchesSLS.size() << "\n";
        cout << "  SLS matching time: " << tm.getTimeMilli() << " ms\n";

        // Summarize SLS match quality before trimming
        summarizeMatches(matchesSLS, "SLS");

        std::sort(matchesSLS.begin(), matchesSLS.end(),
            [](const DMatch& a, const DMatch& b) {
                return a.distance < b.distance;
            });

        if (matchesSLS.size() > maxMatchesToShow) {
            matchesSLS.resize(maxMatchesToShow);
        }

        Mat matchesSLSImage;
        drawMatches(img1, kp1, img2, kp2, matchesSLS,
            matchesSLSImage,
            Scalar::all(-1), Scalar::all(-1),
            std::vector<char>(),
            DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

        imwrite("matches_sls.jpg", matchesSLSImage);
        cout << "  Saved SLS matches to matches_sls.jpg\n";

        namedWindow("SLS Matches", WINDOW_NORMAL);
        resizeWindow("SLS Matches",
            matchesSLSImage.cols / 2,
            matchesSLSImage.rows / 2);
        imshow("SLS Matches", matchesSLSImage);
    }

    cout << "\nDone. Press any key in the image window(s) to exit.\n";
    waitKey(0);
    return 0;
}
