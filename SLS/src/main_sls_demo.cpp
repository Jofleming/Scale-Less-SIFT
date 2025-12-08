#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>

#include "sls/sls_options.hpp"
#include "sls/sls_extractor.hpp"
#include "sls/dense_sift.hpp"

using namespace cv;
using std::cout;
using std::endl;

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

    // Compute dense SIFT (DSIFT) baseline
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

    // Compute Scale-less SIFT (SLS) descriptors
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

    cout << "\nDone.\n";
    return 0;
}
