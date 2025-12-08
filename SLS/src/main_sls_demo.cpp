#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>

#include "sls/sls_options.hpp"
#include "sls/sls_extractor.hpp"
#include "sls/dense_sift.hpp"
#include "sls/FlowUtils.hpp"

using std::cout;
using std::endl;

static cv::Mat descriptorGridToImage(const DescriptorGrid& grid, int dim)
{
    cv::Mat descImg(grid.s2, grid.s1, CV_32FC(dim));

    for (int p = 0; p < grid.numPoints; ++p) {
        int row = p / grid.s1;
        int col = p % grid.s1;

        const float* src = grid.dpMat.col(p).ptr<float>(0);
        float* dst = descImg.ptr<float>(row, col);

        for (int d = 0; d < dim; ++d)
            dst[d] = src[d];
    }
    return descImg;
}

int main(int argc, char** argv)
{
    std::string srcPath = (argc > 1) ? argv[1] : "data/source.jpg";
    std::string tgtPath = (argc > 2) ? argv[2] : "data/target.jpg";

    cv::Mat I1 = cv::imread(srcPath);
    cv::Mat I2 = cv::imread(tgtPath);

    if (I1.empty() || I2.empty()) {
        std::cerr << "ERROR: could not load source/target images from data/ folder.\n";
        return -1;
    }

    cout << "Loaded images:\n";
    cout << "  source: " << srcPath << "  (" << I1.cols << " x " << I1.rows << ")\n";
    cout << "  target: " << tgtPath << "  (" << I2.cols << " x " << I2.rows << ")\n";

    cv::Mat I1d, I2d;
    cv::resize(I1, I1d, cv::Size(), 0.25, 0.25);
    cv::resize(I2, I2d, cv::Size(), 0.25, 0.25);
    cout << "After downsampling: " << I1d.cols << " x " << I1d.rows << "\n\n";

    cout << "[INFO] Computing DSIFT baseline...\n";

    SLSOptions dsiftOpts;
    dsiftOpts.gridSpacing = 4;
    dsiftOpts.sigma = { 2.0f };
    dsiftOpts.dimReduction = 0;
    dsiftOpts.dimReductionCov = 0;
    dsiftOpts.subsDim = 0;

    cv::Mat g1, g2;
    cv::cvtColor(I1d, g1, cv::COLOR_BGR2GRAY);
    cv::cvtColor(I2d, g2, cv::COLOR_BGR2GRAY);
    g1.convertTo(g1, CV_8U);
    g2.convertTo(g2, CV_8U);

    DescriptorGrid ds1 = generateDescriptors(g1, dsiftOpts);
    DescriptorGrid ds2 = generateDescriptors(g2, dsiftOpts);

    cout << "DSIFT baseline:\n";
    cout << "  image1: dpMat = " << ds1.dpMat.rows << " x " << ds1.dpMat.cols
        << ", numPoints = " << ds1.numPoints
        << ", grid = " << ds1.s1 << " x " << ds1.s2 << "\n";
    cout << "  image2: dpMat = " << ds2.dpMat.rows << " x " << ds2.dpMat.cols
        << ", numPoints = " << ds2.numPoints
        << ", grid = " << ds2.s1 << " x " << ds2.s2 << "\n";

    cv::Mat dsDesc1 = descriptorGridToImage(ds1, 128);
    cv::Mat dsDesc2 = descriptorGridToImage(ds2, 128);

    int windowRadius = 5;
    cv::Mat flowDSIFT = sls::computeDenseFlowLocal(dsDesc1, dsDesc2, windowRadius);

    cv::Mat I1_grid;
    cv::resize(I1d, I1_grid, cv::Size(ds1.s1, ds1.s2));

    cv::Mat warpDSIFT_small = sls::warpImage(I1_grid, flowDSIFT);
    cv::Mat warpDSIFT;
    cv::resize(warpDSIFT_small, warpDSIFT, I1d.size());

    cv::imwrite("dsift_warp.png", warpDSIFT);

    cv::Mat flowColorDSIFT = sls::flowToColor(flowDSIFT);
    cv::imwrite("flow_dsift_color.png", flowColorDSIFT);

    cout << "\n[INFO] Computing SLS descriptors...\n";

    bool usePaperParams = true;
    SLSOutput slsOut = extractScalelessDescs(I1d, I2d, usePaperParams);

    cout << "SLS descriptors:\n";
    cout << "  desc1 size: " << slsOut.desc1.rows << " x "
        << slsOut.desc1.cols << " x " << slsOut.desc1.channels() << "\n";
    cout << "  desc2 size: " << slsOut.desc2.rows << " x "
        << slsOut.desc2.cols << " x " << slsOut.desc2.channels() << "\n";

    cv::Mat flowSLS = sls::computeDenseFlowLocal(slsOut.desc1, slsOut.desc2, windowRadius);

    cv::Mat warpSLS_small = sls::warpImage(I1_grid, flowSLS);
    cv::Mat warpSLS;
    cv::resize(warpSLS_small, warpSLS, I1d.size());

    cv::imwrite("sls_warp.png", warpSLS);

    cv::Mat flowColorSLS = sls::flowToColor(flowSLS);
    cv::imwrite("flow_sls_color.png", flowColorSLS);

    cv::Mat slsVis = slsOut.desc1.reshape(1, slsOut.desc1.rows * slsOut.desc1.cols);
    double minVal, maxVal;
    cv::minMaxLoc(slsVis, &minVal, &maxVal);

    cv::Mat slsNorm;
    cv::normalize(slsVis, slsNorm, 0, 255, cv::NORM_MINMAX, CV_8U);
    cv::imwrite("sls_descriptor_visual.png", slsNorm);

    cout << "\n[INFO] Visualization complete. Files saved:\n"
        << "  dsift_warp.png\n"
        << "  flow_dsift_color.png\n"
        << "  sls_warp.png\n"
        << "  flow_sls_color.png\n"
        << "  sls_descriptor_visual.png\n";

    return 0;
}
