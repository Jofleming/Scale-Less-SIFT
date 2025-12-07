#include <opencv2/opencv.hpp>
#include <iostream>

#include "ImageLoader.h"
#include "MultiScaleSIFTExtractor.h"
#include "PCAHelper.h"
#include "SLSBuilder.h"
#include "DenseMatcher.h"
#include "Visualizer.h"

int main() {
    try {
        std::string imgPath1 = "pippy.jpg";   // put file next to main.cpp
        std::string imgPath2 = "pippy.jpg";   // for now use same image

        // 1) Load images (8-bit gray)
        cv::Mat img1 = ImageLoader::loadGray(imgPath1);
        cv::Mat img2 = ImageLoader::loadGray(imgPath2);

        if (img1.empty() || img2.empty()) {
            std::cerr << "Could not load input images.\n";
            return -1;
        }

        // Resize second image to simulate scale change (for testing)
        cv::resize(img2, img2, cv::Size(), 1.5, 1.5, cv::INTER_LINEAR);
        cv::resize(img2, img2, img1.size()); // bring back to same size

        // 2) Multi-scale dense SIFT
        MultiScaleSIFTExtractor msExtractor(5, 1.0f, 6.0f, 4);
        auto maps1 = msExtractor.compute(img1);
        auto maps2 = msExtractor.compute(img2);

        // 3) PCA on SIFT descriptors
        PCAHelper pca;
        pca.collectSamples(maps1, 20000);
        pca.collectSamples(maps2, 20000);
        pca.computePCA(32); // project to 32D

        auto maps1_pca = pca.projectMaps(maps1);
        auto maps2_pca = pca.projectMaps(maps2);

        // 4) SLS descriptors
        SLSBuilder slsBuilder(8); // subspace dim
        cv::Mat sls1 = slsBuilder.buildSLSMap(maps1_pca);
        cv::Mat sls2 = slsBuilder.buildSLSMap(maps2_pca);

        // 5) Dense matching using SLS
        DenseMatcher matcher(3); // search radius
        cv::Mat flow = matcher.match(sls1, sls2);

        // 6) Warp and visualize
        cv::Mat warped = Visualizer::warpImage(img1, flow);

        cv::imshow("Image 1", img1);
        cv::imshow("Image 2", img2);
        cv::imshow("Warped Image 1 -> 2 (SLS)", warped);
        cv::waitKey(0);

    }
    catch (const std::exception& ex) {
        std::cerr << "Exception: " << ex.what() << std::endl;
        return -1;
    }

    return 0;
}
