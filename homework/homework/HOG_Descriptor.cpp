

#include "HOG_Descriptor.h"
#include <iostream>

void HOG_Descriptor::initDetector() {
    // Initialize hog detector
	hog_detector = cv::HOGDescriptor(win_size, block_size, block_step, cell_size, nbins);

    //Fill code here

    is_init = true;

}

void HOG_Descriptor::detectHOGDescriptor(cv::Mat& im, std::vector<float>& feat, cv::Size sz, bool show) {
    if (!is_init) {
        initDetector();
    }

    // Fill code here

    /* pad your image
     * resize your image
     * use the built in function "compute" to get the HOG descriptors
     */
	int cellSide = cell_size.height;


	// Resizing the picture
	Mat resizedImg;
	paddingToNextShape(im, resizedImg, cellSide, SQUARE_RESIZED, sz);

	std::vector<float> descriptors = std::vector<float>();
	hog_detector.compute(resizedImg, descriptors);
	feat.insert(feat.end(), descriptors.begin(), descriptors.end());
	
	if (show == true) {
		this->visualizeHOG(resizedImg, feat, hog_detector, 10);
	}


	//visualize_matrix(features, "Features", 10);

}

//returns instance of cv::HOGDescriptor
cv::HOGDescriptor& HOG_Descriptor::getHog_detector() {
    // Fill code here
	return hog_detector;
}

