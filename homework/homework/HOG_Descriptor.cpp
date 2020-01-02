

#include "HOG_Descriptor.h"
#include <iostream>

void HOG_Descriptor::initDetector() {
    // Initialize hog detector
	hog_detector = cv::HOGDescriptor(win_size, block_size, block_step, cell_size, nbins);

    //Fill code here

    is_init = true;

}

void HOG_Descriptor::detectHOGDescriptor(cv::Mat& im, std::vector<float>& feat, std::vector<float>& labels, int num_classes, cv::Size sz, bool show) {
    if (!is_init) {
        initDetector();
    }

    // Fill code here

    /* pad your image
     * resize your image
     * use the built in function "compute" to get the HOG descriptors
     */
	int cellSide = cell_size.height;

	for (int classe = 0; classe < num_classes; classe++) {
		fs::path p(absolutePath + std::to_string(classe));
		for (auto i = fs::directory_iterator(p); i != fs::directory_iterator(); i++)
		{
			if (!fs::is_directory(i->path())) //we eliminate directories
			{
				cout << absolutePath + std::to_string(classe) + "/" + i->path().filename().string() << endl;
				Mat img = imread(absolutePath + std::to_string(classe) + "/" + i->path().filename().string());

				// Resizing the picture
				Mat resizedImg;
				paddingToNextShape(img, resizedImg, cellSide, SQUARE_RESIZED, newSize);

				if (training == 1) {
					add_augmented_picture(descr, labels, features, resizedImg, classe);
				}
				else {
					(*labels).push_back(classe);
					std::vector<float> descriptors = std::vector<float>();
					descr.compute(resizedImg, descriptors);
					Mat descriptorsToAdd = ((Mat)descriptors).reshape(1, 1);
					descriptorsToAdd.convertTo(descriptorsToAdd, CV_32F);
					(*features).push_back(descriptorsToAdd);
				}

			}
			else
				continue;
		}
	}

	// Resizing the picture
	Mat resizedImg;
	paddingToNextShape(im, resizedImg, cellSide, SQUARE_RESIZED, sz);

	std::vector<float> descriptors = std::vector<float>();
	hog_detector.compute(resizedImg, descriptors);
	feat.insert(feat.end(), descriptors.begin(), descriptors.end());
	(*labels).push_back(classe);
	
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

