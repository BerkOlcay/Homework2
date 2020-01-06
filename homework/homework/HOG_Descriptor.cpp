

#include "HOG_Descriptor.h"
#include <iostream>

cv::Size HOG_Descriptor::win_size = cv::Size(64, 64);
cv::Size HOG_Descriptor::block_size = cv::Size(16, 16);
cv::Size HOG_Descriptor::block_step = cv::Size(8, 8);
cv::Size HOG_Descriptor::cell_size = cv::Size(8, 8);
int HOG_Descriptor::nbins = 9;

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

void HOG_Descriptor::create_dataset(int num_classes, std::string absolutePath, Mat* labels, Mat* features, int training) {

	if (!is_init) {
		initDetector();
	}

	// Resizing the picture so that the whole dataset has the same size


	int cellSide = HOG_Descriptor::cell_size.height;
	Size cellSize = HOG_Descriptor::cell_size;
	Size blockSize = HOG_Descriptor::block_size;
	Size blockStride = HOG_Descriptor::block_step;
	Size winSize = HOG_Descriptor::win_size;
	int nbins = HOG_Descriptor::nbins;

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
				paddingToNextShape(img, resizedImg, cellSide, SQUARE_RESIZED, winSize);

				if (training == 1) {
					add_augmented_picture(hog_detector, labels, features, resizedImg, classe);
				}
				else {
					(*labels).push_back(classe);
					std::vector<float> descriptors = std::vector<float>();
					hog_detector.compute(resizedImg, descriptors);
					Mat descriptorsToAdd = ((Mat)descriptors).reshape(1, 1);
					descriptorsToAdd.convertTo(descriptorsToAdd, CV_32F);
					(*features).push_back(descriptorsToAdd);
				}

			}
			else
				continue;
		}
	}
	(*labels).convertTo(*labels, CV_32S);
	(*features).convertTo(*features, CV_32F);
}

void HOG_Descriptor::add_augmented_picture(HOGDescriptor descr, Mat* labels, Mat* features, Mat resizedImg, int classe) {

	int Nimages = 14;

	// saving Label of the current picture
	for (int i = 0; i < Nimages; i++) {
		(*labels).push_back(classe);
	}

	// Normal image : Compute HOG and add to features
	std::vector<float> descriptors = std::vector<float>();
	descr.compute(resizedImg, descriptors);
	Mat descriptorsToAdd = ((Mat)descriptors).reshape(1, 1);
	descriptorsToAdd.convertTo(descriptorsToAdd, CV_32F);
	(*features).push_back(descriptorsToAdd);

	// Rotated 90 image : Compute HOG and add to features
	Mat rotated90;
	rotate(resizedImg, rotated90, ROTATE_90_CLOCKWISE);
	descriptors = std::vector<float>();
	descr.compute(rotated90, descriptors);
	descriptorsToAdd = ((Mat)descriptors).reshape(1, 1);
	descriptorsToAdd.convertTo(descriptorsToAdd, CV_32F);
	(*features).push_back(descriptorsToAdd);

	// Rotated -90 image : Compute HOG and add to features
	Mat rotatedn90;
	rotate(resizedImg, rotatedn90, ROTATE_90_COUNTERCLOCKWISE);
	descriptors = std::vector<float>();
	descr.compute(rotatedn90, descriptors);
	descriptorsToAdd = ((Mat)descriptors).reshape(1, 1);
	descriptorsToAdd.convertTo(descriptorsToAdd, CV_32F);
	(*features).push_back(descriptorsToAdd);

	// Rotated 180 image : Compute HOG and add to features
	Mat rotated180;
	rotate(resizedImg, rotated180, ROTATE_180);
	descriptors = std::vector<float>();
	descr.compute(rotated180, descriptors);
	descriptorsToAdd = ((Mat)descriptors).reshape(1, 1);
	descriptorsToAdd.convertTo(descriptorsToAdd, CV_32F);
	(*features).push_back(descriptorsToAdd);

	// Flipped 1 : Compute HOG and add to features
	Mat flipped1;
	flip(resizedImg, flipped1, 1);
	descriptors = std::vector<float>();
	descr.compute(flipped1, descriptors);
	descriptorsToAdd = ((Mat)descriptors).reshape(1, 1);
	descriptorsToAdd.convertTo(descriptorsToAdd, CV_32F);
	(*features).push_back(descriptorsToAdd);

	// Flipped 0 : Compute HOG and add to features
	Mat flipped0;
	flip(resizedImg, flipped0, 0);
	descriptors = std::vector<float>();
	descr.compute(flipped0, descriptors);
	descriptorsToAdd = ((Mat)descriptors).reshape(1, 1);
	descriptorsToAdd.convertTo(descriptorsToAdd, CV_32F);
	(*features).push_back(descriptorsToAdd);

	// Flipped -1 : Compute HOG and add to features
	Mat flippedn1;
	flip(resizedImg, flippedn1, -1);
	descriptors = std::vector<float>();
	descr.compute(flippedn1, descriptors);
	descriptorsToAdd = ((Mat)descriptors).reshape(1, 1);
	descriptorsToAdd.convertTo(descriptorsToAdd, CV_32F);
	(*features).push_back(descriptorsToAdd);

	Mat lightImage = Mat::zeros(resizedImg.size(), resizedImg.type());
	float alpha = 1.3;
	float beta = 40;
	/// Do the operation new_image(i,j) = alpha*image(i,j) + beta
	for (int y = 0; y < lightImage.rows; y++)
	{
		for (int x = 0; x < lightImage.cols; x++)
		{
			for (int c = 0; c < 3; c++)
			{
				lightImage.at<Vec3b>(y, x)[c] =
					saturate_cast<uchar>(alpha * (resizedImg.at<Vec3b>(y, x)[c]) + beta);
			}
		}
	}
	descriptors = std::vector<float>();
	descr.compute(lightImage, descriptors);
	descriptorsToAdd = ((Mat)descriptors).reshape(1, 1);
	descriptorsToAdd.convertTo(descriptorsToAdd, CV_32F);
	(*features).push_back(descriptorsToAdd);

	Mat Bhigh = resizedImg + Scalar(75, 75, 75);
	descriptors = std::vector<float>();
	descr.compute(Bhigh, descriptors);
	descriptorsToAdd = ((Mat)descriptors).reshape(1, 1);
	descriptorsToAdd.convertTo(descriptorsToAdd, CV_32F);
	(*features).push_back(descriptorsToAdd);

	Mat Blow = resizedImg - Scalar(75, 75, 75);
	descriptors = std::vector<float>();
	descr.compute(Blow, descriptors);
	descriptorsToAdd = ((Mat)descriptors).reshape(1, 1);
	descriptorsToAdd.convertTo(descriptorsToAdd, CV_32F);
	(*features).push_back(descriptorsToAdd);

	Mat Bhigh2 = resizedImg + Scalar(30, 30, 30);
	descriptors = std::vector<float>();
	descr.compute(Bhigh2, descriptors);
	descriptorsToAdd = ((Mat)descriptors).reshape(1, 1);
	descriptorsToAdd.convertTo(descriptorsToAdd, CV_32F);
	(*features).push_back(descriptorsToAdd);

	Mat Blow2 = resizedImg - Scalar(30, 30, 30);
	descriptors = std::vector<float>();
	descr.compute(Blow2, descriptors);
	descriptorsToAdd = ((Mat)descriptors).reshape(1, 1);
	descriptorsToAdd.convertTo(descriptorsToAdd, CV_32F);
	(*features).push_back(descriptorsToAdd);

	Mat Chigh = resizedImg * 2;
	descriptors = std::vector<float>();
	descr.compute(Chigh, descriptors);
	descriptorsToAdd = ((Mat)descriptors).reshape(1, 1);
	descriptorsToAdd.convertTo(descriptorsToAdd, CV_32F);
	(*features).push_back(descriptorsToAdd);

	Mat Clow = resizedImg / 2;
	descriptors = std::vector<float>();
	descr.compute(Clow, descriptors);
	descriptorsToAdd = ((Mat)descriptors).reshape(1, 1);
	descriptorsToAdd.convertTo(descriptorsToAdd, CV_32F);
	(*features).push_back(descriptorsToAdd);
}