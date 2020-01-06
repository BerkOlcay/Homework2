
#include <opencv2/opencv.hpp>
#include <iostream>


#include "HOG_Descriptor.h"
#include "RandomForest.h"
#include "bounding_box.h"

#include <filesystem>
namespace fs = std::filesystem;
using namespace std;

template<class ClassifierType>
void performanceEval(cv::Ptr<ClassifierType> classifier, cv::Ptr<cv::ml::TrainData> data) {

    /*

        Fill Code

    */

};


void testDTrees() {

	int num_classes = 6;

	/*
	  *
	  * Create your data (i.e Use HOG from task 1 to compute the descriptor for your images)
	  * Train a Forest and evaluate the performance
	  * Experiment with the MaxDepth & TreeCount parameters, to see how it affects the performance

	*/


	String file_path = __FILE__;
	string dir_path = file_path.substr(0, file_path.rfind("\\"));
	string test_dir_path = dir_path + "\\data\\task2\\test";

	Mat train_labels, train_features;

	for (int i = 0; i < num_classes; i++) {
		string train_dir_name = "0" + to_string(i);
		string train_dir_path = dir_path + "\\data\\task2\\train\\" + train_dir_name;

		vector<float> descriptors;
		HOG_Descriptor* hog = new HOG_Descriptor();
		Size default_size = hog->getWinSize();


		for (const auto& entry : fs::directory_iterator(train_dir_path)) {
			string image_path = entry.path().string();
			Mat im = imread(image_path);
			train_labels.push_back(i);
			descriptors = std::vector<float>();
			hog->detectHOGDescriptor(im, descriptors, default_size, false);
			Mat descriptorsToAdd = ((Mat)descriptors).reshape(1, 1);
			descriptorsToAdd.convertTo(descriptorsToAdd, CV_32F);
			train_features.push_back(descriptorsToAdd);
		}
	}
	cv::Ptr<cv::ml::TrainData> train_data = cv::ml::TrainData::create(train_features, cv::ml::ROW_SAMPLE, train_labels);


	Mat test_labels, test_features;

	for (int i = 0; i < num_classes; i++) {
		string train_dir_name = "0" + to_string(i);
		string train_dir_path = dir_path + "\\data\\task2\\test\\" + train_dir_name;

		vector<float> descriptors;
		HOG_Descriptor* hog = new HOG_Descriptor();
		Size default_size = hog->getWinSize();


		for (const auto& entry : fs::directory_iterator(train_dir_path)) {
			string image_path = entry.path().string();
			Mat im = imread(image_path);
			test_labels.push_back(i);
			descriptors = std::vector<float>();
			hog->detectHOGDescriptor(im, descriptors, default_size, false);
			Mat descriptorsToAdd = ((Mat)descriptors).reshape(1, 1);
			descriptorsToAdd.convertTo(descriptorsToAdd, CV_32F);
			test_features.push_back(descriptorsToAdd);
		}
	}

	int treeCount = 1;
	int maxCategories = 6;
	int maxDepth = 10;
	int minSampleCount = 2;
	int CVFolds = 1;

	RandomForest randomforest = RandomForest::RandomForest(treeCount, maxDepth, CVFolds, minSampleCount, maxCategories);


	randomforest.train(train_features, train_labels);

	cv::Ptr<cv::ml::TrainData> test_data = cv::ml::TrainData::create(test_features, cv::ml::ROW_SAMPLE, test_labels);
	Mat y_pred;
	Mat confidence;

	randomforest.predict(test_features, test_labels, &y_pred, test_data, &confidence, true);

	//performanceEval<RandomForest>(forest, train_data);
	//performanceEval<RandomForest>(forest, test_data);

}


void testForest() {

    int num_classes = 6;

    /*
      *
      * Create your data (i.e Use HOG from task 1 to compute the descriptor for your images)
      * Train a Forest and evaluate the performance
      * Experiment with the MaxDepth & TreeCount parameters, to see how it affects the performance

    */


	String file_path = __FILE__;
	string dir_path = file_path.substr(0, file_path.rfind("\\"));
	string test_dir_path = dir_path + "\\data\\task2\\test";

	Mat train_labels, train_features;

	for (int i = 0; i < num_classes; i++) {
		string train_dir_name = "0" + to_string(i);
		string train_dir_path = dir_path + "\\data\\task2\\train\\" + train_dir_name;

		vector<float> descriptors;
		HOG_Descriptor* hog = new HOG_Descriptor();
		Size default_size = hog->getWinSize();


		for (const auto& entry : fs::directory_iterator(train_dir_path)) {
			string image_path = entry.path().string();
			Mat im = imread(image_path);
			train_labels.push_back(i);
			descriptors = std::vector<float>();
			hog->detectHOGDescriptor(im, descriptors, default_size, false);
			Mat descriptorsToAdd = ((Mat)descriptors).reshape(1, 1);
			descriptorsToAdd.convertTo(descriptorsToAdd, CV_32F);
			train_features.push_back(descriptorsToAdd);
		}
	}
	cv::Ptr<cv::ml::TrainData> train_data = cv::ml::TrainData::create(train_features, cv::ml::ROW_SAMPLE, train_labels);


	Mat test_labels, test_features;

	for (int i = 0; i < num_classes; i++) {
		string train_dir_name = "0" + to_string(i);
		string train_dir_path = dir_path + "\\data\\task2\\test\\" + train_dir_name;

		vector<float> descriptors;
		HOG_Descriptor* hog = new HOG_Descriptor();
		Size default_size = hog->getWinSize();


		for (const auto& entry : fs::directory_iterator(train_dir_path)) {
			string image_path = entry.path().string();
			Mat im = imread(image_path);
			test_labels.push_back(i);
			descriptors = std::vector<float>();
			hog->detectHOGDescriptor(im, descriptors, default_size, false);
			Mat descriptorsToAdd = ((Mat)descriptors).reshape(1, 1);
			descriptorsToAdd.convertTo(descriptorsToAdd, CV_32F);
			test_features.push_back(descriptorsToAdd);
		}
	}

	int treeCount = 20;
	int maxCategories = 6;
	int maxDepth = 10;
	int minSampleCount = 2;
	int CVFolds = 1;

	RandomForest randomforest = RandomForest::RandomForest(treeCount, maxDepth, CVFolds, minSampleCount, maxCategories);

	randomforest.train(train_features, train_labels);

	cv::Ptr<cv::ml::TrainData> test_data = cv::ml::TrainData::create(test_features, cv::ml::ROW_SAMPLE, test_labels);
	Mat y_pred;
	Mat confidence;

	randomforest.predict(test_features, test_labels, &y_pred, test_data, &confidence, true);

	std::cout << "average train error " << y_pred << endl;
	std::cout << "average train error " << test_labels << endl;


    //performanceEval<RandomForest>(forest, train_data);
    //performanceEval<RandomForest>(forest, test_data);
}

void task3() {

	int num_classes = 4;

	String file_path = __FILE__;
	string dir_path = file_path.substr(0, file_path.rfind("\\"));

	std::string trainFolder3("\\data\\task3\\train\\0");
	std::string testFolder3("\\data\\task3\\test\\");
	std::string gtfolder("\\data\\task3\\gt");
	std::string resultsfolder("\\data\\task3\\results\\");

	//Train dataset constitution
	Mat labels3, features3;
	std::string absoluteTrainPath3(dir_path + trainFolder3);
	std::cout << "creation of train dataset" << endl;

	//create hog instance
	HOG_Descriptor* hog = new HOG_Descriptor();

	// Constitution of the features matrix - training set
	hog->create_dataset(num_classes, absoluteTrainPath3, &labels3, &features3, 1);

	std::string absoluteTestPath3(dir_path + testFolder3);

	int treeCount = 20;
	int maxCategories = 4;
	int maxDepth = 10;
	int minSampleCount = 2;
	int CVFolds = 1;
	RandomForest randomforest = RandomForest::RandomForest(treeCount, maxDepth, CVFolds, minSampleCount, maxCategories);
	randomforest.train(features3, labels3);

	//variables for the test
	Mat image; //test image
	vector<string> vectorImg; //vector of file names
	string imgName; //name of test file
	int track = 0; //keep track of the nb of test images we read
	vector<int> correct_pred(14, 0); // nb of correct predicted box for each threshold
	int nb_gt = 0;
	vector<int> nb_pred(14, 0); // nb of predicted box for each threshold
	vector<int> threshold{ 50, 53, 55, 57, 60, 63, 65, 67, 70, 73, 75, 80, 83,85 }; //thresholds for the NMS
	vector<float> precision;
	vector<float> recall;
	string resultsFolderAbs = dir_path + resultsfolder;
	float threshold_IOU = 0.2;

	int windowsSizeArray[4] = { 80,112,144,176 };

	for (const auto& entry : fs::directory_iterator(absoluteTestPath3)) {
		string image_path = entry.path().string();
		Mat image = imread(image_path);
		imshow(imgName, image);

		//create the ground truth boxes 
		std::map<int, Rect> ground_truth;
		tie(ground_truth, nb_gt) = create_ground_truth(dir_path + gtfolder + imgName.substr(0, 4) + ".gt.txt", nb_gt);
		vector<Rect> rects = get_multi_sliding_windows(image, track, windowsSizeArray, 4);
		Mat y_pred;
		Mat confidence;
		tie(y_pred, confidence) = prediction(&randomforest, image, track, rects, HOG_Descriptor::win_size.height, HOG_Descriptor::cell_size.height, HOG_Descriptor::block_size.height);
		std::cout << "sliding windows done" << endl;

		for (int i = 0; i < threshold.size(); i++) {
			image = imread(absoluteTestPath3 + imgName);
			vector<int> resultsLabels;
			vector<float> resultsConfidence;
			vector<Rect> resultsRect;
			//keep windows that detects an object with a confidence above the threshold
			classify_windows(image, track, rects, y_pred, confidence, threshold[i], resultsFolderAbs + std::to_string(threshold[i]) + "/", &resultsRect, &resultsLabels, &resultsConfidence);
			// compute NMS and update the number of predicted boxes
			nb_pred[i] = NMS(image, track, nb_pred[i], resultsFolderAbs + std::to_string(threshold[i]) + "/", &resultsRect, &resultsLabels, &resultsConfidence, threshold_IOU);
			// Evaluate the detection result
			correct_pred[i] = int_over_union(resultsRect, resultsLabels, correct_pred[i], ground_truth, image, track, resultsFolderAbs + std::to_string(threshold[i]) + "/");
		}
		track++;
	}

	//compute the precision and recall
	for (int i = 0; i < threshold.size(); i++) {
		std::cout << "the number of predicted bounding boxes " << nb_pred[i] << endl;
		std::cout << " the number of ground truth bounding boxes" << nb_gt << endl;
		precision.push_back((float)correct_pred[i] / (float)nb_pred[i]);
		recall.push_back((float)correct_pred[i] / (float)nb_gt);
		std::cout << "for threshold " << threshold[i] << " precision " << precision[i] << " recall " << recall[i] << endl;
	}
	visualize_vector(precision, "precision", 14);
	visualize_vector(recall, "recall", 14);

	waitKey(0);
}

int main() {
    //testDTrees();
    //testForest();
	task3();

    return 0;
}


