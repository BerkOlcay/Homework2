
#include <opencv2/opencv.hpp>
#include <iostream>


#include "HOG_Descriptor.h"
#include "RandomForest.h"

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
      * Create your data (i.e Use HOG from task 1 to compute the descriptor for your images)
      * Train a single Decision Tree and evaluate the performance
      * Experiment with the MaxDepth parameter, to see how it affects the performance

    */

    cv::Ptr<cv::ml::DTrees> tree = cv::ml::DTrees::create();

    tree->setMaxCategories(maxCategories);
    tree->setMaxDepth(maxDepth);
    tree->setMinSampleCount(minSampleCount);
    tree->setCVFolds(CVFolds);

    cv::Ptr<cv::ml::TrainData> trainData = cv::ml::TrainData::create(subfeatures, cv::ml::ROW_SAMPLE, sublabels);

    // train tree
    tree->train(trainData);
    auto y_pred = cv::OutputArray(sublabels);
    y_pred.clear();
    auto error = tree->calcError(trainData, true, y_pred);

    std::cout << "training done and error " << error << endl;


    performanceEval<cv::ml::DTrees>(tree, train_data);
    performanceEval<cv::ml::DTrees>(tree, test_data);

}


void testForest() {

    int num_classes = 6;

    /*
      *
      * Create your data (i.e Use HOG from task 1 to compute the descriptor for your images)
      * Train a Forest and evaluate the performance
      * Experiment with the MaxDepth & TreeCount parameters, to see how it affects the performance

    */

    performanceEval<RandomForest>(forest, train_data);
    performanceEval<RandomForest>(forest, test_data);
}


int main() {
    testDTrees();
    testForest();
    return 0;
}
