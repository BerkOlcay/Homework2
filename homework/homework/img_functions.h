#include<opencv2/opencv.hpp>
using namespace std;
using namespace cv;

#include <string>
#include <filesystem>
#include <fstream>
#include<opencv2/opencv.hpp>
#include <iostream>
#include <iterator>
#include <algorithm>
#include <opencv2/ml.hpp>

#define UNDEFINED 0
#define SQUARE 1
#define SQUARE_DEFINED 2
#define SMART_RESIZING 3
#define SQUARE_RESIZED 4

int findNextMultiple(int number, int multiple);
void paddingToNextShape(Mat src, Mat &dst, int multiple, int mode, Size &newSize);
std::tuple< std::map<int, Rect>, int> create_ground_truth(std::string path, int nb_gt);
void visualize_vector(std::vector<string> labels, std::string label, int num_elements);
void visualize_vector(std::vector<float> values, std::string label, int num_elements);