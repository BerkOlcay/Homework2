
#include <opencv2/opencv.hpp>

#include "HOG_Descriptor.h"

using namespace std;
#include <iostream>
using namespace cv;




int main() {
    String file_path = __FILE__;
    string dir_path = file_path.substr(0, file_path.rfind("\\"));

    Mat im = imread(dir_path + "\\data\\task1\\obj1000.jpg");

    // Change image to grayscale
    Mat grayImg;
    cvtColor(im, grayImg, COLOR_BGR2GRAY);
    imwrite(dir_path + "\\data\\task1\\grayscale.jpg", grayImg);

    // Flip mirror ( code 1 )
    Mat flipImg;
    flip(im, flipImg, 1);
    imwrite(dir_path + "\\data\\task1\\flip.jpg", flipImg);

    // Resize
    Mat resImg;
    resize(im, resImg, Size(256, 256), 0, 0, INTER_NEAREST);
    imwrite(dir_path + "\\data\\task1\\resize.jpg", resImg);

    // Rotate 90 clockwise
    Mat rotImg;
    rotate(im, rotImg, ROTATE_90_CLOCKWISE);
    imwrite(dir_path + "\\data\\task1\\resize.jpg", rotImg);

    // Padding 10px left and bottom with replicate border
    Mat paddImg;
    int border = 10;
    copyMakeBorder(im, paddImg, 0, border, border, 0, BORDER_REPLICATE);
    imwrite(dir_path + "\\data\\task1\\padding.jpg", paddImg);


    //Fill Code here

    /*
        * Create instance of HOGDescriptor and initialize
        * Compute HOG descriptors
        * visualize
    */
    std::vector<float> descriptors = std::vector<float>();
    HOG_Descriptor* hog = new HOG_Descriptor();
    Size default_size = hog->getWinSize();
    hog->detectHOGDescriptor(im, descriptors, default_size, false);
    cout << descriptors.size() << endl;
    hog->detectHOGDescriptor(grayImg, descriptors, default_size, false);
    cout << descriptors.size() << endl;
    hog->detectHOGDescriptor(flipImg, descriptors, default_size, false);
    cout << descriptors.size() << endl;
    hog->detectHOGDescriptor(resImg, descriptors, default_size, false);
    cout << descriptors.size() << endl;
    hog->detectHOGDescriptor(rotImg, descriptors, default_size, false);
    cout << descriptors.size() << endl;
    hog->detectHOGDescriptor(paddImg, descriptors, default_size, true);
    cout << descriptors.size() << endl;

    return 0;
}