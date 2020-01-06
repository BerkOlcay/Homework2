#include "img_functions.h"

int findNextMultiple(int number, int multiple) {

	if (number % multiple == 0) {
		return number;
	}
	else {
		return ((number / multiple) + 1) * multiple;
	}

}

std::tuple< std::map<int, Rect>, int> create_ground_truth(std::string path, int nb_gt) {
	std::ifstream file(path);
	std::string line;
	std::map<int, Rect> ground_truth;
	while (std::getline(file, line)) {
		std::istringstream stream(line);
		int cpt = 0;
		int classe;
		Rect rectangle;
		while (stream)
		{
			string s;
			if (!std::getline(stream, s, ' ')) break;
			if (cpt == 0)
				classe = std::stoi(s);
			if (cpt == 1)
				rectangle.x = std::stoi(s);
			if (cpt == 2)
				rectangle.y = std::stoi(s);
			if (cpt == 3)
				rectangle.width = abs(std::stoi(s) - rectangle.x);
			if (cpt == 4)
				rectangle.height = abs(std::stoi(s) - rectangle.y);
			cpt++;
			//std::cout << s << endl;
		}
		ground_truth[classe] = rectangle;
		nb_gt++;
	}
	return std::make_tuple(ground_truth, nb_gt);
}

void paddingToNextShape(cv::Mat src, cv::Mat &dst, int multiple, int mode, Size &newSize) {
	
	int newCols, newRows, rows, cols;
	Size s;

	s = src.size();
	rows = s.height;
	cols = s.width;

	if (mode == UNDEFINED) {

		newRows = findNextMultiple(rows, multiple);
		newCols = findNextMultiple(cols, multiple);

		newSize = cv::Size(newCols, newRows);

		int borderTop = (int)(newSize.height - rows) / 2;
		int borderLeft = (int)(newSize.width - cols) / 2;
		int borderBottom = newSize.height - rows - borderTop;
		int borderRight = newSize.width - cols - borderLeft;

		copyMakeBorder(src, dst, borderTop, borderBottom, borderLeft, borderRight, BORDER_REPLICATE);

	}
	else if (mode == SQUARE) {
		// just resize to the samllest size multiple of "multiple". 


		int maxRowsCols;

		if (rows > cols) {
			maxRowsCols = rows;
		}
		else {
			maxRowsCols = cols;
		}

		newRows = findNextMultiple(maxRowsCols, multiple);
		newCols = newRows;

		newSize = cv::Size(newCols, newRows);

		int borderTop = (int)(newSize.height - rows) / 2;
		int borderLeft = (int)(newSize.width - cols) / 2;
		int borderBottom = newSize.height - rows - borderTop;
		int borderRight = newSize.width - cols - borderLeft;

		copyMakeBorder(src, dst, borderTop, borderBottom, borderLeft, borderRight, BORDER_REPLICATE);

	}
	else if (mode == SQUARE_DEFINED) { // valid only if size entered is upper than the one of pictures.
		// directly resize the picture to a square (valid only if the size requested is higher than the size of the pictures from the dataset).

		newRows = newSize.height;
		newCols = newSize.width;

		newSize = cv::Size(newCols, newRows);

		int borderTop = (int)(newSize.height - rows) / 2;
		int borderLeft = (int)(newSize.width - cols) / 2;
		int borderBottom = newSize.height - rows - borderTop;
		int borderRight = newSize.width - cols - borderLeft;

		copyMakeBorder(src, dst, borderTop, borderBottom, borderLeft, borderRight, BORDER_REPLICATE);

	}
	else if (mode == SMART_RESIZING) {
		// resize such that the temporary image upper dimension is 128 and then padding

		newRows = newSize.height;
		newCols = newSize.width;

		float factorRows = (float) rows / newRows;
		float factorCols = (float) cols / newCols;
		int newColsTmp, newRowsTmp;
		Size sizeTmp;
		Mat imgTmp;
		if (factorRows >= 1 || factorCols >= 1) {

			if (factorRows < factorCols) {
				newColsTmp = newSize.width;
				newRowsTmp = (int)((float)rows / factorCols);
				sizeTmp = Size(newColsTmp, newRowsTmp);
				resize(src, imgTmp, sizeTmp);
			}
			else {
				newRowsTmp = newSize.height;
				newColsTmp = (int)((float)cols / factorRows);
				sizeTmp = Size(newColsTmp, newRowsTmp);
				resize(src, imgTmp, sizeTmp);
			}

		}
		else {
			newRowsTmp = rows;
			newColsTmp = cols;
			sizeTmp = Size(newColsTmp, newRowsTmp);
			imgTmp = src.clone();
		}
		
		newSize = cv::Size(newCols, newRows);

		int borderTop = (int)(newSize.height - newRowsTmp) / 2;
		int borderLeft = (int)(newSize.width - newColsTmp) / 2;
		int borderBottom = newSize.height - newRowsTmp - borderTop;
		int borderRight = newSize.width - newColsTmp - borderLeft;

		copyMakeBorder(imgTmp, dst, borderTop, borderBottom, borderLeft, borderRight, BORDER_REPLICATE);

	}
	else if (mode == SQUARE_RESIZED) {
		// convert to square and then to desired size

		int newColsTmp, newRowsTmp;
		Size sizeTmp;
		Mat imgTmp;

		if (rows > cols) {
			newColsTmp = rows;
			newRowsTmp = rows;
		}
		else {
			newColsTmp = cols;
			newRowsTmp = cols;
		}

		int borderTop = (int)(newRowsTmp - rows) / 2;
		int borderLeft = (int)(newColsTmp - cols) / 2;
		int borderBottom = newRowsTmp - rows - borderTop;
		int borderRight = newColsTmp - cols - borderLeft;

		copyMakeBorder(src, imgTmp, borderTop, borderBottom, borderLeft, borderRight, BORDER_REPLICATE);

		newRows = newSize.height;
		newCols = newSize.width;
		newSize = cv::Size(newCols, newRows);

		resize(imgTmp, dst, newSize);

	}

	else {
		printf("Undefined usage \n");
		return;
	}

}

void visualize_vector(std::vector<string> labels, std::string label, int num_elements) {

	printf("Visualization : %s \n", label);
	printf("[");
	for (std::vector<string>::size_type i = 0; i != labels.size() && i != num_elements; i++) {
		printf("%s,", labels[i]);
	}
	printf("...]\n");

}

void visualize_vector(std::vector<float> values, std::string label, int num_elements) {

	printf("Visualization : %s \n", label);
	printf("[");
	for (std::vector<string>::size_type i = 0; i != values.size() && i != num_elements; i++) {
		printf("%f,", values[i]);
	}
	printf("...]\n");

}