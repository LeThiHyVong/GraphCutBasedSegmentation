#include <cstdio>
#include <cstdlib>
#include <fstream>

#include "graphcut\GraphCutSegmentation.h"
#include "lazy\LazySnapping.h"

#define FLAG		int

#define LBUTTON_OFF	0
#define LBUTTON_ON 	1

#define HINT_BACKGROUND 0
#define HINT_FOREGROUND 1

#define SIZE_RATIO 0.005f

#define DST "result\\"
#define SRC "dataset\\"

cv::Mat original_img;
cv::Mat type;
cv::Mat hint_img;

std::vector<std::string> inputList;
std::vector<cv::Point> hintObj, hintBkg;

FLAG    lbutton_flag;
FLAG    hint_flag;


void params_init() {
	lbutton_flag = LBUTTON_OFF;
	hint_flag = HINT_BACKGROUND;
}

void argument_disp() {

	printf("Usage:\n");
	printf("<binary> <mode> <input_file>\n");
	printf("where:\n");
	printf("	- binary: the compiled executable file, e.g InteractiveGraphCut.exe\n");
	printf("	- mode: 0 for only creating hints, 1 for creating hints and segmenting images\n");
	printf("	- input_file: the file contains the list of input images, generated by GenDataList.ps1\n");

}

void mouseHandler(int event, int x, int y, int flags, void *param) {

	cv::Point curPoint(x, y);

	switch (event) {

	case CV_EVENT_LBUTTONUP:
		lbutton_flag = LBUTTON_OFF;
		break;

	case CV_EVENT_LBUTTONDOWN:
		lbutton_flag = LBUTTON_ON;

	case CV_EVENT_MOUSEMOVE: {

		cv::Rect imgRect(cv::Point(), hint_img.size());

		if (lbutton_flag == LBUTTON_ON) {


			if (!imgRect.contains(curPoint))
				return;

			cv::Scalar seedColor;
			switch (hint_flag) {

			case HINT_BACKGROUND:

				hintBkg.push_back(curPoint);
				seedColor = { 255, 0, 0 };
				break;

			case HINT_FOREGROUND:

				hintObj.push_back(curPoint);
				seedColor = { 0, 255, 0 };
				break;
			}

			cv::circle(hint_img, curPoint, int(std::min(hint_img.cols, hint_img.rows) * SIZE_RATIO), seedColor, -1);

		}

		cv::imshow("Image", hint_img);

	}
							 break;

	case CV_EVENT_RBUTTONDOWN:

		hint_flag = 1 - hint_flag;

		if (hint_flag == HINT_BACKGROUND) {

			fprintf(stdout, "hints on background...\n");

		}
		else {

			fprintf(stdout, "hints on foreground...\n");

		}

		break;
	}

}

void setHint(const std::string& fileName) {

	original_img = cv::imread(SRC + fileName + ".jpg");

	if (original_img.empty()) {
		std::cout << fileName << " Image reading error!\n";
		return;
	}

	std::ifstream f(SRC + fileName + ".hint");

	if (f.good()) {
		std::cout << fileName << "'s hint exists.\n";
		f.close();
		return;
	}
	f.close();

	hint_img = original_img.clone();
	hintObj.clear();
	hintBkg.clear();

	cv::namedWindow("Image", CV_WINDOW_NORMAL);
	cv::setMouseCallback("Image", mouseHandler, NULL);
	cv::imshow("Image", original_img);
	cv::waitKey(0);
	cv::destroyAllWindows();
	cv::imwrite(DST + fileName + "_hint.jpg", hint_img, std::vector<int>{CV_IMWRITE_JPEG_QUALITY, 100});

	std::ofstream hintFile(SRC + fileName + ".hint");
	hintFile << hintBkg.size() << std::endl;
	for (auto &p : hintBkg)
		hintFile << p.x << " " << p.y << std::endl;
	hintFile << hintObj.size() << std::endl;
	for (auto &p : hintObj)
		hintFile << p.x << " " << p.y << std::endl;
	hintFile.close();

}

void getObj(const std::string& fileName) {

	original_img = cv::imread(SRC + fileName + ".jpg");
	if (original_img.empty()) {
		std::cout << fileName << " Hint file missing error!\n";
		return;
	}

	type = cv::Mat::zeros(cv::Size(original_img.cols, original_img.rows), CV_8S);
	std::vector<cv::Point> objPts, bkgPts;
	GraphCutSegmentation gc;
	LazySnapping ls;

	std::cout << "starting to segment image" << fileName << std::endl;
	std::ifstream hintFile(SRC + fileName + ".hint");
	int nSeed;
	hintFile >> nSeed;
	for (int i = 0; i < nSeed; i++) {
		int x, y;
		hintFile >> x >> y;
		type.at<char>(y, x) = GraphCutSegmentation::BACKGROUND;
		bkgPts.push_back({ x, y });
		ls.setUpdateB(true);
	}
	hintFile >> nSeed;
	for (int i = 0; i < nSeed; i++) {
		int x, y;
		hintFile >> x >> y;
		type.at<char>(y, x) = GraphCutSegmentation::OBJECT;
		objPts.push_back({ x, y });
		ls.setUpdateF(true);
	}

	uint64_t start, end;

	
	cv::Mat outMask;

	start = cv::getTickCount();
	gc.segment(original_img, type, outMask);
	end = cv::getTickCount();

	std::cout << "Interactive GraphCut: " << double(end - start) / cv::getTickFrequency() << "; ";

	cv::Mat obj;
	original_img.copyTo(obj, outMask);
	cv::imshow("gcObj", obj);
	cv::imwrite(DST + fileName + "_graphcut_object.jpg", obj, std::vector<int>{CV_IMWRITE_JPEG_QUALITY, 100});

	
	ls.setSourceImage(original_img);
	ls.setBackgroundPoints(bkgPts);
	ls.setForegroundPoints(objPts);

	start = cv::getTickCount();
	ls.initMarkers();
	ls.runMaxFlow();
	end = cv::getTickCount();

	std::cout << "Lazy Snapping: " << double(end - start) / cv::getTickFrequency() << std::endl;
	obj = ls.getImageColor();
	cv::imshow("lsObj", obj);
	cv::imwrite(DST + fileName + "_lazy_object.jpg", obj, std::vector<int>{CV_IMWRITE_JPEG_QUALITY, 100});

	cv::waitKey(0);
	cv::destroyAllWindows();
}

void readInputFile(int mode, const std::string& inputFile) {

	params_init();
	std::ifstream ifs(inputFile);
	if (!ifs.good()) {
		argument_disp();
		exit(1);
	}
	std::string tmpFile;
	while (true) {
		std::getline(ifs, tmpFile);
		if (ifs.eof())
			break;
		tmpFile = tmpFile.substr(0, tmpFile.find_last_of('.'));
		inputList.push_back(tmpFile);
	}
		
	for (auto &file : inputList)
		setHint(file);
	if (mode < 1)
		return;
	for (auto &file : inputList)
		getObj(file);

}

int main(int argc, char** argv) {

	switch (argc) {
	case 3:
		readInputFile(std::stoi(argv[1]), argv[2]);
		break;
	default:
		argument_disp();
	}

	return 0;
	
}

