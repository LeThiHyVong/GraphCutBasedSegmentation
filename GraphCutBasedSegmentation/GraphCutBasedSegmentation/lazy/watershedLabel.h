#pragma once
#include <opencv2\opencv.hpp>

class Watershed
{
private:
	cv::Mat src;
	cv::Mat markers;

public:

	Watershed() = default;
	~Watershed() = default;

	void setSourceImage(cv::Mat source) { src = source; }
	void setMarkers(cv::Mat mark) { markers = mark; }

	cv::Mat getMarkersLabel();

};