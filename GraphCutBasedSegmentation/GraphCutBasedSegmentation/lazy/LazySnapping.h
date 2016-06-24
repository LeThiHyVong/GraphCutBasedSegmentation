#ifndef LAZY_SNAPPING_H
#define LAZY_SNAPPING_H

#include <opencv2\opencv.hpp>
#include "..\max_flow\graph.h"
#include <vector>
#include <iostream>
#include <cmath>
#include <list>
#include <fstream>

#define K 64

class LazySnapping
{
private:
	typedef Graph<double, double, double> GraphType;
	std::vector<cv::Point> forePts;
	std::vector<cv::Point> backPts;
	
	cv::Mat src;

	// Graph info
	GraphType* graph;

	// pre-segmentation component number
	int n;

	cv::Mat connected;
	cv::Mat visited;
	std::vector< cv::Vec3f > centers;

	// Foreground K-mean
	cv::Mat foreground_seeds;
	cv::Mat foreground_centers;
	
	// Background K-mean
	cv::Mat background_seeds;
	cv::Mat background_centers;

	// Segment connect to Source
	std::vector< bool > connectToSource;

	// Segment connect to Sink
	std::vector< bool > connectToSink;

	// Final labelling
	std::vector < int > FLabel;

	// control update seeds
	bool isUpdateF;
	bool isUpdateB;

	int KF;
	int KB;

	cv::Mat markers;
	bool isWaterShed;

	const int r[4] = { 0, 1, 0,-1 };
	const int c[4] = { 1, 0,-1, 0 };

	int getComponentNumber(cv::Mat markers);

	void k_meanForeground();

	void k_meanBackground();

	cv::Vec3f getAvgColor(const cv::Mat dataPoints);

	void BFS(int i, int j, int currentLabel, int reLabel, cv::Mat markers, cv::Mat &dataPoints, cv::Mat &visited);

	void initSuperPixel();

	void initNeighboring();

	void initSegment();

	void initWaterShed();

	void initSeeds();

	void getE1(int currentLabel, double* energy);

	float colorDistance(cv::Vec3f color1, cv::Vec3f color2);

	float getE2(int xlabel, int ylabel);

	void initGraph();

	void getLabellingValue();

public:
	LazySnapping();
	~LazySnapping();

	void setSourceImage(cv::Mat image);

	void setUpdateF(bool value);
	void setUpdateB(bool value);

	void setForegroundPoints(const std::vector<cv::Point>& points);

	void setBackgroundPoints(const std::vector<cv::Point>& points);

	void initMarkers();

	void runMaxFlow();

	cv::Mat getImageMask();

	cv::Mat getImageColor();

	cv::Mat changeLabelSegment(int x, int y);

};

#endif

