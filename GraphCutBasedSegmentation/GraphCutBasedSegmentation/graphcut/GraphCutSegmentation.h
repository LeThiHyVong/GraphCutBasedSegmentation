#ifndef GRAPHCUT_SEGMENTATION_H_
#define GRAPHCUT_SEGMENTATION_H_

#include <memory>
#include <opencv2\opencv.hpp>
#include "..\max_flow\graph.h"

class GraphCutSegmentation {
	typedef Graph<double, double, double> GraphType;

public:

	enum PixelType {
		BACKGROUND = -1,
		UNKNOWN = 0,
		OBJECT = 1
	};

	GraphCutSegmentation();

	~GraphCutSegmentation();

	void setNCluster(int);

	void setNDimension(int);

	void setDiscontinuityThreshold(float);

	void setRegionBoundaryRelation(float);

	void setDistanceType(cv::NormTypes);

	void initComponent(const cv::Mat& origImg, const cv::Mat& seedMask);

	void buildGraph(const cv::Mat& seedMask);

	void cutGraph(cv::Mat& outMask);

	void segment(const cv::Mat& img, const cv::Mat& seedMask, cv::Mat& outputMask);	

	void updateSeeds(const std::vector<cv::Point>& newSeeds, PixelType pixType, cv::Mat& outputMask);

	void createDefault();
	void cleanGarbage();

private:

	const std::vector<cv::Point> neighbor8 {
		{-1, -1}, {-1,  0},
		{-1,  1}, { 0,  1},
		{ 1,  1}, { 1,  0},
		{ 1, -1}, { 0, -1}
	};

	std::unique_ptr<GraphType> g;

	int imgWidth, imgHeight;

	float K;

	int    nCluster;
	int    dim;
	float  sigmaSqr;
	float  lambda;
	bool runFirstTime;

	cv::Mat   cluster_idx, center;
	cv::NormTypes distType = cv::NORM_L1;
	std::vector<float> bkgRelativeHistogram;
	std::vector<float> objRelativeHistogram;

	void initParam();

	float calcTWeight(const cv::Point& pix, int pixType, bool toSource = true);

	float calcNWeight(int node1, int node2);

	int convertPixelToNode(const cv::Point&);

	cv::Point convertNodeToPixel(int node);

	float B_pq(const cv::Point&, const cv::Point&);

	void calcK();

	float Pr_bkg(const cv::Point&);

	float Pr_obj(const cv::Point&);

	int getNumNodes(const cv::Mat& img);

	int getNumEdges(const cv::Mat& img);

};

inline int GraphCutSegmentation::getNumEdges(const cv::Mat& img) {
	return img.cols * img.rows * 10;
}

inline int GraphCutSegmentation::getNumNodes(const cv::Mat& img) {
	return img.cols * img.rows;
}

inline int GraphCutSegmentation::convertPixelToNode(const cv::Point& pix)
{
	return pix.y * imgWidth + pix.x;
}

inline cv::Point GraphCutSegmentation::convertNodeToPixel(int node)
{
	return cv::Point(node % imgWidth, node / imgWidth);
}

inline void GraphCutSegmentation::setNCluster(int _cluster)
{
	nCluster = _cluster;
}

inline void GraphCutSegmentation::setNDimension(int _dim)
{
	dim = _dim;
}

inline void GraphCutSegmentation::setDiscontinuityThreshold(float _sigma)
{
	sigmaSqr = _sigma * _sigma;
}

inline void GraphCutSegmentation::setRegionBoundaryRelation(float _lambda)
{
	lambda = _lambda;
}


inline void GraphCutSegmentation::setDistanceType(cv::NormTypes normType)
{
	distType = normType;
}

inline void GraphCutSegmentation::initParam() {
	setNCluster(10);
	setNDimension(3);
	setDiscontinuityThreshold(40000.0f);
	setRegionBoundaryRelation(1.f);
	runFirstTime = true;
}

inline void GraphCutSegmentation::createDefault() {
	initParam();
}

inline void GraphCutSegmentation::cleanGarbage() {
	auto tmpPtr = g.release();
	if (tmpPtr != NULL) {
		tmpPtr->reset();
		delete tmpPtr;
	}
		
	//changedNode->Reset();
	//delete changedNode;
	//changedNode = NULL;	
}


#endif /* FUSION_FRAMEWORK_H_ */
