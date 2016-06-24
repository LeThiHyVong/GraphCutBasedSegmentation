#include "GraphCutSegmentation.h"
#include <cstdio>
#include <vector>
#include <cstdlib>
#include <cstring>
#include <fstream>

void GraphCutSegmentation::initComponent(const cv::Mat& origImg, const cv::Mat& seedMask) {	

	cv::Mat data_points;
	origImg.convertTo(data_points, CV_32FC3);
	data_points = data_points.reshape(0, origImg.rows * origImg.cols);

	cv::kmeans(data_points,
		nCluster,
		cluster_idx,
		cv::TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 50, 1.0),
		1,
		cv::KMEANS_RANDOM_CENTERS,
		center
	);

	// cv::Mat tmp = cluster_idx.reshape(0, origImg.rows);
	// cv::convertScaleAbs(tmp, tmp, int(255 / nCluster));
	// cv::imshow("kmean", tmp);

	calcK();

	std::vector<int> obj_hist(nCluster + 1), bkg_hist(nCluster + 1);
	bkgRelativeHistogram.resize(nCluster);
	objRelativeHistogram.resize(nCluster);

	for (int i = 0; i < imgHeight; i++) {
		for (int j = 0; j < imgWidth; j++) {

			auto which_c = cluster_idx.at<int>(i * imgWidth + j, 0);

			switch (seedMask.at<char>(i, j)) {

			case OBJECT:
				obj_hist[which_c]++;
				obj_hist[nCluster]++;
				break;

			case BACKGROUND:
				bkg_hist[which_c]++;
				bkg_hist[nCluster]++;
				break;

			default:
				break;

			}


		}
	}

	for (int i = 0; i < nCluster; i++) {

		bkgRelativeHistogram[i] = 1.0f * bkg_hist[i] / bkg_hist[nCluster];
		objRelativeHistogram[i] = 1.0f * obj_hist[i] / obj_hist[nCluster];

	}
	
}

void GraphCutSegmentation::buildGraph(const cv::Mat& seedMask) {

	g->add_node(imgWidth * imgHeight);
	
	cv::Rect imgRect(cv::Point(), cv::Size(imgWidth, imgHeight));

	for (int i = 0; i < imgHeight; i++) {

		for (int j = 0; j < imgWidth; j++) {

			int node = i * imgWidth + j;
			cv::Point pix(j, i);

			// Relation to source and sink
			g->add_tweights(
				node,
				calcTWeight(pix, seedMask.at<char>(pix)),
				calcTWeight(pix, seedMask.at<char>(pix), false)
			);

			// Relation to neighbors
			for (auto &k : neighbor8) {

				cv::Point neighborPix = pix + k;

				if (imgRect.contains(neighborPix)) {

					int neighborNode = convertPixelToNode(neighborPix);

					g->add_edge(node, neighborNode,
						calcNWeight(node, neighborNode),
						calcNWeight(neighborNode, node));

				}
			}

		}
	}

}

float GraphCutSegmentation::calcTWeight(const cv::Point& pix, int pixType, bool toSource) {

	float retVal = 0.0f;

	switch (pixType) {

	case OBJECT:
		retVal = (toSource ? K : 0);
		break;

	case BACKGROUND:
		retVal = (toSource ? 0 : K);
		break;

	case UNKNOWN:
		retVal = (toSource ? lambda * Pr_bkg(pix) : lambda * Pr_obj(pix));
		break;

	}

	return retVal;

}

float GraphCutSegmentation::calcNWeight(int node1, int node2)
{
	return B_pq(convertNodeToPixel(node1), convertNodeToPixel(node2));
}

float GraphCutSegmentation::B_pq(const cv::Point& pix1, const cv::Point& pix2)
{

	float intensityDiff = cv::norm(center.row(cluster_idx.at<int>(convertPixelToNode(pix1), 0)),
		center.row(cluster_idx.at<int>(convertPixelToNode(pix2), 0)),
		distType);

	return  w_exp * exp(- intensityDiff * intensityDiff / (2 * sigmaSqr)) /  cv::norm(pix2-pix1);

}

void GraphCutSegmentation::calcK()
{
	cv::Rect imgRect(cv::Point(), cv::Size(imgWidth, imgHeight));
	for (int i = 0; i < imgHeight; i++) {

		for (int j = 0; j < imgWidth; j++) {

			float K_buf = 0;
			cv::Point curPix(j, i);

			for (auto &k : neighbor8) {

				cv::Point neighborPix = curPix + k;

				if (imgRect.contains(neighborPix)) {
					K_buf += B_pq(curPix, neighborPix);
				}

			}

			K = std::max(K, K_buf);

		}
	}

	K = K + 1.0;
}

float GraphCutSegmentation::Pr_bkg(const cv::Point& pix) {

	return w_prob * -log(bkgRelativeHistogram[cluster_idx.at<int>(convertPixelToNode(pix), 0)]);

}

float GraphCutSegmentation::Pr_obj(const cv::Point& pix) {

	return w_prob * -log(objRelativeHistogram[cluster_idx.at<int>(convertPixelToNode(pix), 0)]);
}

void GraphCutSegmentation::cutGraph(cv::Mat& outputMask) {

	outputMask.create(cv::Size(imgWidth, imgHeight), CV_8U);

	float flow = g->maxflow();
	//printf("		Flow = %f\n", flow);

	int node = 0;

	int numBkg = 0, numObj = 0;
	for (int i = 0; i < imgHeight; i++) {
		for (int j = 0; j < imgWidth; j++) {

			if (g->what_segment(node) == GraphType::SOURCE) {
				outputMask.at<uchar>(i, j) = 255;
				numObj++;
			}
			else if (g->what_segment(node) == GraphType::SINK) {
				outputMask.at<uchar>(i, j) = 0;
				numBkg++;
			}
			node++;
		}
	}

	// cv::imshow("Mask", outputMask);
	//printf("		bkg=%d obj=%d total=%d\n", numBkg, numObj, numBkg + numObj);

}

void GraphCutSegmentation::segment(const cv::Mat& img, const cv::Mat& seedMask, cv::Mat& outputMask) {

	g = std::unique_ptr<GraphType>(new GraphType(getNumNodes(img), getNumEdges(img)));
	imgWidth = img.cols;
	imgHeight = img.rows;
	//uint64_t start;

	//printf("	setting up ..\n");

	//start = cv::getTickCount();
	initComponent(img, seedMask);
	//printf("	%.2lf sec\n", double(cv::getTickCount() - start) / cv::getTickFrequency());

	//printf("	************************************************************\n\n");

	//printf("	building graph...\n");

	//start = cv::getTickCount();
	buildGraph(seedMask);
	//printf("	%.2lf sec\n", double(cv::getTickCount() - start) / cv::getTickFrequency());

	//printf("	************************************************************\n\n");

	//printf("	segmenting image...\n");

	//start = cv::getTickCount();
	cutGraph(outputMask);
	//printf("	%.2lf sec\n", double(cv::getTickCount() - start) / cv::getTickFrequency());

}

GraphCutSegmentation::~GraphCutSegmentation() {
	g->reset();
	g.reset();	
}

