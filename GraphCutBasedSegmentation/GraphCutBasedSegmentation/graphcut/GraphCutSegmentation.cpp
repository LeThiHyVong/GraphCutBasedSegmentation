#include "GraphCutSegmentation.h"
#include <cstdio>
#include <vector>
#include <cstdlib>
#include <cstring>
#include <fstream>

void GraphCutSegmentation::calcColorVariance(const cv::Mat & origImg) {
	cv::Scalar tmp = cv::mean(origImg);
	cv::Vec3f avgColor{ (float)tmp[0], (float)tmp[1], (float)tmp[2] };
	sigmaSqr = { 0.0f, 0.0f, 0.0f };
	for (int r = 0; r < origImg.rows; r++)
		for (int c = 0; c < origImg.cols; c++) {
			cv::Vec3f tmpPix = origImg.at<cv::Vec3b>(r, c);
			auto diff = tmpPix - avgColor;
			sigmaSqr += cv::Vec3f{diff[0] * diff[0], \
				diff[1] * diff[1], \
				diff[2] * diff[2]};
		}
	int numPix = origImg.rows * origImg.cols;
	sigmaSqr /= numPix;

}

void GraphCutSegmentation::initComponent(const cv::Mat& origImg, const cv::Mat& seedMask) {

	calcColorVariance(origImg);
	cv::Mat data_points;
	origImg.convertTo(data_points, CV_32FC3);
	data_points = data_points.reshape(0, origImg.rows * origImg.cols);

	cv::kmeans(data_points,
		nCluster,
		cluster_idx,
		cv::TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 50, 1.0),
		1,
		cv::KMEANS_RANDOM_CENTERS
	);

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

void GraphCutSegmentation::buildGraph(const cv::Mat& origImg, const cv::Mat& seedMask) {

	g->add_node(imgWidth * imgHeight);

	cv::Rect imgRect(cv::Point(), cv::Size(imgWidth, imgHeight));
	K = 0.0f;
	std::vector<bool> isAdded(imgWidth * imgHeight, false);

	for (int i = 0; i < imgHeight; i++) {

		for (int j = 0; j < imgWidth; j++) {

			int node = i * imgWidth + j;
			cv::Point pix(j, i);

			auto tmpSumNLink = 0.0f;

			// Relation to neighbors
			for (auto &k : neighbor8) {

				cv::Point neighborPix = pix + k;

				if (imgRect.contains(neighborPix)) {

					int neighborNode = convertPixelToNode(neighborPix);
					auto tmpNWeight = calcNWeight(pix, neighborPix, origImg);
					tmpSumNLink += 2 * tmpNWeight;

					if (!isAdded[neighborNode]) {
						g->add_edge(node, neighborNode,
							tmpNWeight,
							tmpNWeight);
					}


				}
			}
			isAdded[node] = true;
			K = std::max(tmpSumNLink, K);
		}
	}

	K += 1.0f;

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

float GraphCutSegmentation::calcNWeight(const cv::Point& pix1, const cv::Point& pix2, const cv::Mat& origImg)
{
	auto r1 = origImg.at<cv::Vec3b>(pix1), \
		r2 = origImg.at<cv::Vec3b>(pix2);
	float intensityDiff = 0.0f;
	for (int i = 0; i < dim; i++) {
		float tmpDiff = r1[i] * 1.0f - r2[i];
		intensityDiff += (tmpDiff * tmpDiff / (2 * sigmaSqr[i]));
	}
	auto dist = pix2 - pix1;
	return  exp(-intensityDiff)
		/ std::sqrt(dist.x * dist.x + dist.y * dist.y);
}

float GraphCutSegmentation::Pr_bkg(const cv::Point& pix) {

	return -log(bkgRelativeHistogram[cluster_idx.at<int>(convertPixelToNode(pix), 0)]);

}

float GraphCutSegmentation::Pr_obj(const cv::Point& pix) {

	return -log(objRelativeHistogram[cluster_idx.at<int>(convertPixelToNode(pix), 0)]);
}

void GraphCutSegmentation::cutGraph(cv::Mat& outputMask) {

	outputMask.create(cv::Size(imgWidth, imgHeight), CV_8U);
	float flow = 0.0;
	flow = g->maxflow(!runFirstTime, NULL);
	runFirstTime = false;

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

}

void GraphCutSegmentation::segment(const cv::Mat& img, const cv::Mat& seedMask, cv::Mat& outputMask) {

	g.reset(new GraphType(getNumNodes(img), getNumEdges(img)));
	imgWidth = img.cols;
	imgHeight = img.rows;

	initComponent(img, seedMask);

	buildGraph(img, seedMask);

	cutGraph(outputMask);

}

void GraphCutSegmentation::updateSeeds(const std::vector<cv::Point>& newSeeds, PixelType pixType, cv::Mat& outputMask) {
	for (const auto& p : newSeeds) {
		g->mark_node(convertPixelToNode(p));
		g->add_tweights(
			convertPixelToNode(p),
			calcTWeight(p, pixType),
			calcTWeight(p, pixType, false)
		);
	}
	cutGraph(outputMask);
}

GraphCutSegmentation::GraphCutSegmentation() {
	initParam();
}

GraphCutSegmentation::~GraphCutSegmentation() {
	cleanGarbage();
}

