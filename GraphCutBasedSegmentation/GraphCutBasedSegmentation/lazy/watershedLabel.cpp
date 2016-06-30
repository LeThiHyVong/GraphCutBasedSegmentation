
#include "watershedLabel.h"

#include <ctype.h>
#include <stdio.h>
#include "SeedsRevised.h"
#include "Tools.h"

using namespace cv;
using namespace std;

const int MaxR = 2000;
const int MaxC = 2000;

Mat Watershed::getMarkersLabel()
{
	cv::Mat image = src;

	// Number of desired superpixels.
	int superpixels = 300;

	// Number of bins for color histograms (per channel).
	int numberOfBins = 10;

	// Size of neighborhood used for smoothing term, see [1] or [2].
	// 1 will be sufficient, >1 will slow down the algorithm.
	int neighborhoodSize = 1;

	// Minimum confidence, that is minimum difference of histogram intersection
	// needed for block updates: 0.1 is the default value.
	float minimumConfidence = 0.1;

	// The weighting of spatial smoothing for mean pixel updates - the euclidean
	// distance between pixel coordinates and mean superpixel coordinates is used
	// and weighted according to:
	//  (1 - spatialWeight)*colorDifference + spatialWeight*spatialDifference
	// The higher spatialWeight, the more compact superpixels are generated.
	float spatialWeight = 0.25;

	// Instantiate a new object for the given image.
	SEEDSRevisedMeanPixels seeds(image, superpixels, numberOfBins, neighborhoodSize, minimumConfidence, spatialWeight);

	int iterations = 20;
	// Initializes histograms and labels.
	seeds.initialize();
	// Runs a given number of block updates and pixel updates.
	seeds.iterate(iterations);

	// Save a contour image to the following location:
	std::string storeContours = "./contours.png";

	// bgr color for contours:
	int bgr[] = { 0, 0, 204 };

	// seeds.getLabels() returns a two-dimensional array containing the computed
	// superpixel labels.
	cv::Mat contourImage = Draw::contourImage(seeds.getLabels(), image, bgr);
	imwrite(storeContours, contourImage);
	imshow("Contours", contourImage);

	int** labels = seeds.getLabels();

	//cv::Mat labelImage = Draw::labelImage(labels, src);

	//imwrite("LabelsImage.bmp", labelImage);
	//imshow("LabelsImage.bmp", labelImage);

	//cv::Mat meanImage = Draw::meanImage(labels, src);
	//imwrite("MeanImage.bmp", meanImage);
	//imshow("MeanImage.bmp", meanImage);

	markers = Mat::zeros(src.rows, src.cols, CV_32S);

	for (int i = 0; i < markers.rows; i++) {
		for (int j = 0; j < markers.cols; j++)
		{
			markers.at<int>(i, j) = labels[i][j];
		}
			
	}

	return markers;
}
