#include "LazySnapping.h"
#include "watershedLabel.h"
#include <iomanip>

using namespace std;
using namespace cv;

LazySnapping::LazySnapping() : graph(NULL)
{
	forePts.clear();
	backPts.clear();
	isWaterShed = false;
	isUpdateF = false;
	isUpdateB = false;
}

LazySnapping::~LazySnapping()
{
	if (graph)
	{
		graph->reset();
		delete graph;
	}
}

void LazySnapping::setSourceImage(Mat image)
{
	src = image.clone();
}

inline void LazySnapping::setUpdateF(bool value) { isUpdateF = value; }

inline void LazySnapping::setUpdateB(bool value) { isUpdateB = value; }

void LazySnapping::setForegroundPoints(vector<cv::Point> points)
{
	forePts = points;
}

void LazySnapping::setBackgroundPoints(vector<cv::Point> points)
{
	backPts = points;
}

int LazySnapping::getComponentNumber(Mat markers)
{
	double result = 0;
	cv::minMaxIdx(markers, NULL, &result);
	return int(result + 1.0);
}

void LazySnapping::k_meanForeground()
{
	// K-mean for foreground seed
	connectToSource = vector< bool >(n, false);

	for (auto &i : forePts)
	{
		if (0 <= i.y && i.y < markers.rows
			&& 0 <= i.x && i.x <= markers.cols)
		{
			Vec3f idensity = src.at<Vec3b>(i.y, i.x);
			foreground_seeds.push_back(idensity);
			connectToSource[markers.at<int>(i.y, i.x)] = true;
		}
	}

	Mat label;
	Mat center;
	KF = K;
	if (foreground_seeds.rows < KF)
	{
		KF = foreground_seeds.rows;
	}

	kmeans(foreground_seeds,
		KF,
		label,
		cv::TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 50, 1.0),
		1,
		cv::KMEANS_RANDOM_CENTERS,
		center);

	foreground_centers = center;
}

void LazySnapping::k_meanBackground()
{
	// K-mean for background seed
	connectToSink = vector< bool >(n, false);

	for (auto &i : backPts)
	{
		if (0 <= i.y && i.y < markers.rows
			&& 0 <= i.x && i.x < markers.cols)
		{
			Vec3f idensity = src.at<Vec3b>(i.y, i.x);
			background_seeds.push_back(idensity);
			connectToSink[markers.at<int>(i.y, i.x)] = true;
		}
	}

	Mat label;
	Mat center;
	KB = K;
	if (background_seeds.rows < KB)
	{
		KB = background_seeds.rows;
	}

	kmeans(background_seeds,
		KB,
		label,
		cv::TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 50, 1.0),
		1,
		cv::KMEANS_RANDOM_CENTERS,
		center);

	background_centers = center;
}

Vec3f LazySnapping::getAvgColor(const Mat dataPoints)
{
	auto colorAvg = cv::mean(dataPoints);
	return{(float)colorAvg[0], (float)colorAvg[1], (float)colorAvg[2]};
}

void LazySnapping::BFS(int i, int j, int currentLabel, int reLabel, Mat markers, Mat & dataPoints, Mat & visited)
{
	// Create a queue for BFS
	list<pair<int, int>> queue;

	// Mark the current node as visited and enqueue it
	visited.at<int>(i, j) = true;
	queue.push_back(make_pair(i, j));

	int markers_rows = markers.rows;
	int markers_cols = markers.cols;

	while (!queue.empty())
	{
		// Dequeue a vertex from queue and print it
		int x = queue.front().first;
		int y = queue.front().second;

		Vec3f intensity = (Vec3f)src.at<Vec3b>(x, y);
		dataPoints.push_back(intensity);

		markers.at<int>(x, y) = reLabel;

		queue.pop_front();

		// Get all adjacent vertices of the dequeued vertex s
		// If a adjacent has not been visited, then mark it visited
		// and enqueue it
		for (int k = 0; k < 4; k++)
		{
			int xx = x + r[k];
			int yy = y + c[k];

			if (0 <= xx && xx < markers_rows
				&& 0 <= yy && yy < markers_cols)
				if (visited.at<int>(xx, yy) == 0)
				{
					int label = markers.at<int>(xx, yy);
					if (currentLabel == label)
					{
						visited.at<int>(xx, yy) = 1;
						queue.push_back(make_pair(xx, yy));
					}
				}
		}
	}
}

void LazySnapping::initSuperPixel()
{
	Mat visited = Mat::zeros(markers.rows, markers.cols, CV_32S);

	centers.resize(n * 10000);

	int reLabel = 0;
	for (int i = 0; i < markers.rows; i++)
	{
		for (int j = 0; j < markers.cols; j++)
			if (visited.at<int>(i, j) == 0)
			{
				Mat dataPoints;
				int currentLabel = markers.at<int>(i, j);
				BFS(i, j, currentLabel, reLabel, markers, dataPoints, visited);
				centers[reLabel] = getAvgColor(dataPoints);
				reLabel++;
			}
	}

	n = reLabel;

}

void LazySnapping::initNeighboring()
{
	connected = Mat::zeros(n, n, CV_32S);

	for (int x = 0; x < markers.rows; x++)
		for (int y = 0; y < markers.cols; y++)
		{
			int currentLabel = markers.at<int>(x, y);
			for (int k = 0; k < 4; k++)
			{
				int xx = x + r[k];
				int yy = y + c[k];

				if (0 <= xx && xx < markers.rows
					&& 0 <= yy && yy < markers.cols)
				{
					int label = markers.at<int>(xx, yy);
					if (currentLabel != label)
					{
						connected.at<int>(currentLabel, label) = 1;
						connected.at<int>(label, currentLabel) = 1;
					}
				}
			}
		}

}

void LazySnapping::initSegment()
{
	// Get number of segment
	n = getComponentNumber(markers);
	initSuperPixel();
	initNeighboring();
}

void LazySnapping::initWaterShed()
{
	Watershed Watershed;
	Watershed.setSourceImage(src);
	Watershed.setMarkers(markers);
	markers = Watershed.getMarkersLabel();
}

void LazySnapping::initSeeds()
{
	if (isUpdateF)
	{
		k_meanForeground();
	}

	if (isUpdateB)
	{
		k_meanBackground();
	}
}

void LazySnapping::initMarkers()
{
	initWaterShed();
	initSegment();
}

void LazySnapping::getE1(int currentLabel, double * energy)
{
	// average distance
	double df = INFINNITE_MAX;
	double db = INFINNITE_MAX;
	for (int i = 0; i < KF; i++)
	{
		double mindf = norm(centers[currentLabel], foreground_centers.at<Vec3f>(i, 0), NORM_L2);
		df = min(df, mindf);
	}

	for (int i = 0; i < KB; i++)
	{
		double mindb = norm(centers[currentLabel], background_centers.at<Vec3f>(i, 0), NORM_L2);
		db = min(db, mindb);
	}

	energy[0] = df / (db + df);
	energy[1] = db / (db + df);
}

float LazySnapping::colorDistance(Vec3f color1, Vec3f color2)
{
	return (float)sqrt((color1[0] - color2[0])*(color1[0] - color2[0]) +
		(color1[1] - color2[1])*(color1[1] - color2[1]) +
		(color1[2] - color2[2])*(color1[2] - color2[2]));
}

float LazySnapping::getE2(int xlabel, int ylabel)
{
	const float EPSILON = 1;
	float lambda = 10;

	float distance = colorDistance(centers[xlabel], centers[ylabel]);
	return (float)lambda / (EPSILON + distance);
}

void LazySnapping::initGraph()
{
	initMarkers();

	graph = new GraphType(n, 8 * n);

	double e1[2], e2[2];

	graph->add_node(n);

	for (int i = 0; i < n; i++)
	{
		// calculate E1 energy
		if (connectToSource[i])
		{
			e1[0] = 0;
			e1[1] = INFINNITE_MAX;
		}
		else if (connectToSink[i])
		{
			e1[0] = INFINNITE_MAX;
			e1[1] = 0;
		}
		else
		{
			getE1(i, e1);
		}

		graph->add_tweights(i, e1[0], e1[1]);

		for (int j = i + 1; j < n; j++)
			if (connected.at<int>(i, j) == 1)
			{
				float e2 = getE2(i, j);
				graph->add_edge(i, j, e2, e2);
			}
	}
}

void LazySnapping::runMaxFlow()
{
	initSeeds();
	initGraph();
	float flowValue = graph->maxflow();
	getLabellingValue();

	setUpdateF(false);
	setUpdateB(false);
}

void LazySnapping::getLabellingValue()
{
	FLabel.resize(n);

	for (int i = 0; i < n; i++)
		if (graph->what_segment(i) == GraphType::SOURCE)
		{
			FLabel[i] = 1;
		}
		else if (graph->what_segment(i) == GraphType::SINK)
		{
			FLabel[i] = 0;
		}
}

IplImage * LazySnapping::getImageMask()
{
	IplImage* gray = cvCreateImage(src.size(), 8, 1);
	for (int h = 0; h < markers.rows; h++)
	{
		unsigned char* p = (unsigned char*)gray->imageData + h*gray->widthStep;
		for (int w = 0; w < markers.cols; w++)
		{
			int currentLabel = markers.at<int>(h, w);
			if (FLabel[currentLabel] == 1)
			{
				*p = 0;
			}
			else if (FLabel[currentLabel] == 0)
			{
				*p = 255;
			}
			p++;
		}
	}

	return gray;
}

cv::Mat LazySnapping::getImageColor()
{
	cv::Mat showImg = src.clone();
	for (int h = 0; h < markers.rows; h++)
	{
		for (int w = 0; w < markers.cols; w++)
		{
			int currentLabel = markers.at<int>(h, w);
			if (FLabel[currentLabel] == 1)
			{
				showImg.at<Vec3b>(h, w) = { 0, 0, 0 };
			}
			else if (FLabel[currentLabel] == 0)
			{

			}
		}
	}

	return showImg;
}

cv::Mat LazySnapping::changeLabelSegment(int x, int y)
{
	FLabel[markers.at<int>(x, y)] = 1 - FLabel[markers.at<int>(x, y)];
	return getImageColor();
}
