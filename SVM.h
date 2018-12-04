#pragma once
#include "opencv2\opencv.hpp"
//#include "cv.h"
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include "opencv2/imgcodecs.hpp"
#include <opencv2/highgui.hpp>
#include <opencv2/ml.hpp>

using namespace cv;
using namespace cv::ml;
using namespace std;

class MultiTrain {

public:

	MultiTrain();
	MultiTrain(Mat traindata, Mat response);
	MultiTrain(Mat traindata, Mat response, Mat testdata, Mat tlabel);

	~MultiTrain();

	Ptr<SVM> svm = SVM::create();
	Ptr<ml::KNearest> knn = ml::KNearest::create();
	//CvMat *trainData;
	Mat trainDataMat;
	Mat labelMat;
	Mat testDataMat;
	Mat testResponse;

	CvTermCriteria criteria;

	double ACC;

	void loadDataSet(string filename);

	void loadDataSet(string filename, int a);

	void trainModel(string outputdir);

	//void trainModel(Mat input, Mat response);

	void loadModel(string filename);
	
	void cleanMap();

	void testModel(string filename);

	void createConfusionMatrix(string filename);

	void createConfusionMatrix(int a);

	void calculateTrainDatacount(int tab[]);

	void prepareSets(int class_counts[], int counts[], int part, int parts, CvMat *traindata, CvMat *trainlabels, CvMat *testdata, CvMat *testlabels);

	void performCrossValidation(int parts);

	float calcSum(int *tab, int n);

	float MultiTrain::calcSum(Mat tab, int n);

	float calcSum(int *tab, int idx1, int idx2);

	Mat  shuffleRows(const cv::Mat& matrix, Mat& label);
};