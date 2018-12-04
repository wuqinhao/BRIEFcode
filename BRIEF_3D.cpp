#include "stdafx.h"
#include "Time.h"
#include <math.h>
#include <stdlib.h>
#include <string>  
#include <sstream>
#include <iomanip>
#include <fstream>  
#include <iterator>  
//#include <map>
//#include <unordered_map>
#include <windows.h>
//#include <ppl.h>
//#include <locale>
//#include <codecvt>
//#include <iomanip>
#include <random>
#include <cmath>
#include <omp.h>
#include <iostream>
#include "opencv2/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
//#include <opencv2/ml.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
//#include "svm_multiple.cpp"
#include "SVM.h"
//#include <TCHAR.h>
//#include "C:\Program Files\boost\boost_1_62_0\boost\unordered\unordered_map.hpp"
//#include <direct.h>
//#include <pthread.h>  
#define M_PI       3.14159265358979323846 

//using namespace cv::ml;
using namespace cv;
using namespace std;
//using namespace concurrency;
//using namespace boost::unordered;


class BRIEF
{

public:
	Mat image;
	Mat sample;
	//Mat re_image;
	Mat rlt;
	Mat hist;
	Mat histS;
	Mat groundtruth;
	int window_size;
	int sample_pair;
	int flag_col;
	int flag_row;
	//int * result;
	int result_size;
	Mat result_record;
	//double gKernel[5];
	//map<int, int> NBresult;
	//map<int, int> histogram;
	//map<int, int> histogramS;
	//map<location, float> peak;

	BRIEF();
	BRIEF(int windowsize, int samplenumber);
	BRIEF(int windowsize, int sampleno, Mat InputSample);
	//int calBRIEF(Mat img, int colour);
	int calBRIEFOverlap();
	//the order for RGB is 210 

	//map <int, int> compareHistogramNB(map<int, int> m1, map<int, int> m2);
	int imageSegmentation(Mat test_image, int a_size, Mat texture1, Mat texture2, int w_size, int s_pair, int colour, string loca);
	float imageSegmentation(Mat test_image, Mat goundTruth, int w_size, int s_pair, int a_size, int iter, int it_it, Ptr<SVM> svm, string loca);
	//vector<Mat> combine_picture(Mat src1, Mat src2, int shape);
	Mat combine_picture(Mat src1, Mat src2, Mat mask);
	//Mat combine_picture(Mat src1,  Mat mask);
	//int recreate_NBpicture();
	Mat accHistogram();
	//map<int, int> accHistogram(Mat input);
	Mat accHistogramSum();
	//map<int, int> histogramAdd(map<int, int> in1, map<int, int> in2);
	//map <int, int> histogramMinus(map<int, int> m1, map<int, int> m2);
	int writeFile(Mat sample);
	int writeFile(vector<float> input);
	int writeFile(Mat sample, string loca, string info);
	int writeFile3D(Mat src, int histSize[3], int number, string input);
	int writeFile(float m1, string loca);
	int writeFile(int num, double i);
	int writeFile(string loca, double i);
	int writeFile(map<int, double> m1);
	int writeFile(double data[][80], int row, int col);
	int writeFile();
	int readFile(String loca);
	map<int, int> readFile(String loca, map<int, int> out);
	//double timesTogether(map<int, int> m, int n);
	int cleanMap();
	//Mat calc3D(int histSize[3]);
	//map<location, float> findDominantPeakPoint(Mat hist, int histSize[3]);
	//Mat blur3D(Mat hist, int histSize[3]);
	//unordered_map<int, int> mapConvertToUnordered(map<int, int> m1);
	//Mat mapConvertToMat(map<int, int> m1, int hist_size);
	~BRIEF();

protected:

	//bool judgePeak(map<location, float> peak);
	//map<location, float> combinePeak(map<location, float> peak1, map<location, float> peak2, map<location, float> peak3);
	Mat gaussian_kernel(int sigma, int dim);

	int reflect(int M, int x);
	//double * createDGFilter(double gKernel[5]);
	//double(*orthDGFilter(double gKernel[5], double kernel[][5]))[5];
	void init_descriptor(int window_size, int smaple_pair);
	int cal_window_sample(Mat window, Mat sample, double threshold);
	double cal_threshold(Mat image, int gStd);
};

BRIEF::BRIEF()
{
	window_size = 15;
	sample_pair = 8;
	flag_col = 0;
	flag_row = 0;
	//result = NULL;
	result_size = 0;
	init_descriptor(window_size, sample_pair);
}

BRIEF::BRIEF(int windowsize, int sampleno)
{
	window_size = windowsize;
	sample_pair = sampleno;
	flag_col = 0;
	flag_row = 0;
	//result = NULL;
	result_size = 0;
	init_descriptor(windowsize, sampleno);
}

BRIEF::BRIEF(int windowsize, int sampleno, Mat InputSample)
{
	window_size = windowsize;
	sample_pair = sampleno;
	flag_col = 0;
	flag_row = 0;
	//result = NULL;
	InputSample.copyTo(sample);
	result_size = 0;
}


int BRIEF::calBRIEFOverlap()
{
	//descriptor initialization
	if (sample.empty())
		init_descriptor(window_size, sample_pair);
	
	//cout << "cal BRIEF" << endl;
	//window number and BRIEF vector initialization
	flag_col = image.cols - window_size;
	flag_row = image.rows - window_size;
	//cout << flag_col <<" something"<<image.cols<< endl;

	rlt = Mat::zeros(flag_row, flag_col, DataType<float>::type);

	Mat temp;
	double th = cal_threshold(image, 1);

	
	for (int it = 0; it < (flag_row*flag_col); it++)
	{
		int j = it % flag_col;
		int i = it / flag_col;
		Rect rect(j, i, window_size, window_size);
		image(rect).copyTo(temp);
		rlt.at<float>(i, j) = cal_window_sample(temp, sample, th);
		temp.release();
	}

	temp.release();
	return 0;
}

int BRIEF::imageSegmentation(Mat test_image, int a_size, Mat texture1, Mat texture2, int w_size, int s_pair, int colour, string loca)
/* int BRIEF::imageSegmentation(Mat test_image, int a_size, Mat texture1, Mat texture2, int w_size, int s_pair, int colour, string loca)
Mat test_image: input combined image, called BRIEF::combine_picture first.
int a_size: Neighbourhood size.
Mat texture1: Texture 1 model histogram.
Mat texture2: Texture 2 model histogram.
int w_size: Patch size.
int s_size: Sample pair number.
int colour: Colour channel number.
string loca: Output location
*/
{
	image = test_image;
	calBRIEFOverlap();

	//Initialization: nimages - image number
	int nimages = 1;
	//histogram channel, here is 1.
	int channels = 0;
	int dims = 1;
	//histogram length.
	int nbins = (int)pow(2, s_pair);
	int histSize[] = { nbins };
	//each bin's value range
	float range[] = { 0, nbins };
	const float* histRange[] = { range };
	//other settings
	bool uniform = true; bool accumulate = false;

	//the total number of each local neighbourhood formed histogram.
	int r_rows = rlt.rows - a_size + 1;
	int r_cols = rlt.cols - a_size + 1;
	if (r_cols < 0)
		r_cols = 1;

	/*
	vector<Mat> histogramTable;
	for (int i = 0; i < r_rows; i++)
	{
	for (int j = 0; j < r_cols; j++)
	{
	Mat temp;
	Mat temp_hist;
	Rect rect(j, i, a_size, a_size);
	rlt(rect).copyTo(temp);
	calcHist(&temp, nimages, &channels, Mat(), temp_hist, dims, histSize, histRange, uniform, accumulate);
	histogramTable.push_back(temp_hist);
	temp_hist.release();
	temp.release();
	}
	}
	*/
	//output result collect each pixel's classification.
	Mat collect = Mat::zeros(r_rows, r_cols, DataType<float>::type);

	cout << "data prepared" << endl;
	//cout << histogramTable.size() << endl;

	//r1, r2 are similarity between local histogram with model histograms. 
	//temp stored the local neighbourhood. temp_hist stored the local histogram.
	double r1, r2;

	for (int i = 0; i < /*histogramTable.size()*/(r_rows*r_cols); i++)
		//Concurrency::parallel_for(size_t(0), size_t(/*histogramTable.size()*/(r_rows*r_cols)), [&](size_t i)
	{
		Mat temp;
		Mat temp_hist;
		int col = i % r_cols;
		int row = i / r_cols;

		Rect rect(col, row, a_size, a_size);
		rlt(rect).copyTo(temp);
		calcHist(&temp, nimages, &channels, Mat(), temp_hist, dims, histSize, histRange, uniform, accumulate);

		r1 = compareHist(texture1, temp_hist/*histogramTable[i]*/, CV_COMP_CHISQR);
		r2 = compareHist(texture2, temp_hist/*histogramTable[i]*/, CV_COMP_CHISQR);

		//chisquared test: smaller value achieves higher similarity.
		//texture 1 uses 0 to represent. texture 2 uses 255 to represent.
		if (r1 < r2)
			collect.at<float>(row, col) = 1;
		else
			collect.at<float>(row, col) = 2;

		temp_hist.release();
		temp.release();
	}
	//);
	/*
	double correct = 0.0;
	for (int i = 0; i < collect.rows; i++)
	for (int j = 0; j < collect.cols; j++)
	{
	if (j < (int)collect.cols / 2)
	if (collect.at<double>(i, j) == 0.0)
	correct++;
	else continue;
	else
	if (collect.at<double>(i, j) == 1.0)
	correct++;
	}
	correct /= (collect.rows * collect.cols);
	cout << correct << endl;
	*/

	writeFile(collect, loca, "result");

	collect.release();
	//histogramTable.clear();
	return 0;
}

float BRIEF::imageSegmentation(Mat test_image, Mat GT, int w_size, int s_pair, int a_size, int iter, int it_it, Ptr<SVM> svm, string loca)
/* 
Mat test_image: input combined image, called BRIEF::combine_picture first.
Mat goundTruth: input groundtruth.
int a_size: Neighbourhood size.
Ptr<SVM> svm: pretrained texture classifier.
*/
{
	image = test_image;
	calBRIEFOverlap();
	float TP = 0.0;
	float FP = 0.0;
	float TN = 0.0;
	float FN = 0.0;

	//Initialization: nimages - image number
	int nimages = 1;
	//histogram channel, here is 1.
	int channels = 0;
	int dims = 1;
	//histogram length.
	int nbins = (int)pow(2, s_pair);
	int histSize[] = { nbins };
	//each bin's value range
	float range[] = { 0, nbins };
	const float* histRange[] = { range };
	//other settings
	bool uniform = true; bool accumulate = false;

	//the total number of each local neighbourhood formed histogram.
	int r_rows = rlt.rows - a_size + 1;
	int r_cols = rlt.cols - a_size + 1;
	if (r_cols < 0)
		r_cols = 1;

	//groundTruth adjust point
	int ad_row = ceil((w_size + a_size - 1) / 2.0);
	int ad_col = ceil((w_size + a_size - 1) / 2.0);
	

	//output result collect each pixel's classification.
	Mat collect = Mat::zeros(r_rows, r_cols, DataType<float>::type);
	Rect rect(ad_row, ad_col, r_rows, r_cols);
	Mat groundTruth;
	GT(rect).copyTo(groundTruth);
	cout << collect.size() << "     " << groundTruth.size() << endl;

	std::cout << "data prepared" << endl;
	//cout << histogramTable.size() << endl;
	//temp stored the local neighbourhood. temp_hist stored the local histogram.

	for (int i = 0; i < /*histogramTable.size()*/(r_rows*r_cols); i++)
		//Concurrency::parallel_for(size_t(0), size_t(/*histogramTable.size()(r_rows*r_cols)), [&](size_t i)
	{
		Mat temp;
		Mat temp_hist;
		int col = i % r_cols;
		int row = i / r_cols;

		Rect rect(col, row, a_size, a_size);
		rlt(rect).copyTo(temp);
		calcHist(&temp, nimages, &channels, Mat(), temp_hist, dims, histSize, histRange, uniform, accumulate);
		temp_hist.convertTo(temp_hist, CV_32F);
		transpose(temp_hist, temp_hist);

		//if (i < 5)
		{
			cout << svm->predict(temp_hist) << "     " << groundTruth.at<float>(row, col) << endl;
		}

		/*if (groundTruth.at<float>(row, col) == iter)
		{
			//collect.at<float>(row, col) = 1;
			if (svm->predict(temp_hist) == groundTruth.at<float>(row, col))
				TP++;
			else
				FP++;
		}
		else {
			//collect.at<float>(row, col) = 1;
			if (svm->predict(temp_hist) == groundTruth.at<float>(row, col))
				FN++;
			else
				TN++;
		}*/

		if (svm->predict(temp_hist) == groundTruth.at<float>(row, col))
		{	
			collect.at<float>(row, col) = 1;
			if (groundTruth.at<float>(row, col) == iter)
				TP++;
			else 
				FN++;
		}
		else
		{
			collect.at<float>(row, col) = 0;
			if (groundTruth.at<float>(row, col) == iter)
				TN++;
			else FP++;
		}
		//chisquared test: smaller value achieves higher similarity.
		//texture 1 uses 0 to represent. texture 2 uses 255 to represent.
		temp_hist.release();
		temp.release();
	}
	//);
	//cout << sum(sum(collect)) << endl;
	vector<float> record;
	record.push_back(TP);
	record.push_back(FP);
	record.push_back(TN);
	record.push_back(FN);

	writeFile(record);
	collect.release();
	//histogramTable.clear();
	return TP+FN;
}

Mat BRIEF::accHistogram()
{
	int nimages = 1;
	int channels = 0;
	int dims = 1;
	int nbins = (int)pow(2, sample_pair);
	int histSize[] = { nbins };
	float range[] = { 0, nbins };
	const float* histRange[] = { range };

	bool uniform = true; bool accumulate = false;

	calcHist(&rlt, nimages, &channels, Mat(), hist, dims, histSize, histRange, uniform, accumulate);

	//normalize(hist, hist, 0, 1, NORM_MINMAX, -1, Mat());

	return hist;
}

Mat BRIEF::accHistogramSum()
{
	int nimages = 1;
	int channels = 0;
	int dims = 1;
	int nbins = (int)pow(2, sample_pair);
	int histSize[] = { nbins };
	float range[] = { 0, nbins };
	const float* histRange[] = { range };

	bool uniform = true; bool accumulate = true;

	calcHist(&rlt, nimages, &channels, Mat(), histS, dims, histSize, histRange, uniform, accumulate);

	return histS;
}

int BRIEF::cleanMap()
{
	//histogramS.clear();
	//histogram.clear();
	hist.release();
	histS.release();
	rlt.release();
	//delete result;
	return 0;
}

int BRIEF::readFile(String loca) {
	FileStorage fs;
	fs.open(loca, FileStorage::READ);
	if (!fs.isOpened())
	{
		cerr << "Failed to open " << loca << endl;
		//help(av);
		return 1;
	}

	fs["sample"] >> sample;
	return 0;
}

map<int, int> BRIEF::readFile(String loca, map<int, int> out)
{
	ifstream fin(loca);
	if (!fin)
	{
		std::cerr << "Can't open file " << loca << std::endl;
		std::exit(-1);
	}

	string key, value;
	while (getline(fin, key))
	{
		if (key.empty())
			break;
		stringstream strStream(key);
		strStream >> key >> value;
		//cout << stoi(key) << "\t" << stoi(value) << endl;
		out.insert(pair <int, int>(stoi(key), stoi(value)));
	}
	//cout << sum1.at(stoi(key)) << endl;


	fin.close();
	return out;
}

int BRIEF::writeFile3D(Mat src, int histSize[3], int number, string input)
{
	ofstream in;
	//ostringstream convert;
	stringstream strStream;

	//convert << number;
	strStream << "C:\\Doctor of Philosophy\\" << input << "_" << number << ".txt";
	//string fileName = "C:\\Doctor of Philosophy\\z_" + convert.str() + ".txt";
	in.open(strStream.str(), ios::app);

	for (int z = 0; z < histSize[2]; z++)
	{
		//in << "L" << z << "\t" << endl;
		for (int y = 0; y < histSize[1]; y++)
		{
			for (int x = 0; x < histSize[0]; x++)
				in << src.at<float>(x, y, z) << "\t";
			in << endl;
		}
	}

	in << endl;
	in.close();
	strStream.clear();
	return 0;
}

int BRIEF::writeFile(float m1, string loca)
{
	ofstream in;

	in.open(loca, ios::app);
	in << m1 << endl;
	in.close();
	return 0;
}

int BRIEF::writeFile(Mat s_input)
{
	//FileStorage file("C:\\Doctor of Philosophy\\sample.txt", FileStorage::WRITE);
	//file << "sample" << s_input;

	//map <int, int>::iterator m1_Iter;


	ofstream in;
	in.open("C:\\Doctor of Philosophy\\sample.txt", ios::app);
	for (int i = 0; i < s_input.rows; i++) {
		for (int j = 0; j < s_input.cols; j++)
			in << s_input.at<float>(i, j) << "\t";
		in << endl;
	}
	in.close();
	return 0;
}

int BRIEF::writeFile(vector<float> input)
{
	//FileStorage file("C:\\Doctor of Philosophy\\sample.txt", FileStorage::WRITE);
	//file << "sample" << s_input;

	//map <int, int>::iterator m1_Iter;


	ofstream in;
	in.open("C:\\Doctor of Philosophy\\sample.txt", ios::app);
	for (int i = 0; i < input.size(); i++) {
			in << input[i] << "\t";
	}
	in << endl;
	in.close();
	return 0;
}

int BRIEF::writeFile(Mat s_input, string loca, string info)
{
	
	//FileStorage file(loca, FileStorage::APPEND);
	////file <<info << s_input;
	//file.release();
	
	cout << "here write" << endl;
	ofstream in;

	in.open(loca, ios::app);

	for (int i = 0; i < s_input.rows; i++) {
		for (int j = 0; j < s_input.cols; j++)
			in << s_input.at<float>(i, j) << "\t";
		in << endl;
	}
	in.close();
	
	return 0;
}

int BRIEF::writeFile(map<int, double> m1)
{
	map <int, double>::iterator m1_Iter;
	ofstream in;

	in.open("C:\\Doctor of Philosophy\\histogram.txt", ios::app);

	for (m1_Iter = m1.begin(); m1_Iter != m1.end(); m1_Iter++)
		in << m1_Iter->first << "\t" << m1_Iter->second << endl;
	in << endl;
	in.close();
	return 0;
}

int BRIEF::writeFile()
{
	ofstream in;
	in.open("C:\\Doctor of Philosophy\\histogram.txt", ios::app);
	in << "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx" << endl;
	in << endl;
	return 0;
}

int BRIEF::writeFile(string loca, double i)
{
	ofstream in;

	in.open("C:\\Doctor of Philosophy\\histogram.txt", ios::app);
	in << loca << "\t" << i << endl;
	in.close();
	return 0;
}

int BRIEF::writeFile(int num, double i)
{
	ofstream in;

	in.open("C:\\Doctor of Philosophy\\histogram.txt", ios::app);
	in << (num + 1) << "\t" << i << endl;
	in.close();
	return 0;
}

int BRIEF::writeFile(double data[][80], int row, int col)
{
	ofstream in;

	in.open("C:\\Doctor of Philosophy\\histogram.txt", ios::app);
	for (int i = 0; i < row; i++)
	{
		for (int j = 0; j < col; j++)
		{
			in << data[i][j] << "\t";
		}
		in << endl;
	}

	in.close();
	return 0;
}

void BRIEF::init_descriptor(int window_size, int samplepair)
//Other init type using in initializing descriptor: replace the following line with
//the rng.fill part in the code.
//rng.fill(sample, RNG::UNIFORM, 0, window_size, false); // random 1																		  
//randn(sample, Scalar(0), Scalar(window_size)); // random 2
{
	//initial descriptor compare type
	cout << "window size:" << window_size << "; sample no:" << samplepair << endl;
	cout << "miu:" << (float)window_size / 2.0 << "; std:" << (float)window_size / 30.0 << endl;
	sample = Mat::zeros(samplepair, 6, DataType<float>::type);
	Mat Left = Mat::zeros(samplepair, 2, DataType<float>::type);
	Mat Right = Mat::zeros(samplepair, 2, DataType<float>::type);
	Mat channelL = Mat::zeros(samplepair, 1, DataType<float>::type);
	Mat channelR = Mat::zeros(samplepair, 1, DataType<float>::type);

	//initial Gaussian generator
	random_device rd;
	mt19937 gen(rd());
	normal_distribution<float> dist((float)window_size / 2.0, (float)window_size / 5.0);
	normal_distribution<float> out((float)window_size / 2.0, (float)window_size / 5.0);
	uniform_real_distribution<float> dis(0, 2);

	vector<float> containerL;
	for (int i = 0; i < samplepair * 2; ++i)
	{
		float tmp_value = round(dist(gen));
		if (tmp_value >= 0 && tmp_value < window_size)
		{
			containerL.push_back(tmp_value);
		}
		else
			--i;
	}
	vector<float> containerCL;
	for (int i = 0; i < samplepair; ++i)
	{
		float tmp_value = round(dis(gen));
		containerCL.push_back(tmp_value);
	}

	vector<float> containerR;
	for (int i = 0; i < samplepair * 2; ++i)
	{
		float tmp_value = round(out(gen));
		if (tmp_value >= 0 && tmp_value < window_size)
		{
			containerR.push_back(tmp_value);
		}
		else
			--i;
	}
	vector<float> containerCR;
	for (int i = 0; i < samplepair; ++i)
	{
		float tmp_value = round(dis(gen));
		containerCR.push_back(tmp_value);
	}

	Mat tempLeft = Mat::zeros(samplepair, 2, DataType<float>::type);
	Mat tempRight = Mat::zeros(samplepair, 2, DataType<float>::type);

	memcpy(tempLeft.data, containerL.data(), containerL.size()*sizeof(float));
	memcpy(channelL.data, containerCL.data(), containerCL.size()*sizeof(float));
	
	memcpy(tempRight.data, containerR.data(), containerR.size()*sizeof(float));
	memcpy(channelR.data, containerCR.data(), containerCR.size()*sizeof(float));

	//fix points direction
	for (int i = 0; i < samplepair; i++) {
		float diffx1 = tempLeft.at<float>(i, 0) - window_size / 2;
		float diffy1 = tempLeft.at<float>(i, 1) - window_size / 2;
		float diffx2 = tempRight.at<float>(i, 0) - window_size / 2;
		float diffy2 = tempRight.at<float>(i, 1) - window_size / 2;
		if (sqrt(diffx1*diffx1 + diffy1*diffy1)<=sqrt(diffx2*diffx2 + diffy2*diffy2)) {
			float temp = tempLeft.at<float>(i, 0);
			tempLeft.at<float>(i, 0) = tempRight.at<float>(i, 0);
			tempRight.at<float>(i, 0) = temp;
			float temp1 = tempLeft.at<float>(i, 1);
			tempLeft.at<float>(i, 1) = tempRight.at<float>(i, 1);
			tempRight.at<float>(i, 1) = temp1;
		}	
	}
	
	Mat temp = tempLeft - tempRight;
	Mat res;
	for (int i = 0; i < samplepair; i++) {
		float bearingRadians = atan2(temp.at<float>(i, 0), temp.at<float>(i, 1));
		float bearingDegrees = bearingRadians * (180.0 / M_PI); // convert to degrees
		res.push_back((bearingDegrees > 0.0 ? bearingDegrees : (360.0 + bearingDegrees)));
	}

	Mat des;
	sortIdx(res, res, CV_SORT_EVERY_COLUMN);

	Mat temp1 = Mat::zeros(samplepair, 6, DataType<float>::type);
	hconcat(tempLeft, channelL, Left);
	hconcat(tempRight, channelR, Right);
	hconcat(Left, Right, temp1);

	for (int i = 0; i < samplepair; i++) 
		temp1.row(i).copyTo(sample.row(res.at<int>(i, 0)));
	
	cout << sample << endl;
	temp1.release();
	temp.release();
	res.release();
}

double BRIEF::cal_threshold(Mat image, int gStd)
{
	int r = image.rows;
	int c = image.cols;

	Scalar mean, stddev;
	Mat outmat, dev1;

	GaussianBlur(image, outmat, Size(), gStd, gStd);
	dev1 = image - outmat;

	meanStdDev(dev1, mean, stddev); //standard deviation of image
	double threshold = ceil(stddev.val[0] * 3);

	return threshold;
}

Mat BRIEF::gaussian_kernel(int sigma, int dim)
{
	Mat K(dim, dim, CV_32FC1);
	//gaussian kernel 
	float s2 = (float)2.0 * sigma * sigma;
	for (int i = (-sigma); i <= sigma; i++)
	{
		int m = i + sigma;
		for (int j = (-sigma); j <= sigma; j++)
		{
			int n = j + sigma;
			float v = (float)exp(-(1.0*i*i + 1.0*j*j) / s2);
			K.ptr<float>(m)[n] = v;
		}
	}
	Scalar all = sum(K);
	Mat gaussK;
	K.convertTo(gaussK, CV_32FC1, (1 / all[0]));
	all = sum(gaussK);

	K.release();
	return gaussK;
}

int BRIEF::cal_window_sample(Mat window, Mat sample, double threshold) {
	//return the brief of kernel window
	int sum = 0;
	Vec3b p1, p2;

	for (int i = 0; i < sample_pair; i++)
	{
		p1 = window.at<Vec3b>((int)sample.at<float>(i, 0), (int)sample.at<float>(i, 1));
		p2 = window.at<Vec3b>((int)sample.at<float>(i, 3), (int)sample.at<float>(i, 4));
		if ((float)(p1.val[(int)sample.at<float>(i, 2)] - p2.val[(int)sample.at<float>(i, 5)]) > threshold)//comparison
		{
			sum = sum + (int)pow(2, (sample_pair - i - 1));
		}
	}

	return sum;
}
/*
Mat BRIEF::calc3D(int histSize[3])
{
float range[2] = { 0, 255 };
const float * ranges[3] = { range, range, range };
int channels[3] = { 0, 1, 2 };

Mat hist;
calcHist(&image, 1, channels, Mat(), hist, 3, histSize, ranges);

return hist;
}
*/
int BRIEF::reflect(int M, int x)
{
	if (x < 0)
	{
		return -x - 1;
	}
	if (x >= M)
	{
		return 2 * M - x - 1;
	}
	return x;
}

BRIEF::~BRIEF()
{
	cleanMap();
	image.release();
	sample.release();
	//re_image.release();
	rlt.release();
	groundtruth.release();
	//delete [] result;
	//cout << "Finished clean and destroy instance!" << endl;
}
/*
Mat BRIEF::combine_picture(Mat src1, Mat src2, int shape = 1) //default - circle
{
	
	//initial mask
	Mat mask;
	//rect
	if (shape == 0)
	{
		Mat temp[] = { Mat::zeros(src1.rows, src1.cols / 2, CV_8U), Mat::ones(src1.rows, src1.cols / 2, CV_8U) };
		hconcat(temp, 2, mask);
	}

	//circle
	if (shape == 1)
	{
		Mat temp = Mat::zeros(src1.rows, src1.cols, CV_8U);
		temp.copyTo(mask);
		Point centre = Point((int)src1.cols / 2, (int)src1.rows / 2);
		int radius = (int)(src1.cols - centre.y) / 3;
		circle(mask, centre, radius, (255, 255, 255), -1);//circle
	}
	
	//combine and output to image
	//Mat temp = Mat::zeros(mask.rows, mask.cols, CV_8U);
	//add(mask, temp, groundtruth);
	//temp.release();

	//bitwise_or(src1, mask, image);
	//src2.copyTo(image, mask);
	src2.copyTo(image);
	cout << "combine pic" << endl;
	bitwise_and(src1, src1, image, mask);
	return image;
}
*/
Mat BRIEF::combine_picture(Mat src1, Mat src2,  Mat mask) 
{
	//cout << "combine pic" << endl;
	Mat temp;
	
	src2.copyTo(temp);
	//bitwise_xor(src1, src2, temp, mask);
	bitwise_or(src1, src1, temp, mask);

	return temp;
}

string* re_add(string loca)
{
	stringstream strStream;
	string * location = new string[9119];
	strStream << loca << "*.tif";
	cout << strStream.str() << endl;
	string tempDirectory = strStream.str();
	strStream.str("");
	WIN32_FIND_DATA directoryHandle;
	//memset(&directoryHandle, 0, sizeof(WIN32_FIND_DATA));//perhaps redundant???

	std::wstring wideString = std::wstring(tempDirectory.begin(), tempDirectory.end());
	LPCWSTR directoryPath = wideString.c_str();

	//iterate over all files
	HANDLE handle = FindFirstFile(directoryPath, &directoryHandle);
	int i = 0;
	
	while (INVALID_HANDLE_VALUE != handle/* && i<108*/)
	{
		//skip non-files
		if (!(directoryHandle.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY))
		{
			//convert from WCHAR to std::string
			size_t size = wcslen(directoryHandle.cFileName);
			char * buffer = new char[2 * size + 2];
			wcstombs_s(&size, buffer, (2 * size + 2), directoryHandle.cFileName, (2 * size + 2));
			string file(buffer);
			delete[] buffer;
			strStream << loca << file;
			location[i] = strStream.str();
			strStream.str("");
			cout << location[i] << endl;
			i++;
			//label.push_back(iter);
		}

		if (FALSE == FindNextFile(handle, &directoryHandle))
			break;
	}
	cout << i  <<" Here" << endl;

	
	//close the handle
	FindClose(handle);
	return location;
}
/*
int main(int argc, char* argv[])
{

	Mat image;
	Mat image1;

	stringstream strStream;
	string * location1 = new string[40];
	string * location2 = new string[40];
	string * location3 = new string[40];
	/*
	map<int, int>::iterator h_It;
	map<int, int> m1;
	map<int, int> n1;
	map<int, int> m2;
	map<int, int> n2;
	map<int, int> tex1;
	map<int, int> tex2;

	string it_number[7] = { "01", "02", "08", "10", "16", "19", "24" };
	int cnt = 0;
	int Patch_Size = 33;
	int Sample_Number = 0;
	//if (argc == 2)
	//	Sample_Number = atoi((argv[1]));
	int Neighbour = 3;
	int repeat = 0;

	for (int it_it = 0; it_it < 1; it_it++)
	{
		//int it_it = 3;
		//int sn = 11;
		for (int j = 3; j < 4; j++)
		{
			//int j = 6;
			for (int i = 0; i < 40; i++)
			{
				//strStream << "C:\\Users\\wuq\\Documents\\Exp_distribution\\Result\\Patch_Sample\\11\\T" << it_number[it_it] << it_number[j] << "_" << sn << "_" << i << "segmentation.txt";
				//location3[i] = strStream.str();
				//strStream.str("");
				//cout << location3[i] << endl;

				if (i >= 9)
				{
					strStream << "C:\\Users\\wuq\\Documents\\ImageDatabases\\Texture Datasets\\UIUCTex\\T" << it_number[it_it] << "\\T" << it_number[it_it] << "_" << i + 1 << ".jpg";
					location1[i] = strStream.str();
					strStream.str(""); // clean Stringstream

					strStream << "C:\\Users\\wuq\\Documents\\ImageDatabases\\Texture Datasets\\UIUCTex\\T" << it_number[j] << "\\T" << it_number[j] << "_" << i + 1 << ".jpg";
					location2[i] = strStream.str();
					strStream.str("");
				}
				else
				{
					strStream << "C:\\Users\\wuq\\Documents\\ImageDatabases\\Texture Datasets\\UIUCTex\\T" << it_number[it_it] << "\\T" << it_number[it_it] << "_0" << i + 1 << ".jpg";
					location1[i] = strStream.str();
					strStream.str(""); // clean Stringstream
					strStream << "C:\\Users\\wuq\\Documents\\ImageDatabases\\Texture Datasets\\UIUCTex\\T" << it_number[j] << "\\T" << it_number[j] << "_0" << i + 1 << ".jpg";
					location2[i] = strStream.str();
					strStream.str("");
				}
			}// for i end
			strStream.clear();
			cout << "location prepared" << endl;

			
					//testing
					Mat tempImg, tempImg2;

						tempImg = imread(location1[0], CV_LOAD_IMAGE_COLOR);
						tempImg2 = imread(location2[0], CV_LOAD_IMAGE_COLOR);
						namedWindow("Display window", WINDOW_AUTOSIZE);// Create a window for display.
						cout << "fuck" << endl;
						
						cout << tempImg.cols << endl;
						cout << tempImg2.cols << endl;
						BRIEF tempTry = BRIEF::BRIEF(Patch_Size, Sample_Number);
						Mat I;
						tempTry.combine_picture(tempImg, tempImg2, 1).copyTo(I);
						
						cout << I.cols << endl;
						imshow("Display window", I);
						


					


		}// for compare group j
	}// for setup group it_it

	delete[] location1;
	delete[] location2;
	delete[] location3;

	cv::waitKey(0);
	cin.get();
	return 0;
}*/

//general read image
/*int main(int argc, char* argv[])
{
	Mat image;
	Mat image1;

	stringstream strStream;
	int data_size = 7;
	//string * folder = new string[61];
	string * location1 = new string[data_size];
	//string * location2 = new string[data_size];
	//string * location3 = new string[data_size];
	//string * location4 = new string[data_size];

	//string it_number[10] = { "banded", "blotchy", "braided", "bubbly", "bumpy", "chequered", "cobwebbed", "cracked", "crosshatched", "crystalline" };
	//string it_number[10] = { "fabric", "foliage", "glass", "leather", "metal", "paper", "plastic", "stone", "water", "wood" };
	string it_number[11] = { "aluminium_foil", "brown_bread", "corduroy", "cork", "cotton", "cracker", "lettuce_leaf", "linen", "white_bread", "wood","wool" };
	string scale_number[11] = { "15", "48", "42", "16", "46", "60", "23", "44", "52", "54", "22" };
	string sampleN[4] = { "a", "b", "c", "d" };
	//string it_number[4] = { "c1", "c2", "c3", "c4" };
	//string scale_number[7] = { "i", "l1", "l2", "l3", "l4", "l5", "l8" };
	//string slice[4] = { "", "r60", "r120", "r180" };
	int cnt = 0;
	int Patch_Size = 7;
	int Sample_Number = 5;
	BRIEF temp = BRIEF::BRIEF(Patch_Size, Sample_Number);
	Mat sample_use;
	temp.sample.copyTo(sample_use);

	int Neighbour = 3;
	int repeat = 0;

	int file_no = 0;
	string dirName = "C:\\Users\\wuq\\Documents\\ImageDatabases\\Texture Datasets\\Brodatz Rot";
	//string dirName = "C:\\Users\\wuq\\Documents\\ImageDatabases\\Texture Datasets\\KTH-TIPS2-b";
	//string dirName = "C:\\Users\\wuq\\Documents\\ImageDatabases\\Texture Datasets\\fmd\\image";
	//string dirMask = "C:\\Users\\wuq\\Documents\\ImageDatabases\\Texture Datasets\\fmd\\mask";
	
	
	//cout << folder[204] << endl;
	for (int iter = 0; iter < 13; iter += 1) {
		Mat label;
		strStream << dirName << "\\" << iter+1 << "\\";
		//strStream << dirName<<"sample"<< setw(2)<<setfill('0') << iter+1 <<"\\";
		location1 = re_add(strStream.str());
		strStream.str("");
		//strStream << dirName << "\\" << it_number[iter] << "\\sample_b\\";
		//location2 = re_add(strStream.str());
		//strStream.str("");
		//strStream << dirName << "\\" << it_number[iter] << "\\sample_c\\";
		//location3 = re_add(strStream.str());
		//strStream.str("");
		//strStream << dirName << "\\" << it_number[iter] << "\\sample_d\\";
		//location4 = re_add(strStream.str());
		//strStream.str("");

		for (int i = 0; i < data_size; i++)
			label.push_back(iter+1);
		//int startNo = data_size * iter+480;
		int startNo = 0;
		//cout << location1[startNo] << endl;
		vector<BRIEF> Instance_1;
		vector<BRIEF> Instance_2;
		vector<BRIEF> Instance_3;
		vector<BRIEF> testInstance_4;
		Mat tempImg, tempImg2;
		cout << "Generate Image" << endl;
		for (int i = 0+startNo; i < data_size+startNo; i = i + 4)
		{
			
			//tempImg = imread(location1[i], CV_LOAD_IMAGE_COLOR);
			//tempImg2 = imread(location2[i], CV_LOAD_IMAGE_GRAYSCALE);
			//cout << location1[i] << endl;
			BRIEF tempTry = BRIEF::BRIEF(Patch_Size, Sample_Number, sample_use);
			tempTry.image = imread(location1[i], CV_LOAD_IMAGE_COLOR);
			//tempTry.image = tempTry.combine_picture(tempImg,tempImg2);
			//tempImg.release();
			//tempImg2.release();

			//cout << location1[i+1] << endl;
			//tempImg = imread(location1[i+1], CV_LOAD_IMAGE_COLOR);
			//tempImg2 = imread(location2[i+1], CV_LOAD_IMAGE_GRAYSCALE);
			BRIEF tempTry2 = BRIEF::BRIEF(Patch_Size, Sample_Number, sample_use);
			//tempTry2.image = imread(location2[i], CV_LOAD_IMAGE_COLOR);
			tempTry2.image = imread(location1[i+1], CV_LOAD_IMAGE_COLOR);
			//tempTry2.image = tempTry.combine_picture(tempImg, tempImg2);
			//tempImg.release();
			//tempImg2.release();

			//cout << location1[i + 2] << endl;
			//tempImg = imread(location1[i+2], CV_LOAD_IMAGE_COLOR);
			//tempImg2 = imread(location2[i+2], CV_LOAD_IMAGE_GRAYSCALE);
			BRIEF tempTry3 = BRIEF::BRIEF(Patch_Size, Sample_Number, sample_use);
			//tempTry3.image = imread(location3[i], CV_LOAD_IMAGE_COLOR);
			tempTry3.image = imread(location1[i+2], CV_LOAD_IMAGE_COLOR);
			//tempTry3.image = tempTry.combine_picture(tempImg, tempImg2);
			//tempImg.release();
			//tempImg2.release();

			//cout << location1[i + 3] << endl;
			//tempImg = imread(location1[i+3], CV_LOAD_IMAGE_COLOR);
			//tempImg2 = imread(location2[i+3], CV_LOAD_IMAGE_GRAYSCALE);
			if (i < 4)
			{
				//cout << i + 3 << " sadkjfnsiodfiksjf" << endl;
				BRIEF tempTry4 = BRIEF::BRIEF(Patch_Size, Sample_Number, sample_use);
				//tempTry4.image = imread(location4[i], CV_LOAD_IMAGE_COLOR);
				tempTry4.image = imread(location1[i + 3], CV_LOAD_IMAGE_COLOR);
				testInstance_4.push_back(tempTry4);
				location1[i+3].clear();
			}
			//tempTry4.image = tempTry.combine_picture(tempImg, tempImg2);
			//tempTry.combine_picture(tempImg, tempImg2)[0].copyTo(tempTry.image);
			//namedWindow("Display window", WINDOW_AUTOSIZE);// Create a window for display.
			//imshow("Display window", tempTry3.image);
			//tempImg.release();
			//tempImg2.release();

			Instance_1.push_back(tempTry);
			Instance_2.push_back(tempTry2);
			Instance_3.push_back(tempTry3);
			

			//clean temporay elements
			//tempImg.release();
			//tempImg2.release();
			location1[i].clear();
			//location2[i].clear();
			location1[i+1].clear();
			//location2[i+1].clear();
			location1[i+2].clear();
			////location2[i+2].clear();
			
			//location1[i+3].clear();
			//location3[i].clear();
			//location4[i].clear();
		}
		cout << "test prepared " << Instance_1.size() << endl;
		cout << "test prepared " << Instance_2.size() << endl;
		cout << "test prepared " << Instance_3.size() << endl;
		cout << "test prepared " << testInstance_4.size() << endl;
		

		Mat ch_1;
		Mat ch_2;
		Mat ch_3;
		Mat ch_4;
		
		omp_set_dynamic(0);
#pragma omp parallel sections num_threads(4) firstprivate(Instance_1, Instance_2, Instance_3, testInstance_4)
		{
#pragma omp section
		{
			cout << "thread " << omp_get_thread_num << endl;

			for (int i = 0; i < Instance_1.size(); i++)
			{
				Instance_1[i].calBRIEFOverlap();
				ch_1.push_back(Instance_1[i].accHistogram().reshape(0, 1));
			}//end for
		}//end section 1
#pragma omp section
		{
			cout << "thread " << omp_get_thread_num << endl;

			for (int j = 0; j <Instance_2.size(); j++)
			{
				Instance_2[j].calBRIEFOverlap();
				ch_2.push_back(Instance_2[j].accHistogram().reshape(0, 1));
			}//end for
		}//end section 2
#pragma omp section
		{
			cout << "thread " << omp_get_thread_num << endl;

			for (int k = 0; k < Instance_3.size(); k++)
			{
				Instance_3[k].calBRIEFOverlap();
				ch_3.push_back(Instance_3[k].accHistogram().reshape(0, 1));
			}//end for
		}//end section 3
#pragma omp section
		{
			cout << "thread " << omp_get_thread_num << endl;
			
			for (int l = 0; l <testInstance_4.size(); l++)
			{
				testInstance_4[l].calBRIEFOverlap();
				ch_4.push_back(testInstance_4[l].accHistogram().reshape(0, 1));
			}//end for
		}//end section 1
		}//end parallel sections



		cout << ch_1.cols << " " << ch_1.rows << endl;
		cout << ch_2.cols << endl;
		cout << ch_3.cols << endl;
		cout << ch_4.cols << " " << ch_4.rows << endl;
		strStream << "C:\\Doctor of Philosophy\\brodatzRot_S7,n5\\" << iter << ".xml";
		cout << strStream.str() << endl;
		Instance_1[0].writeFile(ch_1, strStream.str(), "data1");
		Instance_2[0].writeFile(ch_2, strStream.str(), "data2");
		Instance_3[0].writeFile(ch_3, strStream.str(), "data3");
		testInstance_4[0].writeFile(ch_4, strStream.str(), "data4");
		Instance_3[0].writeFile(label.reshape(0, 1), strStream.str(), "label");

		cout << "next loop" << endl;
		strStream.str("");
		ch_1.release();
		ch_2.release();
		ch_3.release();
		ch_4.release();
		vector<BRIEF>().swap(Instance_1);
		vector<BRIEF>().swap(Instance_2);
		vector<BRIEF>().swap(Instance_3);
		vector<BRIEF>().swap(testInstance_4);
		label.release();

		//delete[] location1;
		//delete[] location2;
		//delete[] location3;
		//delete[] location4;
		
	}


	cv::waitKey(0);
	std::cin.get();
	return 0;
}*/

//rotation test
int main(int argc, char* argv[])
{
	Mat image;
	Mat image1;

	stringstream strStream;
	int data_size = 7;
	//string * folder = new string[61];
	string * location1 = new string[data_size];
	//string * location2 = new string[data_size];
	//string * location3 = new string[data_size];
	//string * location4 = new string[data_size];

	//string it_number[10] = { "banded", "blotchy", "braided", "bubbly", "bumpy", "chequered", "cobwebbed", "cracked", "crosshatched", "crystalline" };
	//string it_number[10] = { "fabric", "foliage", "glass", "leather", "metal", "paper", "plastic", "stone", "water", "wood" };
	string it_number[11] = { "aluminium_foil", "brown_bread", "corduroy", "cork", "cotton", "cracker", "lettuce_leaf", "linen", "white_bread", "wood","wool" };
	string scale_number[11] = { "15", "48", "42", "16", "46", "60", "23", "44", "52", "54", "22" };
	string sampleN[4] = { "a", "b", "c", "d" };
	//string it_number[4] = { "c1", "c2", "c3", "c4" };
	//string scale_number[7] = { "i", "l1", "l2", "l3", "l4", "l5", "l8" };
	//string slice[4] = { "", "r60", "r120", "r180" };

	string dirName = "C:\\Users\\wuq\\Documents\\ImageDatabases\\Texture Datasets\\Colored Brodatz";
	strStream << dirName << "\\";
	//strStream << dirName<<"sample"<< setw(2)<<setfill('0') << iter+1 <<"\\";
	location1 = re_add(strStream.str());
	strStream.str("");

	int cnt = 0;
	int Patch_Size = 15;
	int Sample_Number = 10;
	BRIEF temp = BRIEF::BRIEF(Patch_Size, Sample_Number);
	/*
	for (int i = 0; i < 13; i++) {
		MultiTrain inot = MultiTrain::MultiTrain();
		strStream << "C:\\Doctor of Philosophy\\brodatzRot\\" << i << ".xml";
		cout << strStream.str() << endl;
		inot.loadDataSet(strStream.str());
		strStream.str("");
	
		/*temp.readFile("C:\\Doctor of Philosophy\\rotate\\sample.xml");
		//temp.writeFile(temp.sample, "C:\\Doctor of Philosophy\\rotate\\sample.xml", "sample");
		Mat sample_use;
		temp.sample.copyTo(sample_use);
		temp.writeFile(temp.sample);

		sample_use = sample_use.reshape(0, sample_use.rows * 2);
		Mat sample_rot;
		sample_use.colRange(0, 2).copyTo(sample_rot);
		sample_rot -= Patch_Size / 2;

		cout << temp.sample << endl;
		temp.image = imread(location1[0], CV_LOAD_IMAGE_COLOR);

		int i = 90-75;
		//for (int i = 30; i < 45; i = i + 15) {
			float theta = i * M_PI / 180;
			Mat Rot1 = (Mat_<float>(1, 2) << cos(theta), -sin(theta));
			Mat Rot2 = (Mat_<float>(1, 2) << sin(theta), cos(theta));
			//cout << sample_use << endl;
			cout << Rot1 << endl;
			cout << Rot2 << endl;
			Mat result1, result2;

			//sample_use.copyTo(sample_rot);


			for (int i = 0; i < sample_use.rows; i++) {
				multiply(sample_rot.row(i), Rot1, result1);
				multiply(sample_rot.row(i), Rot2, result2);
				add(result1, result2, sample_rot.row(i));
			}

			sample_rot.convertTo(sample_rot, DataType<int>::type);

			Mat mask1 = (sample_rot > Patch_Size / 2);
			Mat mask2 = (sample_rot < -Patch_Size / 2);
			//Mat mask;
			//bitwise_and(mask1, mask2, mask);

			sample_rot.setTo(Patch_Size / 2, mask1);
			sample_rot.setTo(-Patch_Size / 2, mask2);
			sample_rot += Patch_Size / 2;
			sample_rot.convertTo(sample_rot, DataType<float>::type);
			Mat matArray[] = { sample_rot, sample_use.col(2) };
			Mat f_sample;
			hconcat(matArray, 2, f_sample);

			sample_rot = f_sample.reshape(0, Sample_Number);
			//sample_use = sample_use.reshape(0, Sample_Number);

			//cout << sample_use << endl;

			cout << sample_rot << endl;

			sample_rot.copyTo(temp.sample);
			temp.calBRIEFOverlap();
			temp.accHistogram().reshape(0, 1);
			normalize(temp.hist, temp.hist, 1, NORM_L1);
			/*BRIEF tempTry = BRIEF::BRIEF(Patch_Size, Sample_Number, sample_use);
			tempTry.image = imread(location1[0], CV_LOAD_IMAGE_COLOR);
			tempTry.calBRIEFOverlap();
			tempTry.accHistogram().reshape(0, 1);
			normalize(tempTry.hist, tempTry.hist, 1, NORM_L1);

			BRIEF tempTry1 = BRIEF::BRIEF(Patch_Size, Sample_Number, sample_rot);
			tempTry1.image = imread(location1[0], CV_LOAD_IMAGE_COLOR);
			cout << tempTry1.sample << endl;
			tempTry1.calBRIEFOverlap();
			tempTry1.accHistogram().reshape(0, 1);
			normalize(tempTry1.hist, tempTry1.hist, 1, NORM_L1);
			
		strStream << "C:\\Doctor of Philosophy\\brodatzRot\\xxxxxxxx" << i << ".txt";
		cout << strStream.str() << endl;
		//tempTry.writeFile(tempTry.hist);
		temp.writeFile(inot.trainDataMat, strStream.str(), "");
		strStream.str("");
		temp.cleanMap();
	}
		//result1.release();
		//result2.release();
	//}
	*/
	cv::waitKey(0);
	std::cin.get();
	return 0;
}

/*int main(int argc, char* argv[])
{
	Mat image;
	Mat image1;

	stringstream strStream;
	int data_size = 3;
	//string * folder = new string[61];
	string * location1 = new string[data_size];
	//string * location2 = new string[data_size];
	//string * location3 = new string[data_size];
	//string * location4 = new string[data_size];

	//string it_number[10] = { "banded", "blotchy", "braided", "bubbly", "bumpy", "chequered", "cobwebbed", "cracked", "crosshatched", "crystalline" };
	//string it_number[10] = { "fabric", "foliage", "glass", "leather", "metal", "paper", "plastic", "stone", "water", "wood" };
	string it_number[11] = { "aluminium_foil", "brown_bread", "corduroy", "cork", "cotton", "cracker", "lettuce_leaf", "linen", "white_bread", "wood","wool" };
	string scale_number[11] = { "15", "48", "42", "16", "46", "60", "23", "44", "52", "54", "22" };
	string sampleN[4] = { "a", "b", "c", "d" };
	//string it_number[4] = { "c1", "c2", "c3", "c4" };
	//string scale_number[7] = { "i", "l1", "l2", "l3", "l4", "l5", "l8" };
	//string slice[4] = { "", "r60", "r120", "r180" };
	int cnt = 0;
	int Patch_Size = 7;
	int Sample_Number = 5;
	BRIEF temp = BRIEF::BRIEF(Patch_Size, Sample_Number);
	Mat sample_use;
	temp.sample.copyTo(sample_use);

	int Neighbour = 3;
	int repeat = 0;

	int file_no = 0;
	string dirName = "C:\\Users\\wuq\\Documents\\ImageDatabases\\Texture Datasets\\Brodatz Rot";
	//string dirName = "C:\\Users\\wuq\\Documents\\ImageDatabases\\Texture Datasets\\KTH-TIPS2-b";
	//string dirName = "C:\\Users\\wuq\\Documents\\ImageDatabases\\Texture Datasets\\fmd\\image";
	//string dirMask = "C:\\Users\\wuq\\Documents\\ImageDatabases\\Texture Datasets\\fmd\\mask";


	//cout << folder[204] << endl;
	for (int iter = 0; iter < 1; iter += 1) {
		Mat label;
		strStream << dirName << "\\" << iter + 1 << "\\";
		//strStream << dirName<<"sample"<< setw(2)<<setfill('0') << iter+1 <<"\\";
		location1 = re_add(strStream.str());
		strStream.str("");
		//strStream << dirName << "\\" << it_number[iter] << "\\sample_b\\";
		//location2 = re_add(strStream.str());
		//strStream.str("");
		//strStream << dirName << "\\" << it_number[iter] << "\\sample_c\\";
		//location3 = re_add(strStream.str());
		//strStream.str("");
		//strStream << dirName << "\\" << it_number[iter] << "\\sample_d\\";
		//location4 = re_add(strStream.str());
		//strStream.str("");

		for (int i = 0; i < data_size; i++)
			label.push_back(iter + 1);
		//int startNo = data_size * iter+480;
		int startNo = 0;
		//cout << location1[startNo] << endl;
		vector<BRIEF> Instance_1;
		vector<BRIEF> Instance_2;
		vector<BRIEF> Instance_3;
		//vector<BRIEF> testInstance_4;
		Mat tempImg, tempImg2;
		cout << "Generate Image" << endl;
		for (int i = 0 + startNo; i < data_size + startNo; i = i + 4)
		{

			//tempImg = imread(location1[i], CV_LOAD_IMAGE_COLOR);
			//tempImg2 = imread(location2[i], CV_LOAD_IMAGE_GRAYSCALE);
			//cout << location1[i] << endl;
			BRIEF tempTry = BRIEF::BRIEF(Patch_Size, Sample_Number, sample_use);
			tempTry.image = imread(location1[i], CV_LOAD_IMAGE_COLOR);
			//tempTry.image = tempTry.combine_picture(tempImg,tempImg2);
			tempImg.release();
			tempImg2.release();

			//cout << location1[i+1] << endl;
			//tempImg = imread(location1[i+1], CV_LOAD_IMAGE_COLOR);
			//tempImg2 = imread(location2[i+1], CV_LOAD_IMAGE_GRAYSCALE);
			BRIEF tempTry2 = BRIEF::BRIEF(Patch_Size, Sample_Number, sample_use);
			//tempTry2.image = imread(location2[i], CV_LOAD_IMAGE_COLOR);
			tempTry2.image = imread(location1[i + 1], CV_LOAD_IMAGE_COLOR);
			//tempTry2.image = tempTry.combine_picture(tempImg, tempImg2);
			tempImg.release();
			tempImg2.release();

			//cout << location1[i + 2] << endl;
			//tempImg = imread(location1[i+2], CV_LOAD_IMAGE_COLOR);
			//tempImg2 = imread(location2[i+2], CV_LOAD_IMAGE_GRAYSCALE);
			BRIEF tempTry3 = BRIEF::BRIEF(Patch_Size, Sample_Number, sample_use);
			//tempTry3.image = imread(location3[i], CV_LOAD_IMAGE_COLOR);
			tempTry3.image = imread(location1[i + 2], CV_LOAD_IMAGE_COLOR);
			//tempTry3.image = tempTry.combine_picture(tempImg, tempImg2);
			tempImg.release();
			tempImg2.release();

			//cout << location1[i + 3] << endl;
			//tempImg = imread(location1[i+3], CV_LOAD_IMAGE_COLOR);
			//tempImg2 = imread(location2[i+3], CV_LOAD_IMAGE_GRAYSCALE);

			//BRIEF tempTry4 = BRIEF::BRIEF(Patch_Size, Sample_Number, sample_use);
			//tempTry4.image = imread(location4[i], CV_LOAD_IMAGE_COLOR);
			//tempTry4.image = imread(location1[i+3], CV_LOAD_IMAGE_COLOR);

			//tempTry4.image = tempTry.combine_picture(tempImg, tempImg2);
			//tempTry.combine_picture(tempImg, tempImg2)[0].copyTo(tempTry.image);
			//namedWindow("Display window", WINDOW_AUTOSIZE);// Create a window for display.
			//imshow("Display window", tempTry3.image);
			//tempImg.release();
			//tempImg2.release();

			Instance_1.push_back(tempTry);
			Instance_2.push_back(tempTry2);
			Instance_3.push_back(tempTry3);
			//testInstance_4.push_back(tempTry4);

			//clean temporay elements
			//tempImg.release();
			//tempImg2.release();
			location1[i].clear();
			//location2[i].clear();
			location1[i + 1].clear();
			//location2[i+1].clear();
			location1[i + 2].clear();
			////location2[i+2].clear();
			//location1[i+3].clear();
			//location2[i+3].clear();
			//location3[i].clear();
			//location4[i].clear();
		}
		cout << "test prepared " << Instance_1.size() << endl;
		cout << "test prepared " << Instance_2.size() << endl;
		cout << "test prepared " << Instance_3.size() << endl;
		//cout << "test prepared " << testInstance_4.size() << endl;


		Mat ch_1;
		Mat ch_2;
		Mat ch_3;

		Instance_1[0].calBRIEFOverlap();
		ch_1.push_back(Instance_1[0].accHistogram().reshape(0, 1));

		Instance_2[0].calBRIEFOverlap();
		ch_2.push_back(Instance_2[0].accHistogram().reshape(0, 1));

		Instance_3[0].calBRIEFOverlap();
		ch_3.push_back(Instance_3[0].accHistogram().reshape(0, 1));


		strStream << "C:\\Doctor of Philosophy\\brodatzRot_S7,n5\\" << iter << ".xml";
		cout << strStream.str() << endl;
		Instance_1[0].writeFile(ch_1, strStream.str(), "data1");
		Instance_2[0].writeFile(ch_2, strStream.str(), "data2");
		Instance_3[0].writeFile(ch_3, strStream.str(), "data3");
		Instance_3[0].writeFile(label.reshape(0, 1), strStream.str(), "label");

		cout << "next loop" << endl;
		strStream.str("");
		ch_1.release();
		ch_2.release();
		ch_3.release();
		//ch_4.release();
		vector<BRIEF>().swap(Instance_1);
		vector<BRIEF>().swap(Instance_2);
		vector<BRIEF>().swap(Instance_3);
		//vector<BRIEF>().swap(testInstance_4);
		label.release();

		//delete[] location1;
		//delete[] location2;
		//delete[] location3;
		//delete[] location4;

	}


	cv::waitKey(0);
	std::cin.get();
	return 0;
}

/*int main(int argc, char* argv[])
{
	Mat image;
	Mat image1;
	stringstream strStream;
	int data_size = 108;
	string * location1Train = new string[data_size * 3];
	string * location1Test = new string[data_size];
	string * location2Train = new string[data_size * 3];
	string * location2Test = new string[data_size];


	string it_number[11] = { "aluminium_foil",
		"brown_bread",
		"corduroy",
		"cork",
		"cotton",
		"cracker",
		"lettuce_leaf",
		"linen",
		"white_bread",
		"wood",
		"wool" };
	string scale_number[11] = { "15", "48", "42", "16","46", "60", "23","44", "52", "54", "22" };
	string sampleN[4] = { "a","b","c","d" };

	int cnt = 0;
	int Patch_Size = 15;
	int half_PS = Patch_Size / 2 + 1;
	int Sample_Number = 10;
	BRIEF temp = BRIEF::BRIEF(Patch_Size, Sample_Number);
	Mat sample_use;
	temp.sample.copyTo(sample_use);

	int Neighbour = 3;
	int repeat = 0;

	int file_no = 0;

	vector<BRIEF> instance;
	vector<Mat> maskstack;
	MultiTrain inot;
	Mat recordResult = Mat::zeros(data_size*5,1, DataType<float>::type);
	int rR = 0;
	inot.loadModel("C:\\Doctor of Philosophy\\CBRIEF_result\\seg_svm_model_linear.xml");
	//Mat train, label;
	for (int iter = 0; iter < 1; iter++)
		for (int it_it = iter + 1; it_it < 2; it_it++)
		{
			Mat labelT;
			int j = 1;
			//test section by sample d
			for (int i = 0; i < data_size; i++)
			{
				j = i / 12 + 2;
				strStream << "C:\\Users\\wuq\\Documents\\ImageDatabases\\Texture Datasets\\KTH-TIPS2-b\\" << it_number[iter] << "\\sample_" << sampleN[3] << "\\" << scale_number[iter] << sampleN[3] << "-scale_" << j << "_im_" << (i % 12 + 1) << "_col.png";
				location1Test[i] = strStream.str();
				strStream.str(""); // clean Stringstream
				labelT.push_back(iter + 1);
			}
			j = 1;
			for (int i = 0; i < data_size; i++)
			{
				j = i / 12 + 2;
				strStream << "C:\\Users\\wuq\\Documents\\ImageDatabases\\Texture Datasets\\KTH-TIPS2-b\\" << it_number[it_it] << "\\sample_" << sampleN[3] << "\\" << scale_number[it_it] << sampleN[3] << "-scale_" << j << "_im_" << (i % 12 + 1) << "_col.png";
				location2Test[i] = strStream.str();
				strStream.str(""); // clean Stringstream
				labelT.push_back(it_it + 1);
			}

			Mat whole_mask;
			strStream << "C:\\Users\\wuq\\Documents\\ImageDatabases\\DRIVE\\DRIVE\\test\\1st_manual\\0" << iter << "_manual1.jpg";
			if (iter == 0)
				whole_mask = imread("C:\\Users\\wuq\\Documents\\ImageDatabases\\DRIVE\\DRIVE\\test\\1st_manual\\01_manual1.jpg", CV_LOAD_IMAGE_GRAYSCALE);
			else
				whole_mask = imread(strStream.str(), CV_LOAD_IMAGE_GRAYSCALE);

			cout << strStream.str() << whole_mask.size() << endl;
			strStream.str(""); // clean Stringstream
			whole_mask = whole_mask / 255;

			Mat tempImg, tempImg2;
			Mat mask, maskTemp;
			//test prepare
			for (int repeat = 0; repeat < 20; repeat++)
			for (int i = data_size/4*3+1; i < data_size; i = i + 1)
			{
				tempImg = imread(location1Test[i], CV_LOAD_IMAGE_COLOR);
				tempImg2 = imread(location2Test[i], CV_LOAD_IMAGE_COLOR);
				//cout << location2Test[i] << endl;

				BRIEF tempTry = BRIEF::BRIEF(Patch_Size, Sample_Number, sample_use);
				unsigned seed = time(0);
				srand(seed);
				int random_x = rand() % (whole_mask.cols - 200);
				int random_y = rand() % (whole_mask.rows - 200);
				//cout << random_x << "  " << random_y << endl;

				Rect rect(random_x, random_y, 200, 200);
				whole_mask(rect).copyTo(maskTemp);

				Mat g1, g2;
				g1 = Mat::ones(200, 200, DataType<float>::type)*iter;
				g2 = Mat::ones(200, 200, DataType<float>::type)*it_it;
				tempTry.combine_picture(tempImg, tempImg2, maskTemp).copyTo(tempTry.image);
				tempTry.combine_picture(g1, g2, maskTemp).copyTo(mask);

				instance.push_back(tempTry);
				maskstack.push_back(mask);
				recordResult.at<float>(rR,0) = tempTry.imageSegmentation(tempTry.image, mask, Patch_Size, Sample_Number, 11, iter, it_it, inot.svm, "");
				cout << recordResult.at<float>(rR, 0) << endl;rR++;
				//clean temporay elements
				tempImg.release();
				tempImg2.release();
				mask.release();
				maskTemp.release();
			}

			cout << "test prepared " << instance.size() << endl;
			//instance[0].writeFile(recordResult);
			//namedWindow("Display window", WINDOW_AUTOSIZE);// Create a window for display.
			//imshow("Display window", maskstack[0]*255);
			//cv::waitKey(0);
			//std::cin.get();
			return 0;
		}
}
			
			
			
			//inot.svm->predict();
			

			vector<BRIEF> testInstance_1(instance.begin(), instance.begin() + 20);
			vector<BRIEF> testInstance_2(instance.begin() + 21, instance.begin() + 40);
			vector<BRIEF> testInstance_3(instance.begin() + 41, instance.begin() + 60);
			vector<BRIEF> testInstance_4(instance.begin() + 61, instance.end());

			Mat ch_1;
			Mat ch_2;
			Mat ch_3;
			Mat ch_4;
			omp_set_dynamic(0);
			#pragma omp parallel sections num_threads(4) firstprivate(testInstance_1, testInstance_2, testInstance_3, testInstance_4)
			{
			#pragma omp section
			{
			cout << "thread " << omp_get_thread_num << endl;

			for (int i = 0; i < 41; i++)
			{
			testInstance_1[i].calBRIEFOverlap();
			ch_1.push_back(testInstance_1[i].accHistogram().reshape(0, 1));
			}//end for
			}//end section 1
			#pragma omp section
			{
			cout << "thread " << omp_get_thread_num << endl;

			for (int j = 0; j < 41; j++)
			{
			testInstance_2[j].calBRIEFOverlap();
			ch_2.push_back(testInstance_2[j].accHistogram().reshape(0, 1));
			}//end for
			}//end section 2
			#pragma omp section
			{
			cout << "thread " << omp_get_thread_num << endl;

			for (int k = 0; k < 40; k++)
			{
			testInstance_3[k].calBRIEFOverlap();
			ch_3.push_back(testInstance_3[k].accHistogram().reshape(0, 1));
			}//end for
			}//end section 3
			#pragma omp section
			{
			cout << "thread " << omp_get_thread_num << endl;

			for (int l = 0; l < 40; l++)
			{
			testInstance_4[l].calBRIEFOverlap();
			ch_4.push_back(testInstance_4[l].accHistogram().reshape(0, 1));
			}//end for
			}//end section 1
			}//end parallel sections
			
		}


	cv::waitKey(0);
	std::cin.get();
	return 0;
}
}*/