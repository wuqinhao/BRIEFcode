#pragma once
#include "stdafx.h"
#include "Time.h"
#include <math.h>
#include <stdlib.h>
#include <string>  
#include <sstream>
#include <iomanip>
#include <fstream>  
#include <iterator>  
#include <windows.h>
#include <random>
#include <cmath>
#include <omp.h>
#include <iostream>
#include "opencv2/core.hpp"
#include <opencv2/core/core.hpp>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include <opencv2/opencv.hpp>

#include "SVM.h"

#define M_PI       3.14159265358979323846 

using namespace cv;
using namespace std;

class BRIEF
{

public:
	Mat image;
	Mat sample;
	Mat rlt;
	Mat hist; //single image histogram
	Mat histS;	//class histogram
	Mat groundtruth;
	int window_size;
	int sample_pair;
	int flag_col;
	int flag_row;
	int result_size;
	Mat result_record;

	BRIEF();
	BRIEF(int windowsize, int samplenumber);
	BRIEF(int windowsize, int sampleno, Mat InputSample);
	int calBRIEFOverlap();
	//the order for RGB is BGR

	//single channel chi-square segmentation
	int imageSegmentation(Mat test_image, int a_size, Mat texture1, Mat texture2, int w_size, int s_pair, int colour, string loca);
	//SVM classifier segmentation
	float imageSegmentation(Mat test_image, Mat goundTruth, int w_size, int s_pair, int a_size, int iter, int it_it, Ptr<SVM> svm, string loca);
	Mat combine_picture(Mat src1, Mat src2, Mat mask); //combine texture together
	Mat accHistogram();
	Mat accHistogramSum();
	int writeFile(Mat sample);
	int writeFile(vector<float> input);
	int writeFile(Mat sample, string loca, string info);
	int writeFile3D(Mat src, int histSize[3], int number, string input);
	int writeFile(float m1, string loca);
	int writeFile(int num, double i);
	int writeFile(string loca, double i);
	int writeFile(map<int, double> m1);
	int writeFile();
	int readFile(String loca);
	map<int, int> readFile(String loca, map<int, int> out);
	int cleanMap();
	~BRIEF();

protected:

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

	//cout << rlt.size() << endl;
	//the total number of each local neighbourhood formed histogram.
	int r_rows = rlt.rows - a_size + 1;
	int r_cols = rlt.cols - a_size + 1;
	if (r_cols < 0)
		r_cols = 1;

	//groundTruth adjust point
	int ad_row = ceil((w_size + a_size - 1) / 2.0);
	int ad_col = ceil((w_size + a_size - 1) / 2.0);

	cout << ad_row << endl;
	//output result collect each pixel's classification.
	Mat collect(Size(r_cols, r_rows), CV_8UC3, Scalar(0));
	cout << collect.size();
	//Mat check = Mat::zeros(r_rows, r_cols, DataType<float>::type);
	Rect rect(ad_col, ad_row, r_cols, r_rows);
	Mat groundTruth;
	GT(rect).copyTo(groundTruth);
	cout << "     " << groundTruth.size() << endl;
	//namedWindow("1", WINDOW_AUTOSIZE);// Create a window for display.
	//imshow("1", (groundTruth*255));

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

		//check.at<float>(col, row) = rlt.at<float>(col, row);
		Rect rect(col, row, a_size, a_size);
		rlt(rect).copyTo(temp);
		calcHist(&temp, nimages, &channels, Mat(), temp_hist, dims, histSize, histRange, uniform, accumulate);
		temp_hist.convertTo(temp_hist, CV_32F);

		transpose(temp_hist, temp_hist);
		//temp_hist.colRange(1, 1024).copyTo(temp_hist);
		//if (i > (r_rows*r_cols-50))
		//{
		//	cout << svm->predict(temp_hist) << "     " << groundTruth.at<float>(row, col) << endl;
		//}

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
		//if ((int)svm->predict(temp_hist)==1)
		//	collect.at<int>(row, col) = 0;
		//else collect.at<int>(row, col) = 1;
		if (svm->predict(temp_hist) == groundTruth.at<uchar>(row, col))
		{
			//collect.at<float>(row, col) = 1;
			if (groundTruth.at<uchar>(row, col) == iter) {
				TP++;
				Vec3b color = collect.at<Vec3b>(row, col);
				color[0] = 255;
				color[1] = 255;
				color[2] = 255;
				collect.at<Vec3b>(row, col) = color;
			}
			else {
				FN++;

			}
		}
		else
		{
			//collect.at<float>(row, col) = 0;
			if (groundTruth.at<uchar>(row, col) == iter) {
				TN++;
				Vec3b color = collect.at<Vec3b>(row, col);
				color[0] = 0;
				color[1] = 0;
				color[2] = 255;
				collect.at<Vec3b>(row, col) = color;
			}
			else {
				FP++;
				Vec3b color = collect.at<Vec3b>(row, col);
				color[0] = 0;
				color[1] = 255;
				color[2] = 0;
				collect.at<Vec3b>(row, col) = color;
			}
		}
		//chisquared test: smaller value achieves higher similarity.
		//texture 1 uses 0 to represent. texture 2 uses 255 to represent.
		temp_hist.release();
		temp.release();
	}
	//cout << "sum: "<<sum(collect) << endl;
	collect *= 255;
	namedWindow("3", WINDOW_NORMAL);// Create a window for display.
	resizeWindow("3", 600, 600);
	imshow("3", collect * 255);
	//namedWindow("4", WINDOW_AUTOSIZE);// Create a window for display.
	//imshow("4", check*255);

	//);
	//cout << sum(sum(collect)) << endl;
	vector<float> record;
	record.push_back(TP); cout << " " << TP << " ";
	record.push_back(FP); cout << FP << " ";
	record.push_back(TN); cout << TN << " ";
	record.push_back(FN); cout << FN << endl;

	//writeFile(check);
	collect.release();
	//histogramTable.clear();
	return TP + FN;
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

	hist.release();
	histS.release();
	rlt.release();

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
	FileStorage file("C:\\Doctor of Philosophy\\sample.xml", FileStorage::WRITE);
	file << "sample" << s_input;
	file.release();

	//ofstream in;
	//in.open("C:\\Doctor of Philosophy\\sample.txt", ios::app);
	//for (int i = 0; i < s_input.rows; i++) {
	//	for (int j = 0; j < s_input.cols; j++)
	//		in << s_input.at<float>(i, j) << "\t";
	//	in << endl;
	//}
	//in.close();
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

	FileStorage file(loca, FileStorage::APPEND);
	file << info << s_input;
	file.release();
	/*
	cout << "here write" << endl;
	ofstream in;

	in.open(loca, ios::app);

	for (int i = 0; i < s_input.rows; i++) {
	for (int j = 0; j < s_input.cols; j++)
	in << s_input.at<float>(i, j) << "\t";
	in << endl;
	}
	in.close();
	*/
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
	in << sample << endl;
	in << endl;
	in.close();
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
	//normal_distribution<float> dis((float)1.0, (float)3.0 / 5.0);

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
		if (tmp_value >= 0 && tmp_value < 3)
			containerCL.push_back(tmp_value);
		else --i;
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
		if (tmp_value >= 0 && tmp_value < 3)
			containerCR.push_back(tmp_value);
		else --i;
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
		if (sqrt(diffx1*diffx1 + diffy1*diffy1) <= sqrt(diffx2*diffx2 + diffy2*diffy2)) {
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
	double threshold = ceil(stddev.val[0] * 3) + abs(mean.val[0]);

	return threshold;
}

int BRIEF::cal_window_sample(Mat window, Mat sample, double threshold) {
	//return the brief of kernel window
	threshold = 0;
	int sum = 0;
	Vec3b p1, p2;

	for (int i = 0; i < sample_pair; i++)
	{
		p1 = window.at<Vec3b>((int)sample.at<float>(i, 0), (int)sample.at<float>(i, 1));
		p2 = window.at<Vec3b>((int)sample.at<float>(i, 3), (int)sample.at<float>(i, 4));
		float result = p1.val[(int)sample.at<float>(i, 2)] - p2.val[(int)sample.at<float>(i, 5)];
		if (result > threshold)//comparison
		{
			sum = sum + (int)pow(2, (sample_pair - i - 1));
		}
	}

	return sum;
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
//old version
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

Mat BRIEF::combine_picture(Mat src1, Mat src2, Mat mask)
//opencv rewrite version
{
	//cout << "combine pic" << endl;
	Mat temp;

	src2.copyTo(temp);
	//bitwise_xor(src1, src2, temp, mask);
	bitwise_or(src1, src1, temp, mask);

	return temp;
}