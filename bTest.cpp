#include "stdafx.h"
#include "Time.h"
#include <iostream>
#include "opencv2/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <math.h>
#include <stdlib.h>
#include <string>  
#include <sstream>  
#include <fstream>  
#include <iterator>  
#include <map>
#include <unordered_map>
#include <windows.h>
#include <ppl.h>
#include <iomanip>
#include <random>
#include <cmath>
#include <omp.h>
#include <TCHAR.h>
#define M_PI       3.14159265358979323846 

using namespace cv;
using namespace std;

typedef std::tuple<int, int, int> location;

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
	int result_size;

	BRIEF();
	BRIEF(Mat img);
	BRIEF(Mat img, Mat sp);
	BRIEF(Mat img1, Mat img2, String sp_loca);
	int calBRIEF(int window_size, int sample_pair, int colour);
	int calBRIEFOverlap(int w_size, int s_pair, int colour);
	//the order for RGB is 210 

	int imageSegmentation(Mat test_image, int a_size, Mat texture1, Mat texture2, int w_size, int s_pair, int colour, string loca);
	Mat combine_picture(Mat src1, Mat src2, int shape);
	Mat accHistogram();
	Mat accHistogramSum();
	int writeFile(Mat sample);
	int writeFile(Mat sample, string loca);
	int writeFile3D(Mat src, int histSize[3], int number, string input);
	int writeFile(map<int, int> m1, string loca);
	int writeFile(int num, double i);
	int writeFile(string loca, double i);
	int writeFile(map<int, double> m1);
	int writeFile(double data[][80], int row, int col);
	int writeFile();
	int readFile(String loca);
	map<int, int> readFile(String loca, map<int, int> out);
	int cleanMap();
	~BRIEF();

protected:

	Mat gaussian_kernel(int sigma, int dim);
	int reflect(int M, int x);
	void init_descriptor(int window_size, int smaple_pair);
	int cal_window_sample(Mat window, Mat sample, double threshold, int colour);
	double cal_threshold(Mat image, int channel, int gStd);
};

BRIEF::BRIEF(Mat img)
{
	image = img;
	window_size = 7;
	sample_pair = 9;
	flag_col = 0;
	flag_row = 0;
	result_size = 0;

}

BRIEF::BRIEF(Mat img, Mat sp)
{
	image = img;
	sample = sp;
	window_size = 7;
	sample_pair = 9;
	flag_col = 0;
	flag_row = 0;
	result_size = 0;

}

BRIEF::BRIEF(Mat img1, Mat img2, String sp_loca)
{
	image = combine_picture(img1, img2, 1);
	readFile(sp_loca);
	window_size = 7;
	sample_pair = 9;
	flag_col = 0;
	flag_row = 0;
	result_size = 0;
}

BRIEF::BRIEF()
{
	window_size = 7;
	sample_pair = 9;
	flag_col = 0;
	flag_row = 0;
	result_size = 0;
}

int BRIEF::calBRIEF(int w_size, int s_pair, int colour)
{
	window_size = w_size;
	sample_pair = s_pair;

	//descriptor initialization
	if (sample.empty())
		if (pow(window_size, 2) > (sample_pair * 2)) //check input value
		{
			/*sample = */init_descriptor(window_size, sample_pair);
		}
		else
		{
			cout << "Sample pair number too large";
			return 1;
		}

	//window number and BRIEF vector initialization
	flag_col = image.cols / window_size;
	flag_row = image.rows / window_size;
	result_size = (int)flag_col * flag_row;
	rlt = Mat::zeros(flag_row, flag_col, DataType<float>::type);
	//result = new int[result_size];
	//int * ptr = result;

	Mat temp;
	int cColour = 0;
	if (colour == 0)
		cColour = 3;
	double th = cal_threshold(image, cColour, 1);

	for (int i = 0; i < flag_row; i++)
		for (int j = 0; j < flag_col; j++)
		{
			Rect rect(j * window_size, i * window_size, window_size, window_size);
			image(rect).copyTo(temp);
			//*ptr = cal_window_sample(temp, sample, th, colour);
			//rlt.at<float>(i, j) = *ptr;
			rlt.at<float>(i, j) = cal_window_sample(temp, sample, th, colour);
			//ptr++;
			temp.release();
		}
	//delete ptr;

	temp.release();
	return 0;
}

int BRIEF::calBRIEFOverlap(int w_size, int s_pair, int colour)
{
	window_size = w_size;
	sample_pair = s_pair;

	//descriptor initialization
	if (sample.empty())
		/*sample = */init_descriptor(window_size, sample_pair);

	//window number and BRIEF vector initialization
	flag_col = image.cols - window_size + 1;
	flag_row = image.rows - window_size + 1;

	result_size = (int)flag_col * flag_row;
	rlt = Mat::zeros(flag_row, flag_col, DataType<float>::type);
	//result = new int[result_size];
	//int * ptr = result;

	Mat temp;
	int cColour = 0;
	if (colour == 0)
		cColour = 3;
	double th = cal_threshold(image, cColour, 1);

	for (int i = 0; i < flag_row; i++)
		for (int j = 0; j < flag_col; j++)
		{
			Rect rect(j, i, window_size, window_size);
			image(rect).copyTo(temp);
			//*ptr = cal_window_sample(temp, sample, th, colour);
			//rlt.at<float>(i, j) = *ptr;
			rlt.at<float>(i, j) = cal_window_sample(temp, sample, th, colour);
			//ptr++;
			temp.release();
		}
	//delete ptr;

	temp.release();
	return 0;
}
/*
map <int, int> BRIEF::compareHistogramNB(map<int, int> m1, map<int, int> n2)
{
	map <int, int>::iterator m1_Iter;
	map <int, int>::iterator m2_Iter;

	//Find the same bins from the model histogram
	for (m1_Iter = m1.begin(); m1_Iter != m1.end(); m1_Iter++)
	{
		m2_Iter = n2.find(m1_Iter->first);
		if (m2_Iter != n2.end())
		{
			double p1 = 1.0 / 1.0 + ((double)m1_Iter->second / (double)m2_Iter->second);
			double p2 = 1.0 / 1.0 + ((double)m2_Iter->second / (double)m1_Iter->second);
			if (p1 > p2)
				NBresult.insert(pair<int, int>(m1_Iter->first, 1));
			else
				NBresult.insert(pair<int, int>(m2_Iter->first, 0));
		}
		else
		{
			NBresult.insert(pair<int, int>(m1_Iter->first, 0));
		}
	}

	for (m2_Iter = n2.begin(); m2_Iter != n2.end(); m2_Iter++)
	{
		m1_Iter = NBresult.find(m2_Iter->first);
		if (m1_Iter == NBresult.end())
			NBresult.insert(pair<int, int>(m2_Iter->first, 0));
	}

	int total = pow(2, sample_pair);

	for (int i = 0; i <= total; i++)
	{
		m1_Iter = NBresult.find(i);
		if (0 == i)
			m1_Iter->second = 2;
		if (m1_Iter == NBresult.end())
			NBresult.insert(pair<int, int>(i, 2));
	}

	return NBresult;
}
*/
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
	calBRIEFOverlap(w_size, s_pair, colour);

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
	Mat collect = Mat::zeros(r_rows, r_cols, DataType<double>::type);

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
			collect.at<double>(row, col) = 1;
		else
			collect.at<double>(row, col) = 2;

		temp_hist.release();
		temp.release();
	}
	//);

	writeFile(collect, loca);
	
	collect.release();
	//histogramTable.clear();
	return 0;
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

int BRIEF::writeFile(map<int, int> m1, string loca)
{
	map <int, int>::iterator m1_Iter;
	ofstream in;

	in.open(loca, ios::app);

	for (m1_Iter = m1.begin(); m1_Iter != m1.end(); m1_Iter++)
		in << m1_Iter->first << "\t" << m1_Iter->second << endl;
	in << endl;
	in.close();
	return 0;
}

int BRIEF::writeFile(Mat s_input)
{
	FileStorage file("C:\\Doctor of Philosophy\\sample.txt", FileStorage::WRITE);
	file << "sample" << s_input;

	//map <int, int>::iterator m1_Iter;

	/*
	ofstream in;
	in.open("C:\\Doctor of Philosophy\\sample.txt", ios::app);
	for (int i = 0; i < s_input.rows; i++) {
	for (int j = 0; j < s_input.cols; j++)
	in << s_input.at<double>(i, j) << "\t";
	in << endl;
	}
	in.close();*/
	return 0;
}

int BRIEF::writeFile(Mat s_input, string loca)
{
	/*
	FileStorage file(loca, FileStorage::WRITE);
	file << "sample" << sample;
	file.release();
	*/
	ofstream in;

	in.open(loca, ios::app);

	for (int i = 0; i < s_input.rows; i++) {
		for (int j = 0; j < s_input.cols; j++)
			in << s_input.at<double>(i, j) << "\t";
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
	cout << "miu:" << (float)window_size / 2.0 << "; std:" << (float)window_size / 5.0 << endl;
	sample = Mat::zeros(samplepair, 4, DataType<float>::type);
	
	//initial Gaussian generator
	random_device rd;
	mt19937 gen(rd());
	normal_distribution<float> dist((float)window_size / 2.0, (float)window_size / 5.0);
	
	//std::map<int, int> hist;
	vector<float> container;
	for (int i = 0; i < samplepair * 4; ++i)
	{
		float tmp_value = round(dist(gen));
		if (tmp_value >= 0 && tmp_value < window_size) 
		{
			//++hist[tmp_value];
			container.push_back(tmp_value);
		}
		else 
			--i;
	}

	memcpy(sample.data, container.data(), container.size()*sizeof(float));
	
	//output histogram image
	/*for (auto p : hist) {
		std::cout << std::fixed << std::setprecision(1) << std::setw(2)
			<< p.first << ' ' << std::string(p.second, '*') << '\n';
	}*/

	cout << "new backup" << endl;
	//sple.copyTo(sample);
	cout << sample << endl;
	//return sple;
}

double BRIEF::cal_threshold(Mat image, int channel, int gStd)
{
	int r = image.rows;
	int c = image.cols;

	Scalar mean, stddev;
	Mat outmat, dev1;
	/*
	Mat g = gaussian_kernel(gStd, channel); //calculate the gaussian kernal
	filter2D(image, outmat, image.depth(), g);
	*/
	GaussianBlur(image, outmat, Size(), gStd, gStd);
	//subtract(image, outmat, dev1);
	dev1 = image - outmat;

	meanStdDev(dev1, mean, stddev); //standard deviation of image
	double threshold = stddev.val[0] * 3;

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

int BRIEF::cal_window_sample(Mat window, Mat sample, double threshold, int colour) {
	//return the brief of kernel window
	int sum = 0;
	Vec3b p1, p2;
	int g1, g2;
	if (colour != 0)
	{

		for (int i = 0; i < sample_pair; i++)
		{
			p1 = window.at<Vec3b>((int)sample.at<float>(i, 0), (int)sample.at<float>(i, 1));
			p2 = window.at<Vec3b>((int)sample.at<float>(i, 2), (int)sample.at<float>(i, 3));
			if ((double)(p1[colour] - p2[colour]) > threshold)//comparison
			{
				sum = sum + (int)pow(2, (sample_pair - i - 1));
			}
		}
	}
	else {
		for (int i = 0; i < sample_pair; i++)
		{
			g1 = (int)window.at<uchar>((int)sample.at<float>(i, 0), (int)sample.at<float>(i, 1));
			g2 = (int)window.at<uchar>((int)sample.at<float>(i, 2), (int)sample.at<float>(i, 3));
			double t = g1 - g2;
			if (t > threshold)//comparison
			{
				sum = sum + (int)pow(2, (sample_pair - i - 1));
			}
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

Mat BRIEF::combine_picture(Mat src1, Mat src2, int shape) //default - circle
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
		mask = temp;
		Point centre = Point((int)src1.cols / 2, (int)src1.rows / 2);
		int radius = (int)(src1.cols - centre.y) / 2;
		circle(mask, centre, radius, (255, 255, 255), -1);//circle
	}
	//combine and output to image
	mask.copyTo(groundtruth);

	bitwise_or(src1, mask, image);
	src2.copyTo(image, mask);

	return image;
}

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
	*/
	string it_number[7] = { "01", "02", "08", "10", "16", "19", "24" };
	int cnt = 0;
	int Patch_Size = 33;
	int Sample_Number = 0;
	//if (argc == 2)
	//	Sample_Number = atoi((argv[1]));
	int Neighbour = 3;
	int repeat = 0;

	//for (int it_it = 3; it_it < 4; it_it++)
	//{
	int it_it = 3;
		//int sn = 11;
	//	for (int j = 6; j < 7; j++)
		//{
		int j = 6;
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

			for (Sample_Number = 20; Sample_Number <= 21; Sample_Number++)
			{
				
				if (Sample_Number == 20)
				repeat = 7;
				for (; repeat < 20; repeat++)
				{
					//writing location
					for (int i = 0; i < 40; i++)
					{
						strStream << "C:\\Users\\wuq\\Documents\\Exp_distribution\\Result\\Patch_Sample\\TestParallel" << Patch_Size << "\\T" << it_number[it_it] << it_number[j] << "_" << Sample_Number << "_" << repeat << "_" << i << "segmentation.txt";
						location3[i] = strStream.str();
						strStream.str("");
					}

					BRIEF brief, texture1, texture2;
					Mat m1, m2, tex1, tex2;
					strStream.clear();

					//image 1
					texture1 = BRIEF::BRIEF();
					for (int i = 0; i < 20; i++)
					{
						texture1.image = imread(location1[i], CV_LOAD_IMAGE_GRAYSCALE);
						texture1.calBRIEFOverlap(Patch_Size, Sample_Number, 0);
						texture1.accHistogramSum();
							//texture1.sample.release();
					}
					texture1.histS.copyTo(m1);
					texture1.cleanMap();
					cout << "texture 1 first prepared" << endl;

					for (int i = 20; i < 40; i++)
						{
							texture1.image = imread(location1[i], CV_LOAD_IMAGE_GRAYSCALE);
							texture1.calBRIEFOverlap(Patch_Size, Sample_Number, 0);
							texture1.accHistogramSum();
						}
					texture1.histS.copyTo(tex1);
					brief = BRIEF::BRIEF(texture1.image, texture1.sample);
					texture1.cleanMap();
					cout << "texture 1 second prepared" << endl;

					//image 2
					texture2 = BRIEF::BRIEF(texture1.image, texture1.sample);
					for (int i = 0; i < 20; i++)
						{
							texture2.image = imread(location2[i], CV_LOAD_IMAGE_GRAYSCALE);
							texture2.calBRIEFOverlap(Patch_Size, Sample_Number, 0);
							texture2.accHistogramSum();
						}
					texture2.histS.copyTo(m2);
					texture2.cleanMap();
					cout << "texture 2 first prepared" << endl;

					for (int i = 20; i < 40; i++)
						{
							texture2.image = imread(location2[i], CV_LOAD_IMAGE_GRAYSCALE);
							texture2.calBRIEFOverlap(Patch_Size, Sample_Number, 0);
							texture2.accHistogramSum();
						}
					texture2.histS.copyTo(tex2);
					texture2.cleanMap();
					cout << "texture 2 second prepared" << endl;

					int startnumber = 0;
					if (repeat == 14)
						if (Sample_Number == 11)
							startnumber = 38;

					//testing
					vector<BRIEF> instance;
					instance.resize(40);
					for (int i = startnumber; i < 40; i++)
					{
						Mat tempImg, tempImg2;
						tempImg = imread(location1[i], CV_LOAD_IMAGE_GRAYSCALE);
						cout << location1[i] << endl;
						tempImg2 = imread(location2[i], CV_LOAD_IMAGE_GRAYSCALE);
						cout << location2[i] << endl;
						BRIEF tempTry = BRIEF::BRIEF(tempImg, texture1.sample);
						tempTry.combine_picture(tempImg, tempImg2, 0).copyTo(tempTry.image);
						instance[i] = tempTry;

						tempImg.release();
						tempImg2.release();
					}

					vector<BRIEF> testInstance_1(instance.begin(), instance.begin() + 10);
					vector<BRIEF> testInstance_2(instance.begin() + 10, instance.begin() + 20);
					vector<BRIEF> testInstance_3(instance.begin() + 20, instance.begin() + 30);
					vector<BRIEF> testInstance_4(instance.begin() + 30, instance.end());

					omp_set_dynamic(0);
#pragma omp parallel sections num_threads(2) firstprivate(testInstance_1,testInstance_2,testInstance_3,testInstance_4) shared(location3)
					{
#pragma omp section
						{
							for (int i = 0; i < 10; i++)
							{
								cout << "thread " << omp_get_thread_num << endl;
								cout << "\t" << location3[i] << endl;
								testInstance_1[i].imageSegmentation(testInstance_1[i].image, Neighbour, tex1, tex2, Patch_Size, Sample_Number, 0, location3[i]);
							}//end for
						}//end section 1
#pragma omp section
						{
							for (int j = 0; j < 10; j++)
							{
								cout << "thread " << omp_get_thread_num << endl;
								cout << "\t" << location3[j+10] << endl;
								testInstance_2[j].imageSegmentation(testInstance_2[j].image, Neighbour, tex1, tex2, Patch_Size, Sample_Number, 0, location3[j+10]);
							}//end for
						}//end section 2
#pragma omp section
						{
							for (int k = 0; k < 10; k++)
							{
								cout << "thread " << omp_get_thread_num << endl;
								cout << "\t" << location3[k+20] << endl;
								testInstance_3[k].imageSegmentation(testInstance_3[k].image, Neighbour, m1, m2, Patch_Size, Sample_Number, 0, location3[k+20]);
							}//end for
						}//end section 3
#pragma omp section
						{
							for (int l = 0; l < 10; l++)
							{
								cout << "thread " << omp_get_thread_num << endl;
								cout << "\t" << location3[l+30] << endl;
								testInstance_4[l].imageSegmentation(testInstance_4[l].image, Neighbour, m1, m2, Patch_Size, Sample_Number, 0, location3[l+30]);
							}//end for
						}//end section 1
					}//end parallel sections

					m1.release();
					m2.release();
					tex1.release();
					tex2.release();

					vector<BRIEF>().swap(instance);
					instance.clear();
					vector<BRIEF>().swap(testInstance_1);
					instance.clear(); 
					vector<BRIEF>().swap(testInstance_2);
					instance.clear(); 
					vector<BRIEF>().swap(testInstance_3);
					instance.clear();
					vector<BRIEF>().swap(testInstance_4);
					instance.clear();
					cout << "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx" << endl;
					cout << "sample pair " << Sample_Number << endl;
					cout << "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx" << endl;
				}
				
			}
		//}// for compare group j
	//}// for setup group it_it
	
	delete [] location1;
	delete [] location2;
	delete [] location3;
	
	cv::waitKey(0);
	cin.get();
	return 0;
}
