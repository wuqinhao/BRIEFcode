#include "stdafx.h"
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <math.h>
#include <stdlib.h>
#include <string>  
#include <sstream>  
#include <fstream>  
#include <iterator>  
#include <iomanip>
#include <map>
//#include <pthread.h>  
#define M_PI       3.14159265358979323846 

using namespace cv;
using namespace std;

typedef std::tuple<int, int, int> location;

class BRIEF
{

public:
	Mat image;
	Mat sample;
	int window_size;
	int sample_pair;
	int * result;
	int result_size;
	double gKernel[5];
	map<int, int> histogram;
	map<int, int> histogramS;
	map<location, float> peak;

	BRIEF();
	BRIEF(Mat img);
	int calBRIEF(int window_size, int sample_pair, int colour);
	int calBRIEF(Mat img, int colour);
	//the order for RGB is 210 

	double compareHistogramWeight(map<int, int> m1, map<int, int> m2);
	double compareHistogramChiSquare(map<int, int> m1, map<int, int> m2);
	map<int, int> accHistogram();
	map<int, int> accHistogramSum();
	int writeFile3D(Mat src, int histSize[3]);
	int writeFile(map<int, int> m1);
	int writeFile(int num, double i);
	int writeFile(string loca, double i);
	int writeFile(map<int, double> m1);
	int writeFile(double data[][80], int row, int col);
	int writeFile();
	double timesTogether(map<int, int> m, int n);
	int cleanMap();
	Mat calc3D(int histSize[3]);
	map<location, float> findDominantPeakPoint(Mat hist, int histSize[3]);
	Mat blur3D(Mat hist, int histSize[3]);
	~BRIEF();

protected:

	int flag_col;
	int flag_row;

	bool judgePeak(map<location, float> peak);
	map<location, float> combinePeak(map<location, float> peak1, map<location, float> peak2, map<location, float> peak3);
	Mat gaussian_kernel(int sigma, int dim);
	
	int reflect(int M, int x);
	double * createDGFilter(double gKernel[5]);
	double (*orthDGFilter(double gKernel[5], double kernel[][5]))[5];
	Mat init_descriptor(int window_size, int smaple_pair);
	int cal_window_sample(Mat window, Mat sample, double threshold, int colour);
	double cal_threshold(Mat image);

};

BRIEF::BRIEF(Mat img)
{
	image = img;
	window_size = 8;
	sample_pair = 17;
	flag_col = 0;
	flag_row = 0;
	result = 0;
	result_size = 0;
	
}

BRIEF::BRIEF()
{
	window_size = 8;
	sample_pair = 17;
	flag_col = 0;
	flag_row = 0;
	result = 0;
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
			sample = init_descriptor(window_size, sample_pair);
		}
		else
		{
			cout << "Sample pair number too large";
			return 1;
		}

	//window number and BRIEF vector initialization
	flag_col = image.cols / window_size;
	flag_row = image.rows / window_size;
	result_size = (int) flag_col * flag_row;
	result = new int[result_size];
	int * ptr = result;

	Mat temp;
	double th = cal_threshold(image);

	for (int i = 0; i < flag_row; i++)
		for (int j = 0; j < flag_col; j++)
		{
			Rect rect(j * window_size, i * window_size, window_size, window_size);
			image(rect).copyTo(temp);
			*ptr = cal_window_sample(temp, sample, th, colour);
			ptr++;
		}

	return 0;
}

int BRIEF::calBRIEF(Mat img, int colour)
{
	image = img;
	//descriptor initialization
	if (sample.empty())
	{
		if (pow(window_size, 2) > (sample_pair * 2)) //check input value
		{
			sample = init_descriptor(window_size, sample_pair);
		}
		else
		{
			cout << "Sample pair number too large";
			return 1;
		}
	}

	//window number and BRIEF vector initialization
	flag_col = ceil(image.cols / window_size);
	flag_row = ceil(image.rows / window_size);
	//result vector size
	result_size = (int) flag_col * flag_row;
	result = new int[result_size];
	//result vector pointer
	int * ptr = result;

	Mat temp;
	double th = cal_threshold(image);

	for (int i = 0; i < flag_row; i++)
		for (int j = 0; j < flag_col; j++)
		{
			Rect rect(j * window_size, i * window_size, window_size, window_size);
			image(rect).copyTo(temp);
			*ptr = cal_window_sample(temp, sample, th, colour);
			ptr++;
		}
	return 0;
}

double BRIEF::compareHistogramWeight(map<int, int> m1, map<int, int> n2)
{
	map <int, int>::iterator m1_Iter;
	map <int, int>::iterator m2_Iter;
	map <int, int> m2;

	//Find the same bins from the model histogram
	for (m1_Iter = m1.begin(); m1_Iter != m1.end(); m1_Iter++)
	{
		m2_Iter = n2.find(m1_Iter->first);
		if (m2_Iter != n2.end())
			m2.insert(pair<int, int>(m2_Iter->first, (m2_Iter->second / 40)));
	}

	//Calculate each mean value
	double sum1 = 0.0, sum2 = 0.0, s1 = 0.0, s2 = 0.0, ti = 0.0;
	double weight = (1 + pow(2, sample_pair)) * pow(2, sample_pair) / 2;//sigmai Weighti

																	 //First image
	for (m1_Iter = m1.begin(); m1_Iter != m1.end(); m1_Iter++)
		sum1 += (m1_Iter->second) * (m1_Iter->first);
	double mean1 = (double)sum1 / weight;
	//cout << "mean1:" << mean1 << endl;
	for (m1_Iter = m1.begin(); m1_Iter != m1.end(); m1_Iter++)
		s1 += pow((m1_Iter->second - mean1), 2) * (m1_Iter->first);
	double cov1 = (double)s1 / weight;
	//cout << "cov1:" << cov1 << endl;
	//Second image
	for (m2_Iter = m2.begin(); m2_Iter != m2.end(); m2_Iter++)
		sum2 += (m2_Iter->second) * (m2_Iter->first);
	double mean2 = (double)sum2 / weight;
	//cout << "mean2:" << mean2 << endl;
	for (m2_Iter = m2.begin(); m2_Iter != m2.end(); m2_Iter++)
		s2 += pow((m2_Iter->second - mean2), 2) * (m2_Iter->first);
	double cov2 = (double)s2 / weight;
	//cout << "cov2:" << cov2 << endl;

	for (int i = 0; i < pow(2, sample_pair); i++)
	{
		m1_Iter = m1.find(i);
		m2_Iter = m2.find(i);
		if (m1_Iter != m1.end() && m2_Iter != m2.end())
			ti += ((m1_Iter->second) - mean1) * ((m2_Iter->second) - mean2) * (m1_Iter->first);
	}

	double cov = ti / weight;
	//cout << "cov:" << cov << endl;
	double correlation = cov / (sqrt(cov1*cov2));
	return correlation;
}

double BRIEF::compareHistogramChiSquare(map<int, int> m1, map<int, int> m2)
{
	map <int, int>::iterator m1_Iter;
	map <int, int>::iterator m2_Iter;

	//Calculate each mean value
	double ti = 0;

	for (m1_Iter = m1.begin(); m1_Iter != m1.end(); m1_Iter++)
	{
		m2_Iter = m2.find(m1_Iter->first);
		if (m2_Iter != m2.end())
		{
			int d = (m1_Iter->second) - (m2_Iter->second);
			ti += pow(d, 2) / (double)(m1_Iter->second);
		}
		else ti += m1_Iter->second;
	}

	return (double)ti;
}

map<int, int> BRIEF::accHistogram()
{
	map<int, int>::iterator h_It;
	int * ptr = result;
	for (int i = 0; i < result_size; i++)
	{
		h_It = histogram.find((*ptr));
		if (h_It == histogram.end())
			histogram.insert(pair <int, int>(*ptr, 1));
		else (h_It->second)++;
		ptr++;
	}
	map<int, int> h = histogram;

	histogram.clear();
	return h;
}

map<int, int> BRIEF::accHistogramSum()
{
	map<int, int>::iterator h_It;
	int * ptr = result;
	for (int i = 0; i < result_size; i++)
	{
		h_It = histogramS.find((*ptr));
		if (h_It == histogramS.end())
			histogramS.insert(pair <int, int>(*ptr, 1));
		else (h_It->second)++;
		ptr++;
	}
	map<int, int> h = histogramS;

	return h;
}

double BRIEF::timesTogether(map<int, int> m, int n)
{
	double a = 1;
	map<int, int>::iterator h_It;

	//h_It++;//avoid 0 value
	for (h_It = m.begin(); h_It != m.end(); h_It++)
	{
		if (h_It->second != 0)
			a = a * h_It->second / n;
	}
	return a;
}

int BRIEF::cleanMap()
{
	histogramS.clear();
	return 0;
}

int BRIEF::writeFile3D(Mat src, int histSize[3])
{
	ofstream in;

	in.open("C:\\Doctor of Philosophy\\z.txt", ios::app);

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

	return 0;
}

int BRIEF::writeFile(map<int, int> m1)
{
	map <int, int>::iterator m1_Iter;
	ofstream in;

	in.open("C:\\Doctor of Philosophy\\histogram.txt", ios::app);

	for (m1_Iter = m1.begin(); m1_Iter != m1.end(); m1_Iter++)
		in << m1_Iter->first << "\t" << m1_Iter->second << endl;
	in << endl;
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

Mat BRIEF::init_descriptor(int window_size, int sample_pair) {
	//initial descriptor compare type
	Mat sample(sample_pair, 4, CV_32F);
	RNG rng;
	rng.fill(sample, RNG::NORMAL, (window_size - 1) / 2, window_size / 5, false);

	for (int i = 0; i < sample_pair; i++)
	{
		for (int j = 0; j < 4; j++)
		{
			if (sample.at<float>(i, j) >(window_size - 1))
				switch (j)
				{
				case 0:
					sample.at<float>(i, j) = (float)window_size - 1;
					sample.at<float>(i, j + 1) = (-sample.at<float>(i, 2)) / (sample.at<float>(i, 0) - sample.at<float>(i, 2))*(sample.at<float>(i, 1) - sample.at<float>(i, 3)) + sample.at<float>(i, 3);
					break;
				case 1:
					sample.at<float>(i, j) = (float)window_size - 1;
					sample.at<float>(i, j - 1) = (-sample.at<float>(i, 3)) / (sample.at<float>(i, 1) - sample.at<float>(i, 3))*(sample.at<float>(i, 0) - sample.at<float>(i, 2)) + sample.at<float>(i, 2);
					break;
				case 2:
					sample.at<float>(i, j) = (float)window_size - 1;
					sample.at<float>(i, j + 1) = (-sample.at<float>(i, 0)) / (sample.at<float>(i, 2) - sample.at<float>(i, 0))*(sample.at<float>(i, 3) - sample.at<float>(i, 1)) + sample.at<float>(i, 1);
					break;
				case 3:
					sample.at<float>(i, j) = (float)window_size - 1;
					sample.at<float>(i, j - 1) = (-sample.at<float>(i, 1)) / (sample.at<float>(i, 3) - sample.at<float>(i, 1))*(sample.at<float>(i, 2) - sample.at<float>(i, 0)) + sample.at<float>(i, 0);
					break;
				}
			if (sample.at<float>(i, j) < 0)
				switch (j)
				{
				case 0:
					sample.at<float>(i, j) = 0;
					sample.at<float>(i, j + 1) = (-sample.at<float>(i, 2)) / (sample.at<float>(i, 0) - sample.at<float>(i, 2))*(sample.at<float>(i, 1) - sample.at<float>(i, 3)) + sample.at<float>(i, 3);
					break;
				case 1:
					sample.at<float>(i, j) = 0;
					sample.at<float>(i, j - 1) = (-sample.at<float>(i, 3)) / (sample.at<float>(i, 1) - sample.at<float>(i, 3))*(sample.at<float>(i, 0) - sample.at<float>(i, 2)) + sample.at<float>(i, 2);
					break;
				case 2:
					sample.at<float>(i, j) = 0;
					sample.at<float>(i, j + 1) = (-sample.at<float>(i, 0)) / (sample.at<float>(i, 2) - sample.at<float>(i, 0))*(sample.at<float>(i, 3) - sample.at<float>(i, 1)) + sample.at<float>(i, 1);
					break;
				case 3:
					sample.at<float>(i, j) = 0;
					sample.at<float>(i, j - 1) = (-sample.at<float>(i, 1)) / (sample.at<float>(i, 3) - sample.at<float>(i, 1))*(sample.at<float>(i, 2) - sample.at<float>(i, 0)) + sample.at<float>(i, 0);
					break;
				}
			cout << (int)sample.at<float>(i, j) << "\t";
		}cout << endl;
	}
	return sample;
}

double BRIEF::cal_threshold(Mat image)
{
	int r = image.rows;
	int c = image.cols;

	Scalar mean, stddev;
	Mat outmat, dev;

	Mat g = gaussian_kernel(1, 3); //calculate the gaussian kernal

	filter2D(image, outmat, image.depth(), g);
	subtract(image, outmat, dev);

	Mat dev1 = dev;
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
	return gaussK;
}

int BRIEF::cal_window_sample(Mat window, Mat sample, double threshold, int colour) {
	//return the brief of kernel window
	int sum = 0;
	Vec3b p1, p2;

	for (int i = 0; i < sample_pair; i++)
	{
		p1 = window.at<Vec3b>((int)sample.at<float>(i, 0), (int)sample.at<float>(i, 1));
		p2 = window.at<Vec3b>((int)sample.at<float>(i, 2), (int)sample.at<float>(i, 3));
		if ((double)(p1[colour] - p2[colour]) > threshold)//comparison
		{
			sum = sum + (int)pow((sample_pair - i - 1), 2);
		}
	}
	return sum;
}

Mat BRIEF::calc3D(int histSize[3])
{
	float range[2] = { 0, 255 };
	const float * ranges[3] = { range, range, range };
	int channels[3] = { 0, 1, 2 };

	Mat hist;
	calcHist(&image, 1, channels, Mat(), hist, 3, histSize, ranges);
	
	return hist;
}

map<location, float> BRIEF::findDominantPeakPoint(Mat hist, int histSize[3])
{
	Mat hist_c = hist.clone();
	createDGFilter(gKernel);
	map<location, float> peak_x, peak_y, peak_z;
	int loc_x, loc_y, loc_z;
	int flag = 0; // to judge the plat in peak or in valley; default in valley
	int dflag = 0; // delete node flag;

	for (int i = 0; i < 1; i++) {
		hist_c = blur3D(hist_c, histSize);
		cout << "hello" << endl;
		
		flag = 0;
		dflag = 0;

		// along X direction
		for (int z = 0; z < histSize[2]; z++)
			for (int y = 0; y < histSize[1]; y++)
				for (int x = 0; x < histSize[0] - 1; x++)
				{
					if (hist_c.at<float>(x, y, z) < hist_c.at<float>(x + 1, y, z))
						flag = 1;
					else if (flag == 1)
					{
						if (hist_c.at<float>(x, y, z) > hist_c.at<float>(x + 1, y, z))
						{
							peak_x.insert(pair<location, float>(location(x, y, z), hist_c.at<float>(x, y, z)));
							flag = 0;
						}
						if (hist_c.at<float>(x, y, z) == hist_c.at<float>(x + 1, y, z))
							peak_x.insert(pair<location, float>(location(x, y, z), hist_c.at<float>(x, y, z)));
					}
				}

		// along Y direction
		for (int z = 0; z < histSize[2]; z++)
			for (int x = 0; x < histSize[0]; x++)
				for (int y = 0; y < histSize[1] - 1; y++)
				{
					if (hist_c.at<float>(x, y, z) < hist_c.at<float>(x, y + 1, z))
						flag = 1;
					else if (flag == 1)
					{
						if (hist_c.at<float>(x, y, z) > hist_c.at<float>(x, y + 1, z))
						{
							peak_y.insert(pair<location, float>(location(x, y, z), hist_c.at<float>(x, y, z)));
							flag = 0;
						}
						if (hist_c.at<float>(x, y, z) == hist_c.at<float>(x, y + 1, z))
							peak_y.insert(pair<location, float>(location(x, y, z), hist_c.at<float>(x, y, z)));
					}
				}

		// along Z direction
		for (int y = 0; y < histSize[1]; y++)
			for (int x = 0; x < histSize[0]; x++)
				for (int z = 0; z < histSize[2] - 1; z++)
				{
					if (hist_c.at<float>(x, y, z) < hist_c.at<float>(x, y, z + 1))
						flag = 1;
					else if (flag == 1)
					{
						if (hist_c.at<float>(x, y, z) > hist_c.at<float>(x, y, z + 1))
						{
							peak_z.insert(pair<location, float>(location(x, y, z), hist_c.at<float>(x, y, z)));
							flag = 0;
						}
						if (hist_c.at<float>(x, y, z) == hist_c.at<float>(x, y, z + 1))
							peak_z.insert(pair<location, float>(location(x, y, z), hist_c.at<float>(x, y, z)));
					}
				}

		peak = combinePeak(peak_x, peak_y, peak_z);
		if (judgePeak(peak))
		{
			cout << "Don't need" << endl;
			return peak;
		}
		else
		{
			cout << "failed " << i << endl;
		}
	}
	cout << "Nothing really happened" << endl;
	return peak;
	/*
	float sum_peak = 0, th_dpeak;
	map<location, float>::iterator it_peak;
	for (it_peak = peak.begin(); it_peak != peak.end(); it_peak++)
		sum_peak += it_peak->second;
	th_dpeak = sum_peak / peak.size() * 0.01;

	for (it_peak = peak.begin(); it_peak != peak.end();)
	{
		if (it_peak->second < th_dpeak)
		{
			peak.erase(it_peak);
			it_peak++;
			continue;
		}
		it_peak++;
	}
	*/
	
}

bool BRIEF::judgePeak(map<location, float> peak)
{
	// use euclidean distance as threshold: 50
	location l1, l2;
	double distance;
	int flag = 0;
	for (auto it = peak.begin(); it != peak.end();)
	{
		for (auto it2 = it; it2 != peak.end();) 
		{
			l1 = it->first;
			l2 = (it2++)->first;
			distance = sqrt(pow(get<0>(l1) - get<0>(l2), 2) + pow(get<1>(l1) - get<1>(l2), 2) + pow(get<2>(l1) - get<2>(l2), 2));
			cout << distance << " ";
			if (distance < 5.0)
				flag++;
		}
		cout << endl;
		it++;
	}
	return flag;
}

map<location, float> BRIEF::combinePeak(map<location, float> peak1, map<location, float> peak2, map<location, float> peak3)
{
	map<location, float>::iterator it1, it2, it3;
	
	for (it1 = peak1.begin(); it1 != peak1.end();)
	{
		it2 = peak2.find(it1->first);
		if (it2 == peak2.end())
		{
			peak1.erase(it1++);
			continue;
		}
		it3 = peak3.find(it1->first);
		if (it3 == peak3.end())
		{
			peak1.erase(it1++);
			continue;
		}
		it1++;
	}

	return peak1;
}

Mat BRIEF::blur3D(Mat hist, int histSize[3])
{
	Mat src, temp1, temp2, dist;
	src = hist.clone();
	temp1 = hist.clone();
	temp2 = hist.clone();
	dist = hist.clone();
	float sum, x1, y1, z1;
	
	
	// along z - direction
	for (int z = 0; z < histSize[2]; z++)
	{
		for (int y = 0; y < histSize[1]; y++)
			for (int x = 0; x < histSize[0]; x++)
			{
				//i++;
				sum = 0.0;
				switch (y)
				{
				case 0: if (0 == (src.at<float>(x, y, z) || src.at<float>(x, y + 1, z)
					|| src.at<float>(x, y + 2, z)))
					break;
						else
						{
							for (int i = -2; i <= 2; i++)
							{
								y1 = reflect(histSize[1], y - i);
								sum = sum + gKernel[i + 2] * src.at<float>(x, y1, z);
							}
							temp1.at<float>(x, y, z) = sum;
							//cout << temp1.at<float>(x, y, z) << " " << i << endl;
							break;
						}
				case 1: if (0 == (src.at<float>(x, y, z) || src.at<float>(x, y + 1, z)
					|| src.at<float>(x, y + 2, z) || src.at<float>(x, y - 1, z)))
					break;
						else
						{
							for (int i = -2; i <= 2; i++)
							{
								y1 = reflect(histSize[1], y - i);
								sum = sum + gKernel[i + 2] * src.at<float>(x, y1, z);
							}
							temp1.at<float>(x, y, z) = sum;
							//cout << temp1.at<float>(x, y, z) << " " << i << endl;
							break;
						}
				case 255: if (0 == (src.at<float>(x, y, z) || src.at<float>(x, y - 1, z)
					|| src.at<float>(x, y - 2, z)))
					break;
						else
						{
							for (int i = -2; i <= 2; i++)
							{
								y1 = reflect(histSize[1], y - i);
								sum = sum + gKernel[i + 2] * src.at<float>(x, y1, z);
							}
							temp1.at<float>(x, y, z) = sum;
							//cout << temp1.at<float>(x, y, z) << " " << i << endl;
							break;
						}
				case 254: if (0 == (src.at<float>(x, y, z) || src.at<float>(x, y + 1, z)
					|| src.at<float>(x, y - 2, z) || src.at<float>(x, y - 1, z)))
					break;
						else
						{
							for (int i = -2; i <= 2; i++)
							{
								y1 = reflect(histSize[1], y - i);
								sum = sum + gKernel[i + 2] * src.at<float>(x, y1, z);
							}
							temp1.at<float>(x, y, z) = sum;
							//cout << temp1.at<float>(x, y, z) << " " << i << endl;
							break;
						}
				default: if (0 == (src.at<float>(x, y, z) || src.at<float>(x, y + 1, z)
					|| src.at<float>(x, y + 2, z) || src.at<float>(x, y - 2, z)
					|| src.at<float>(x, y - 1, z)))
					break;
						 else
						 {
							 for (int i = -2; i <= 2; i++)
							 {
								 y1 = reflect(histSize[1], y - i);
								 sum = sum + gKernel[i + 2] * src.at<float>(x, y1, z);
							 }
							 temp1.at<float>(x, y, z) = sum;
							 //cout << temp1.at<float>(x, y, z) << " " << i << endl;
							 break;
						 }
				}
			}
		for (int y = 0; y < histSize[1]; y++)
			for (int x = 0; x < histSize[0]; x++)
			{
				sum = 0.0;
				switch (x)
				{
				case 0: if (0 == (temp1.at<float>(x, y, z) || temp1.at<float>(x + 1, y, z)
					|| temp1.at<float>(x + 2, y, z)))
					break;
						else
						{
							for (int i = -2; i <= 2; i++)
							{
								x1 = reflect(histSize[0], x - i);
								sum = sum + gKernel[i + 2] * temp1.at<float>(x1, y, z);
							}
							temp2.at<float>(x, y, z) = sum;
							break;
						}
				case 1: if (0 == (temp1.at<float>(x, y, z) || temp1.at<float>(x + 1, y, z)
					|| temp1.at<float>(x + 2, y, z) || temp1.at<float>(x - 1, y, z)))
					break;
						else
						{
							for (int i = -2; i <= 2; i++)
							{
								x1 = reflect(histSize[0], x - i);
								sum = sum + gKernel[i + 2] * temp1.at<float>(x1, y, z);
							}
							temp2.at<float>(x, y, z) = sum;
							break;
						}
				case 255: if (0 == (temp1.at<float>(x, y, z) || temp1.at<float>(x - 1, y, z)
					|| temp1.at<float>(x - 2, y, z)))
					break;
						else
						{
							for (int i = -2; i <= 2; i++)
							{
								x1 = reflect(histSize[0], x - i);
								sum = sum + gKernel[i + 2] * temp1.at<float>(x1, y, z);
							}
							temp2.at<float>(x, y, z) = sum;
							break;
						}
				case 254: if (0 == (temp1.at<float>(x, y, z) || temp1.at<float>(x - 1, y, z)
					|| temp1.at<float>(x - 2, y, z) || temp1.at<float>(x + 1, y, z)))
					break;
						else
						{
							for (int i = -2; i <= 2; i++)
							{
								x1 = reflect(histSize[0], x - i);
								sum = sum + gKernel[i + 2] * temp1.at<float>(x1, y, z);
							}
							temp2.at<float>(x, y, z) = sum;
							break;
						}
				default: if (0 == (temp1.at<float>(x, y, z) || temp1.at<float>(x + 1, y, z)
					|| temp1.at<float>(x + 2, y, z) || temp1.at<float>(x - 2, y, z)
					|| temp1.at<float>(x - 1, y, z)))
					break;
						 else
						 {
							 for (int i = -2; i <= 2; i++)
							 {
								 x1 = reflect(histSize[0], x - i);
								 sum = sum + gKernel[i + 2] * temp1.at<float>(x1, y, z);
							 }
							 temp2.at<float>(x, y, z) = sum;
							 break;
						 }
				}
			}
	}
	
	//along x - direction
	for (int x = 0; x < histSize[0]; x++)
	{
		for (int y = 0; y < histSize[1]; y++)
			for (int z = 0; z < histSize[2]; z++)
			{
				sum = 0.0;
				switch (y)
				{
				case 0: if (0 == (temp2.at<float>(x, y, z) || temp2.at<float>(x, y + 1, z)
					|| temp2.at<float>(x, y + 2, z)))
					break;
						else
						{
							for (int i = -2; i <= 2; i++)
							{
								y1 = reflect(histSize[1], y - i);
								sum = sum + gKernel[i + 2] * temp2.at<float>(x, y1, z);
							}
							temp1.at<float>(x, y, z) = sum;
							break;
						}
				case 1: if (0 == (temp2.at<float>(x, y, z) || temp2.at<float>(x, y + 1, z)
					|| temp2.at<float>(x, y + 2, z) || temp2.at<float>(x, y - 1, z)))
					break;
						else
						{
							for (int i = -2; i <= 2; i++)
							{
								y1 = reflect(histSize[1], y - i);
								sum = sum + gKernel[i + 2] * temp2.at<float>(x, y1, z);
							}
							temp1.at<float>(x, y, z) = sum;
							break;
						}
				case 255: if (0 == (temp2.at<float>(x, y, z) || temp2.at<float>(x, y - 1, z)
					|| temp2.at<float>(x, y - 2, z)))
					break;
						else
						{
							for (int i = -2; i <= 2; i++)
							{
								y1 = reflect(histSize[1], y - i);
								sum = sum + gKernel[i + 2] * temp2.at<float>(x, y1, z);
							}
							temp1.at<float>(x, y, z) = sum;
							break;
						}
				case 254: if (0 == (temp2.at<float>(x, y, z) || temp2.at<float>(x, y + 1, z)
					|| temp2.at<float>(x, y - 2, z) || temp2.at<float>(x, y - 1, z)))
					break;
						else
						{
							for (int i = -2; i <= 2; i++)
							{
								y1 = reflect(histSize[1], y - i);
								sum = sum + gKernel[i + 2] * temp2.at<float>(x, y1, z);
							}
							temp1.at<float>(x, y, z) = sum;
							break;
						}
				default: if (0 == (temp2.at<float>(x, y, z) || temp2.at<float>(x, y + 1, z)
					|| temp2.at<float>(x, y + 2, z) || temp2.at<float>(x, y -2, z)
					|| temp2.at<float>(x, y - 1, z)))
					break;
						 else
						 {
							 for (int i = -2; i <= 2; i++)
							 {
								 y1 = reflect(histSize[1], y - i);
								 sum = sum + gKernel[i + 2] * temp2.at<float>(x, y1, z); 
							 }
							 temp1.at<float>(x, y, z) = sum;
							 break;
						 }
				}

			}
		for (int y = 0; y < histSize[1]; y++)
			for (int z = 0; z < histSize[2]; z++)
			{
				sum = 0.0;
				switch (z)
				{
				case 0: if (0 == (temp1.at<float>(x, y, z) || temp1.at<float>(x, y, z + 1)
					|| temp1.at<float>(x, y, z + 2)))
					break;
						else
						{
							for (int i = -2; i <= 2; i++)
							{
								z1 = reflect(histSize[2], z - i);
								sum = sum + gKernel[i + 2] * temp1.at<float>(x, y, z1);
							}
							temp2.at<float>(x, y, z) = sum;
							break;
						}
				case 1: if (0 == (temp1.at<float>(x, y, z) || temp1.at<float>(x, y, z + 1)
					|| temp1.at<float>(x, y, z + 2) || temp1.at<float>(x, y, z - 1)))
					break;
						else
						{
							for (int i = -2; i <= 2; i++)
							{
								z1 = reflect(histSize[2], z - i);
								sum = sum + gKernel[i + 2] * temp1.at<float>(x, y, z1);
							}
							temp2.at<float>(x, y, z) = sum;
							break;
						}
				case 255: if (0 == (temp1.at<float>(x, y, z) || temp1.at<float>(x, y, z - 1)
					|| temp1.at<float>(x, y, z - 2)))
					break;
						else
						{
							for (int i = -2; i <= 2; i++)
							{
								z1 = reflect(histSize[2], z - i);
								sum = sum + gKernel[i + 2] * temp1.at<float>(x, y, z1);
							}
							temp2.at<float>(x, y, z) = sum;
							break;
						}
				case 254: if (0 == (temp1.at<float>(x, y, z) || temp1.at<float>(x, y, z + 1)
					|| temp1.at<float>(x, y, z - 2) || temp1.at<float>(x, y, z - 1)))
					break;
						else
						{
							for (int i = -2; i <= 2; i++)
							{
								z1 = reflect(histSize[2], z - i);
								sum = sum + gKernel[i + 2] * temp1.at<float>(x, y, z1);
							}
							temp2.at<float>(x, y, z) = sum;
							break;
						}
				default: if (0 == (temp1.at<float>(x, y, z) || temp1.at<float>(x, y, z + 1)
					|| temp1.at<float>(x, y, z + 2) || temp1.at<float>(x, y, z - 2)
					|| temp1.at<float>(x, y, z - 1)))
					break;
						 else
						 {
							 for (int i = -2; i <= 2; i++)
							 {
								 z1 = reflect(histSize[2], z - i);
								 sum = sum + gKernel[i + 2] * temp1.at<float>(x, y, z1);
							 } 
							 temp2.at<float>(x, y, z) = sum;
							 break;
						 }
				}
			}
	}
	
	// along y - direction
	for (int y = 0; y < histSize[1]; y++)
	{
		for (int x = 0; x < histSize[0]; x++)
			for (int z = 0; z < histSize[2]; z++)
			{
				sum = 0.0;
				switch (x)
				{
				case 0: if (0 == (temp2.at<float>(x, y, z) || temp2.at<float>(x + 1, y, z)
					|| temp2.at<float>(x + 2, y, z)))
					break;
						else
						{
							for (int i = -2; i <= 2; i++)
							{
								x1 = reflect(histSize[0], x - i);
								sum = sum + gKernel[i + 2] * temp2.at<float>(x1, y, z);
							}
							temp1.at<float>(x, y, z) = sum;
							break;
						}
				case 1: if (0 == (temp2.at<float>(x, y, z) || temp2.at<float>(x + 1, y, z)
					|| temp2.at<float>(x + 2, y, z) || temp2.at<float>(x - 1, y, z)))
					break;
						else
						{
							for (int i = -2; i <= 2; i++)
							{
								x1 = reflect(histSize[0], x - i);
								sum = sum + gKernel[i + 2] * temp2.at<float>(x1, y, z);
							}
							temp1.at<float>(x, y, z) = sum;
							break;
						}
				case 255: if (0 == (temp2.at<float>(x, y, z) || temp2.at<float>(x - 1, y, z)
					|| temp2.at<float>(x - 2, y, z)))
					break;
						  else
						  {
							  for (int i = -2; i <= 2; i++)
							  {
								  x1 = reflect(histSize[0], x - i);
								  sum = sum + gKernel[i + 2] * temp2.at<float>(x1, y, z);
							  }
							  temp1.at<float>(x, y, z) = sum; 
							  break;
						  }
				case 254: if (0 == (temp2.at<float>(x, y, z) || temp2.at<float>(x - 1, y, z)
					|| temp2.at<float>(x - 2, y, z) || temp2.at<float>(x + 1, y, z)))
					break;
						  else
						  {
							  for (int i = -2; i <= 2; i++)
							  {
								  x1 = reflect(histSize[0], x - i);
								  sum = sum + gKernel[i + 2] * temp2.at<float>(x1, y, z);
							  }
							  temp1.at<float>(x, y, z) = sum; 
							  break;
						  }
				default: if (0 == (temp2.at<float>(x, y, z) || temp2.at<float>(x + 1, y, z)
					|| temp2.at<float>(x + 2, y, z) || temp2.at<float>(x - 2, y, z)
					|| temp2.at<float>(x - 1, y, z)))
					break;
						 else
						 {
							// cout << temp2.at<float>(x, y, z) << " ";
							 for (int i = -2; i <= 2; i++)
							 {
								 x1 = reflect(histSize[0], x - i);
								 sum = sum + gKernel[i + 2] * temp2.at<float>(x1, y, z);
							 }
							 temp1.at<float>(x, y, z) = sum; 
							 break;
						 }
				}
				
			}
		
		for (int z = 0; z < histSize[2]; z++)
			for (int x = 0; x < histSize[0]; x++)
			{
				sum = 0.0;
				switch (z)
				{
				case 0: if (0 == (temp1.at<float>(x, y, z) || temp1.at<float>(x, y, z + 1)
					|| temp1.at<float>(x, y, z + 2)))
					break;
						else
						{
							for (int i = -2; i <= 2; i++)
							{
								z1 = reflect(histSize[2], z - i);
								sum = sum + gKernel[i + 2] * temp1.at<float>(x, y, z1);
							}
							hist.at<float>(y, x, z) = sum;
							break;
						}
				case 1: if (0 == (temp1.at<float>(x, y, z) || temp1.at<float>(x, y, z + 1)
					|| temp1.at<float>(x, y, z + 2) || temp1.at<float>(x, y, z - 1)))
					break;
						else
						{
							for (int i = -2; i <= 2; i++)
							{
								z1 = reflect(histSize[2], z - i);
								sum = sum + gKernel[i + 2] * temp1.at<float>(x, y, z1);
							}
							hist.at<float>(y, x, z) = sum;
							break;
						}
				case 255: if (0 == (temp1.at<float>(x, y, z) || temp1.at<float>(x, y, z - 1)
					|| temp1.at<float>(x, y, z - 2)))
					break;
						  else
						  {
							  for (int i = -2; i <= 2; i++)
							  {
								  z1 = reflect(histSize[2], z - i);
								  sum = sum + gKernel[i + 2] * temp1.at<float>(x, y, z1);
							  }
							  hist.at<float>(y, x, z) = sum;
							  break;
						  }
				case 254: if (0 == (temp1.at<float>(x, y, z) || temp1.at<float>(x, y, z + 1)
					|| temp1.at<float>(x, y, z - 2) || temp1.at<float>(x, y, z - 1)))
					break;
						  else
						  {
							  for (int i = -2; i <= 2; i++)
							  {
								  z1 = reflect(histSize[2], z - i);
								  sum = sum + gKernel[i + 2] * temp1.at<float>(x, y, z1);
							  }
							  hist.at<float>(y, x, z) = sum; 
							  break;
						  }
				default: if (0 == (temp1.at<float>(x, y, z) || temp1.at<float>(x, y, z + 1)
					|| temp1.at<float>(x, y, z + 2) || temp1.at<float>(x, y, z - 2)
					|| temp1.at<float>(x, y, z - 1)))
					break;
						 else
						 {
							 for (int i = -2; i <= 2; i++)
							 {
								 z1 = reflect(histSize[2], z - i);
								 sum = sum + gKernel[i + 2] * temp1.at<float>(x, y, z1);
							 }
							 hist.at<float>(y, x, z) = sum; 
						
							 break;
						 }
				}

			}
	}
	
	return hist;
}

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

double* BRIEF::createDGFilter(double gKernel[5])
{
	// set standard deviation to 1.0
	double sigma = 1.0;
	double r, s = 2.0 * sigma * sigma;

	// sum is for normalization
	double sum = 0.0;

	// generate 1D derivated Gaussian kernel
	for (int x = -2; x <= 2; x++)
	{
		r = sqrt(x*x);
		//gKernel[x + 2] = (exp(-(r*r) / s)) / (sqrt(2 * M_PI) * sigma) * (-2 * x) / s;
		gKernel[x + 2] = (exp(-(r*r) / s)) / (M_PI * s);
		sum += gKernel[x + 2];
	}

	return gKernel;
}

double (* BRIEF::orthDGFilter(double gKernel[5], double kernel[][5]))[5]
//gKernel is the original 1 dimensional Gaussian kernel; kernel is the distance 2D kernel
{
	double sum = 0.0;
	for (int i = 0; i < 5; i++)
	{
		double term = gKernel[i];
		for (int j = 0; j < 5; j++)
		{
			if (0 == (term*gKernel[j]))
				kernel[i][j] = 0.0;
			else
				kernel[i][j] = term * gKernel[j];
		}
	}

	return kernel;
}

BRIEF::~BRIEF() {}

int main(int argc, char** argv)
{
	Mat image;
	Mat result;
	//Vector<Mat> out;
	stringstream strStream;
	string * location2 = new string[40];
	string * location1 = new string[40];
	string number;
	map<int, int>::iterator h_It;
	map<int, int>::iterator s1_It;
	map<int, int>::iterator s2_It;
	//double rt[256][80];
	map<int, int> sum1;
	map<int, int> sum2;
	map<int, int> m1;
	map<int, int> m2;
	BRIEF brief;
	map<int, int> r1;
	double b1 = 0;
	double b2 = 0;

	for (int i = 0; i < 40; i++)
	{
		if (i >= 9)
		{
			strStream << "C:\\Users\\wuq\\Documents\\Source\\MREye PNG format\\Controls\\c (" << i + 1 << ").png";
			location1[i] = strStream.str();
			strStream.str(""); // clean Stringstream
		}
		else
		{
			strStream << "C:\\Users\\wuq\\Documents\\Source\\Texture Datasets\\UIUCTex\\T01_bark1\\T01_0" << i + 1 << ".jpg";
			location1[i] = strStream.str();
			strStream.str(""); // clean Strin==gstream
		}
	}
	strStream.clear();
	for (int i = 0; i < 40; i++)
	{
		if (i >= 9)
		{
			strStream << "C:\\Users\\wuq\\Documents\\Source\\Texture Datasets\\UIUCTex\\T25_plaid\\T25_" << i + 1 << ".jpg";
			location2[i] = strStream.str();
			strStream.str(""); // clean Stringstream
		}
		else
		{
			strStream << "C:\\Users\\wuq\\Documents\\Source\\Texture Datasets\\UIUCTex\\T25_plaid\\T25_0" << i + 1 << ".jpg";
			location2[i] = strStream.str();
			strStream.str(""); // clean Stringstream
		}
	}
	strStream.clear();

	//image = imread(location1[10], CV_LOAD_IMAGE_COLOR);

	//brief.image = imread(location1[10], CV_LOAD_IMAGE_COLOR);
	brief.image = imread("C:\\Users\\wuq\\Documents\\Longreport\\used.jpg", CV_LOAD_IMAGE_COLOR);
	//Rect myROI(750,400,500,500);
	//Mat crop_img = brief.image(myROI);
	if (!brief.image.data)                              // Check for invalid input
	{
		cout << "Could not open or find the image" << std::endl;
		return -1;
	}
	/*try {
		imwrite("C:\\Users\\wuq\\Documents\\images\\crop_img.png", crop_img);
	}
	catch (runtime_error& ex) {
		fprintf(stderr, "Exception converting image to PNG format: %s\n", ex.what());
		return 1;
	}*/
	//for (int i = 1; i < 31; i = i + 2)
	//blur(brief.image, brief.image, Size(25, 25));
	//blur(brief.image, brief.image, Size(25, 25));
	//Mat image = imread(location1[5], CV_LOAD_IMAGE_COLOR);
	//namedWindow("w", WINDOW_AUTOSIZE);
	//imshow("w", brief.image);
	
	//cout << cv::getBuildInformation() << endl;
	
	int histSize[3] = { 256, 256, 256 };
	result = brief.calc3D(histSize);
	brief.writeFile3D(result, histSize);
	//brief.findDominantPeakPoint(result, histSize);
	
	
	//cout << "Hist.dims = " << result.dims << endl;
	//cout << "Value: " << result.at<double>(0, 0, 0) << endl;
	//cout << "Hist.rows = " << result.size.p[0] << endl;
	//cout << "Hist.datastart = " << result.datastart<< endl;
	//brief.writeFile3D(result);
	system("pause");
	//waitKey(0);
	return 0;

}