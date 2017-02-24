#include "stdafx.h"
//#include "Time.h"
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
//#include <iomanip>
#include <map>
#include <unordered_map>
#include <windows.h>
#include <ppl.h>
//#include "C:\Program Files\boost\boost_1_62_0\boost\unordered\unordered_map.hpp"
//#include <direct.h>
//#include <pthread.h>  
#define M_PI       3.14159265358979323846 


using namespace cv;
using namespace std;
//using namespace concurrency;
//using namespace boost::unordered;

typedef std::tuple<int, int, int> location;


class BRIEF
{

public:
	Mat image;
	Mat sample;
	Mat re_image;
	Mat rlt;
	int window_size;
	int sample_pair;
	int flag_col;
	int flag_row;
	int * result;
	int result_size;
	double gKernel[5];
	map<int, int> NBresult;
	map<int, int> histogram;
	map<int, int> histogramS;
	map<location, float> peak;

	BRIEF();
	BRIEF(Mat img);
	BRIEF(Mat img, Mat sp);
	BRIEF(Mat img1, Mat img2, String sp_loca);
	int calBRIEF(int window_size, int sample_pair, int colour);
	int calBRIEF(Mat img, int colour);
	int calBRIEFOverlap(int w_size, int s_pair, int colour);
	//the order for RGB is 210 

	map <int, int> compareHistogramNB(map<int, int> m1, map<int, int> m2);
	double compareHistogramWeight(map<int, int> m1, map<int, int> m2);
	double compareHistogramWeight(unordered_map<int, int> m1, unordered_map<int, int> m2);
	double compareHistogramChiSquare(map<int, int> m1, map<int, int> m2);
	double compareHistogramChiSquare(unordered_map<int, int> m1, unordered_map<int, int> m2);
	int imageSegmentation(Mat test_image, int a_size, map<int, int> texture1, map<int, int> texture2, int w_size, int s_pair, int colour, string loca);
	Mat combine_picture(Mat src1, Mat src2, int shape);
	int recreate_NBpicture();
	map<int, int> accHistogram();
	map<int, int> accHistogram(Mat input);
	map<int, int> accHistogramSum();
	map<int, int> histogramAdd(map<int, int> in1, map<int, int> in2);	
	map <int, int> histogramMinus(map<int, int> m1, map<int, int> m2);
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
	double timesTogether(map<int, int> m, int n);
	int cleanMap();
	Mat calc3D(int histSize[3]);
	unordered_map<int, int> mapConvertToUnordered(map<int, int> m1);
	Mat mapConvertToMat(map<int, int> m1, int hist_size);
	~BRIEF();
	
protected:
	Mat gaussian_kernel(int sigma, int dim);
	
	int reflect(int M, int x);
	double * createDGFilter(double gKernel[5]);
	double (*orthDGFilter(double gKernel[5], double kernel[][5]))[5];
	Mat init_descriptor(int window_size, int smaple_pair);
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
	result = 0;
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
	result = 0;
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
	result = 0;
	result_size = 0;
}

BRIEF::BRIEF()
{
	window_size = 7;
	sample_pair = 9;
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
		//if (pow(window_size, 2) > (sample_pair * 2)) //check input value
		{
			sample = init_descriptor(window_size, sample_pair);
		}
		/*else
		{
			cout << "Sample pair number too large";
			return 1;
		}*/

	//window number and BRIEF vector initialization
	flag_col = image.cols / window_size;
	flag_row = image.rows / window_size;
	result_size = (int) flag_col * flag_row;
	rlt = Mat::zeros(flag_row, flag_col, DataType<float>::type);
	result = new int[result_size];
	int * ptr = result;
	
	Mat temp;
	if (colour == 0)
		colour = 1;
	double th = cal_threshold(image, colour, 1);

	for (int i = 0; i < flag_row; i++)
		for (int j = 0; j < flag_col; j++)
		{
			Rect rect(j * window_size, i * window_size, window_size, window_size);
			image(rect).copyTo(temp);
			*ptr = cal_window_sample(temp, sample, th, colour);
			rlt.at<float>(i, j) = *ptr;
			ptr++;
		}

	return 0;
}

int BRIEF::calBRIEFOverlap(int w_size, int s_pair, int colour)
{
	window_size = w_size;
	sample_pair = s_pair;

	//descriptor initialization
	if (sample.empty())
		//if (pow(window_size, 2) > (sample_pair * 2)) //check input value
		{
			sample = init_descriptor(window_size, sample_pair);
		}
		/*else
		{
			cout << "Sample pair number too large";
			return 1;
		}*/

	//window number and BRIEF vector initialization
	flag_col = image.cols - window_size + 1;
	flag_row = image.rows - window_size + 1;
	
	result_size = (int)(flag_col * flag_row);
	rlt = Mat::zeros(flag_row, flag_col, DataType<float>::type); 
	result = new int[result_size];
	int *ptr = result;

	Mat temp;
	int cColour = 0;
	if (colour == 0)
		cColour = 3;
	double th = cal_threshold(image, cColour, 1);
	cout << th << endl;
	
	for (int i = 0; i < flag_row; i++)
		for (int j = 0; j < flag_col; j++)
		{
			Rect rect( j, i, window_size, window_size);
			image(rect).copyTo(temp);
			*ptr = cal_window_sample(temp, sample, th, colour);
			rlt.at<float>(i, j) = *ptr;
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
	if (colour == 0)
		colour = 1;
	double th = cal_threshold(image, colour, 3);

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

int BRIEF::recreate_NBpicture()
{
	map <int, int>::iterator m1_Iter;
	
	if (NBresult.empty()) 
	{
		cout << "there is no NBresult, run map <int, int> BRIEF::compareHistogramNB" << endl;
		return -1;
	}
	
	
	Mat p_zero, p_one, p_two;
	p_zero = image.clone();
	p_one = image.clone();
	p_two = image.clone();
	re_image = p_one.clone();

	rectangle(p_zero, Point(0, 0), Point(image.cols, image.rows), 0, -1, 8);
	rectangle(p_one, Point(0, 0), Point(image.cols, image.rows), 256, -1, 8);
	rectangle(p_two, Point(0, 0), Point(image.cols, image.rows), 128, -1, 8);
	
	for (int i = 0; i < flag_row; i++)
		for (int j = 0; j < flag_col; j++)
		{
			Rect rect(j * window_size, i * window_size, window_size, window_size);
			m1_Iter = NBresult.find(result[i*flag_row + j]);
			if (m1_Iter->second == 0) {
				p_zero(rect).copyTo(re_image(rect));
			}
			else if (m1_Iter->second == 1) {
				p_one(rect).copyTo(re_image(rect));
			}
			else{
				p_two(rect).copyTo(re_image(rect));
			}
		}

	return 0;
}

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

double BRIEF::compareHistogramWeight(unordered_map<int, int> m1, unordered_map<int, int> n2)
{
	unordered_map <int, int>::iterator m1_Iter;
	unordered_map <int, int>::iterator m2_Iter;
	unordered_map <int, int> m2;

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
	double ti = 0.0;

	for (m1_Iter = m1.begin(); m1_Iter != m1.end(); m1_Iter++)
	{
		m2_Iter = m2.find(m1_Iter->first);
		if (m2_Iter != m2.end())
		{
			int d = (m1_Iter->second) - (m2_Iter->second);
			ti += pow(d, 2) / (double)(m1_Iter->second);
		}
		else 
			ti += m1_Iter->second;
	}

	return (double)ti;
}

double BRIEF::compareHistogramChiSquare(unordered_map<int, int> m1, unordered_map<int, int> m2)
{
	unordered_map <int, int>::iterator m1_Iter;
	unordered_map <int, int>::iterator m2_Iter;

	//Calculate each mean value
	double ti = 0.0;

	for (m1_Iter = m1.begin(); m1_Iter != m1.end(); m1_Iter++)
	{
		m2_Iter = m2.find(m1_Iter->first);
		if (m2_Iter != m2.end())
		{
			int d = (m1_Iter->second) - (m2_Iter->second);
			ti += pow(d, 2) / (double)(m1_Iter->second + m2_Iter->second);
		}
	}

	return (double)ti;
}

int BRIEF::imageSegmentation(Mat test_image, int a_size, map<int, int> texture1, map<int, int> texture2, int w_size, int s_pair, int colour, string loca)
{
	image = test_image;
	calBRIEFOverlap(w_size, s_pair, colour);
	int histSize = pow(2, s_pair);

	cout << histSize << endl;
	float range[] = {0, histSize };
	const float* histRange = { range };
	bool uniform = true; bool accumulate = false;
	rlt.convertTo(rlt, CV_32FC1);

	//int a_size = 10;
	int r_rows = rlt.rows - a_size + 1;
	int r_cols = rlt.cols - a_size + 1;
	Mat collect = Mat::zeros(r_rows, r_cols, DataType<double>::type);
	//Mat t1 = mapConvertToMat(texture1, histSize);
	//Mat t2 = mapConvertToMat(texture2, histSize);

	
	cout << "data prepared" << endl;
	vector<map<int, int>> histogramTable;
	//vector<Mat> histogramTable;
	map<int, int> histogramTemp;
	double r1, r2;
	Mat temp;
	Mat temp_hist;
	
	for (int i = 0; i < r_rows; i++)
	{
		for (int j = 0; j < r_cols; j++)
		{
			Rect rect(j, i, a_size, a_size);
			rlt(rect).copyTo(temp);
			//calcHist(&temp, 1, 0, Mat(), temp_hist, 1, &histSize, &histRange, uniform, accumulate);

			histogramTemp = accHistogram(temp);
			histogramTable.push_back(histogramTemp);
			//histogramTable.push_back(temp_hist);
			histogramTemp.clear();
			//cleanMap();
		}
	}

	//cout << histogramTable.at((flag_row - 3)*(flag_col - 3)).begin()->first << endl;
	cout << "histogram finished" << endl;

	int row = 0;
	cout << histogramTable.size() << endl;
	//for (int i = 0; i < histogramTable.size(); i++)
	Concurrency::parallel_for(size_t(0), size_t(histogramTable.size()), [&](size_t i)
	{
		int col = i % r_cols;
		int row = i / r_cols;
		r1 = compareHistogramChiSquare(histogramTable[i], texture1);
		r2 = compareHistogramChiSquare(histogramTable[i], texture2);
		if (r1 < r2)
			collect.at<double>(row, col) = 1;
		else
			collect.at<double>(row, col) = 2;
	});
	/*
	int h_count = 0;
	Concurrency::parallel_for(size_t(0), size_t(r_rows), [&](size_t i)
	{
		for (int j = 0; j < r_cols; j++)
		{
			//writeFile(histogramTable[h_count]);
			//r1 = compareHist(t1, histogramTable[h_count], 1);
			//r2 = compareHist(t2, histogramTable[h_count], 1);
			r1 = compareHistogramChiSquare(histogramTable[h_count], texture1);
			r2 = compareHistogramChiSquare(histogramTable[h_count], texture2);
			if (r1 < r2)
				collect.at<double>(i, j) = 1;
			else
				collect.at<double>(i, j) = 2;
			h_count++;
		}
		cout << i << " " ;
	});
	*/
	cout << "compareHistogramChiSquare" << endl;
	vector<map<int, int>>().swap(histogramTable);
	//re_image = collect.clone();
	//writeFile(collect);
	writeFile(collect, loca);

	return 0;
}

map<int, int> BRIEF::accHistogram()
{
	map<int, int>::iterator h_It;

	int * ptr = result;
	for (int i = 0; i < result_size; i++)
	{	
		//histogram
		h_It = histogram.find((*ptr));
		if (h_It == histogram.end())
			histogram.insert(pair <int, int>(*ptr, 1));
		else 
			(h_It->second)++;
		ptr++;
	}
	map<int, int> h = histogram;
	return h;
}

map<int, int> BRIEF::accHistogram(Mat input)
{
	map<int, int>::iterator h_It;
	map<int, int> h;

	//int * ptr = result;
	for (int i = 0; i < input.rows; i++)
		for (int j = 0; j < input.cols; j++)
		{
			//histogram
			h_It = h.find((input.at<float>(i,j)));
			if (h_It == h.end())
				h.insert(pair <int, int>((input.at<float>(i, j)), 1));
			else
				(h_It->second)++;
		}

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

map<int, int> BRIEF::histogramAdd(map<int, int> in1, map<int, int> in2)
{
	map<int, int> sum;
	map<int, int>::iterator m1, m2;
	int bin_number = pow(2, sample_pair);

	for (int i = 0; i < bin_number; i++)
	{
		m1 = in1.find(i);
		m2 = in2.find(i);

		if (m1 != in1.end())
		{
			if (m2 != in2.end())
				sum.insert(pair<int, int>(m1->first, (m1->second + m2->second)));
			else
				sum.insert(pair<int, int>(m1->first, m1->second));
		}
		else if (m2 != in2.end())
		{
			sum.insert(pair<int, int>(m2->first, m2->second));
		}
	}

	return sum;
}

map <int, int> BRIEF::histogramMinus(map<int, int> in1, map<int, int> in2)
{
	map<int, int> sum;
	map<int, int>::iterator m1, m2;
	int bin_number = pow(2, sample_pair);
	int result;

	for (int i = 0; i < bin_number; i++)
	{
		m1 = in1.find(i);
		m2 = in2.find(i);

		if (m1 != in1.end())
		{
			if (m2 != in2.end())
			{
				result = abs(m1->second - m2->second);
				sum.insert(pair<int, int>(m1->first, result));
			}
			else
				sum.insert(pair<int, int>(m1->first, m1->second));
		}
		else if (m2 != in2.end())
		{
			sum.insert(pair<int, int>(m2->first, m2->second));
		}
	}

	return sum;
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

unordered_map<int, int> BRIEF::mapConvertToUnordered(map<int, int> m1)
{
	unordered_map<int, int> output;
	map<int, int>::iterator h_It;
	for (h_It = m1.begin(); h_It != m1.end(); h_It++)
		output.insert(pair<int, int>(h_It->first, h_It->second));

	return output;
}

Mat BRIEF::mapConvertToMat(map<int, int> m1, int hist_size)
{
	Mat output = Mat::zeros(hist_size,1, CV_32FC1);
	map<int, int>::iterator h_It;

	for (h_It = m1.begin(); h_It != m1.end(); h_It++) {
		output.at<float>(h_It->first, 0) = (float)(h_It->second);
	}
	return output;
}

int BRIEF::cleanMap()
{
	histogramS.clear();
	histogram.clear();
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
	strStream << "C:\\Doctor of Philosophy\\"<< input <<"_"<< number << ".txt";
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
	FileStorage file("C:\\Doctor of Philosophy\\sample.txt",FileStorage::WRITE);
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

Mat BRIEF::init_descriptor(int window_size, int sample_pair) {
	//initial descriptor compare type
	Mat sample = Mat::zeros(sample_pair, 4, DataType<float>::type);
	/*double R = window_size / 2;
	
	Mat mask = Mat::ones(sample_pair, 4, CV_32FC1);
	mask = mask * R;

	for (int i = 0; i < sample_pair - 1; i++)
	{
		sample.at<float>(i, 2) = R * cos(i * 2.0 * M_PI / sample_pair);
		sample.at<float>(i, 3) = R * sin(i * 2.0 * M_PI / sample_pair);
	}
	sample = mask + sample;

	cout << sample << endl;
	*/
	//cv::theRNG().state = time(NULL);
	RNG rng;
	rng.fill(sample, RNG::NORMAL, (window_size - 1) / 2, window_size / 5, false); // Gaussian 1
	//rng.fill(sample, RNG::UNIFORM, 0, window_size, false); // random 1
	
	//randn(sample, Scalar(0), Scalar(window_size)); // random 2
	cout << sample << endl;
	//Gaussian 2
	/*
	Mat sx1(sample_pair, 1, CV_32F);
	Mat sx2(sample_pair, 1, CV_32F);
	Mat sy1(sample_pair, 1, CV_32F);
	Mat sy2(sample_pair, 1, CV_32F);
	RNG rng;
	rng.fill(sx1, RNG::NORMAL, (window_size - 1) / 2, window_size / 5, false);
	rng.fill(sx2, RNG::NORMAL, (window_size - 1) / 2, window_size / 5, false);
	rng.fill(sy1, RNG::NORMAL, 0, window_size / 7, false);
	rng.fill(sy2, RNG::NORMAL, 0, window_size / 7, false);
	sy1 = sy1 + sx1;
	sy2 = sy2 + sx2;
	//rng.fill(sample, RNG::UNIFORM, 0, window_size, false);
	
	sample.push_back(sx1);
	sample.push_back(sy1);
	sample.push_back(sx2);
	sample.push_back(sy2);
	transpose(sample, sample);
	sample = sample.reshape(1, 4);
	transpose(sample, sample);
	*/
	for (int i = 0; i < sample_pair; i++)
	{
		for (int j = 0; j < 4; j++)
		{
			if (sample.at<float>(i, j) > (float)(window_size - 1))
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
		}
	}
	
	return sample;
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
	double threshold = stddev.val[0] * 2;

	return threshold;
} 

Mat BRIEF::gaussian_kernel(int sigma, int dim)
{
	Mat K(dim, dim, CV_32FC1);
	//gaussian kernel 
	float s2 = (float)2.0 * sigma * sigma;
	double sumK = 0.0;

	for (int i = (-dim); i <= dim; i++)
	{
		int m = i + dim;
		for (int j = (-dim); j <= dim; j++)
		{
			int n = j + dim;
			float v = (float)exp(-(1.0*i*i + 1.0*j*j) / s2) / (M_PI*s2);
			K.ptr<float>(m)[n] = v;
			sumK += v;
		}
	}
	/*
	Scalar all = sum(K);
	Mat gaussK;
	K.convertTo(gaussK, CV_32FC1, (1 / all[0]));
	all = sum(gaussK);*/
	Mat gaussK = K * (1/sumK);

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
			//cout << sample.at<float>(i, 0)<<" "<< sample.at<float>(i, 1) << " " << sample.at<float>(i, 2)<<" "<< sample.at<float>(i, 3) << endl;
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

Mat BRIEF::calc3D(int histSize[3])
{
	float range[2] = { 0, 255 };
	const float * ranges[3] = { range, range, range };
	int channels[3] = { 0, 1, 2 };

	Mat hist;
	calcHist(&image, 1, channels, Mat(), hist, 3, histSize, ranges);
	
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
		gKernel[x + 2] = (exp(-(r*r) / s)) / (sqrt(2 * M_PI) * sigma);
		//gKernel[x + 2] = (exp(-(r*r) / s)) / (M_PI * s);
		sum += gKernel[x + 2];
	}

	for (int i = 0; i < 5; ++i)
			gKernel[i] /= sum;

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

	bitwise_or(src1, mask, image);
	src2.copyTo(image, mask);

	return image;
}

int main(int argc, char** argv)
{

	Mat image;
	Mat image1;
	Mat yoo;
	Mat dataAssamble = Mat::zeros(40, 512, DataType<float>::type);

	stringstream strStream;
	string * location1 = new string[40];
	string * location2 = new string[40];
	string * location3 = new string[40];

	map<int, int>::iterator h_It;
	map<int, int> m1;
	map<int, int> n1;
	map<int, int> m2;
	map<int, int> n2;
	map<int, int> tex1;
	map<int, int> tex2;
	BRIEF brief;
	string it_number[7] = { "01", "02", "08", "10", "16", "19", "24" };
	int cnt = 0;
	int Patch_Size = 9;
	int Sample_Number = 2;

	for (int it_it = 3; it_it < 4; it_it++)
	{

		for (int j = 6; j < 7; j++)
		{
			for (Sample_Number = 14; Sample_Number <= 15; Sample_Number++) 
			{
				strStream.clear();
				for (int i = 0; i < 40; i++)
				{
					strStream << "C:\\Users\\wuq\\Documents\\Exp_distribution\\Result\\Patch_Sample\\9\\T" << it_number[it_it] << it_number[j] << "_" << Sample_Number << "_" << i << "segmentation.txt";
					location3[i] = strStream.str();
					strStream.str("");
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

				//image 1
				for (int i = 0; i < 20; i++)
				{
					brief.image = imread(location1[i], CV_LOAD_IMAGE_GRAYSCALE);
					brief.calBRIEFOverlap(Patch_Size, Sample_Number, 0);
					brief.accHistogramSum();
				}
				m1 = brief.histogramS;
				brief.cleanMap();
				
				for (int i = 20; i < 40; i++)
				{
					brief.image = imread(location1[i], CV_LOAD_IMAGE_GRAYSCALE);
					brief.calBRIEFOverlap(Patch_Size, Sample_Number, 0);
					brief.accHistogramSum();
				}
				tex1 = brief.histogramS;
				brief.cleanMap();
				
				//image 2
				for (int i = 0; i < 20; i++)
				{
					brief.image = imread(location2[i], CV_LOAD_IMAGE_GRAYSCALE);
					brief.calBRIEFOverlap(Patch_Size, Sample_Number, 0);
					brief.accHistogramSum();
				}
				m2 = brief.histogramS;
				brief.cleanMap();
				for (int i = 20; i < 40; i++)
				{
					brief.image = imread(location2[i], CV_LOAD_IMAGE_GRAYSCALE);
					brief.calBRIEFOverlap(Patch_Size, Sample_Number, 0);
					brief.accHistogramSum();
				}
				tex2 = brief.histogramS;
				brief.cleanMap();
				
				int startnumber = 0;
				if (Sample_Number == 14)
					startnumber = 11;
				
				for (int i = startnumber; i < 40; i++)
				{
					image = imread(location1[i], CV_LOAD_IMAGE_GRAYSCALE);
					cout << location1[i] << endl;
					image1 = imread(location2[i], CV_LOAD_IMAGE_GRAYSCALE);
					cout << location2[i] << endl;
					cout << location3[i] << endl;
					brief.image = brief.combine_picture(image, image1, 0);
					if (i < 20)
						brief.imageSegmentation(brief.image, 5, tex1, tex2, Patch_Size, Sample_Number, 0, location3[i]);
					else
						brief.imageSegmentation(brief.image, 5, m1, m2, Patch_Size, Sample_Number, 0, location3[i]);
					brief.cleanMap();
				}
				m1.clear();
				m2.clear();
				tex1.clear();
				tex2.clear();
				brief.sample.release();
			}
		}// for compare group j
	}// for setup group it_it
	
	system("pause");
	cv::waitKey(0);
	return 0;
}