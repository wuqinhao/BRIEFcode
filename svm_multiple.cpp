#include "stdafx.h"

#include "opencv2/core/core.hpp"

#include "opencv2/ml/ml.hpp"

#include "opencv2/highgui/highgui.hpp"

#include "SVM.h"

#include <iostream>

#include <string>
#include <algorithm>    // std::shuffle
#include <random>       // std::default_random_engine
#include <chrono>       // std::chrono::system_clock
#include <stdio.h>

using namespace std;
using namespace cv;

const int labelno = 10;
//k-NN classifier
void MultiTrain::createConfusionMatrix(int a)

{
	const int labelNo = labelno;
	int slice = 2;//trainDataMat.rows;
				   //Mat confusionMatrix = Mat::zeros(labelNo, labelNo, DataType<int>::type);
	int confusionMatrix[labelNo][labelNo] = { 0 };
	float overall;
	int TP = 0;

	//if (trainDataMat.empty())
	//	this->loadDataSet(filename);
	//cout << "predict start" << endl;

	//Mat shuffle_Mat;
	if (labelMat.rows == 1)
		transpose(labelMat, labelMat);
	labelMat.convertTo(labelMat, CV_32SC1);
	//shuffle_labelMat.convertTo(shuffle_labelMat, CV_32F);
	//hconcat(trainDataMat, shuffle_labelMat, shuffle_Mat);
	//Ptr<TrainData> whole = TrainData::create(trainDataMat, ROW_SAMPLE, labelMat);
	Mat shuffle_trainDataMat = shuffleRows(trainDataMat, labelMat);

	//shuffle_trainDataMat.col((shuffle_trainDataMat.cols-1)).copyTo(shuffle_labelMat);
	//shuffle_trainDataMat(Range(0, shuffle_trainDataMat.rows), Range(0, shuffle_trainDataMat.cols - 1)).copyTo(shuffle_trainDataMat);
	//shuffle_labelMat.convertTo(shuffle_labelMat, CV_32SC1);
	//whole->shuffleTrainTest();
	//whole->setTrainTestSplitRatio((double)slice);

	//cout << "shuffle data " << shuffle_trainDataMat.rows << endl;

	Mat trainfeed, labelfeed, testfeed, tlabelfeed;
	int totalrow = trainDataMat.rows;
	int separate = totalrow / (slice);
	
	shuffle_trainDataMat(Range(0, separate), Range(0, shuffle_trainDataMat.cols - 1)).copyTo(trainfeed);
	shuffle_trainDataMat(Range(0, separate), Range(shuffle_trainDataMat.cols - 1, shuffle_trainDataMat.cols)).copyTo(labelfeed);
	shuffle_trainDataMat(Range(separate, shuffle_trainDataMat.rows), Range(0, shuffle_trainDataMat.cols - 1)).copyTo(testfeed);
	shuffle_trainDataMat(Range(separate, shuffle_trainDataMat.rows), Range(shuffle_trainDataMat.cols - 1, shuffle_trainDataMat.cols)).copyTo(tlabelfeed);
	
	/*
	shuffle_trainDataMat(Range(0, separate), Range(0, shuffle_trainDataMat.cols - 1)).copyTo(testfeed);
	shuffle_trainDataMat(Range(0, separate), Range(shuffle_trainDataMat.cols - 1, shuffle_trainDataMat.cols)).copyTo(tlabelfeed);
	shuffle_trainDataMat(Range(separate, shuffle_trainDataMat.rows), Range(0, shuffle_trainDataMat.cols - 1)).copyTo(trainfeed);
	shuffle_trainDataMat(Range(separate, shuffle_trainDataMat.rows), Range(shuffle_trainDataMat.cols - 1, shuffle_trainDataMat.cols)).copyTo(labelfeed);
	*/
	shuffle_trainDataMat.release();

	//cout << "Training cross validation" << endl;
	//cout << trainfeed.rows << endl;
	//cout << labelfeed.rows << endl;
	//cout << testfeed.rows << endl;
	//cout << tlabelfeed.rows << endl;
	labelfeed.convertTo(labelfeed, CV_32SC1);
	tlabelfeed.convertTo(tlabelfeed, CV_32SC1);
	Ptr<TrainData> td = TrainData::create(trainfeed, ROW_SAMPLE, labelfeed);
	//td->setTrainTestSplitRatio((double)slice, true);
	//td->setTrainTestSplit(10.0,true);
	//Mat trainD = td->getTrainSamples();
	//Ptr<TrainData> trainPtr = TrainData::create(td->getTrainSamples(), ROW_SAMPLE, td->getTrainSampleIdx());
	//svm->trainAuto(td);

	//Mat testLabel32F;
	//tlabelfeed.convertTo(testLabel32F, CV_32F);
	Ptr<TrainData> data =
		TrainData::create(testfeed,
			ROW_SAMPLE,
			tlabelfeed);
	Mat res;
	//Ptr<TrainData> testPtr = TrainData::create(td->getTestResponses(), ROW_SAMPLE, td->getTestSampleIdx());
	//float err = svm->calcError(data, true, res);
	//cout << 100 - err << endl;
	//cout << res << endl;
	//cout << tlabelfeed << endl;
	
	//KNN
	int K = 1;
	//Mat response, dist;
	Ptr<ml::KNearest>  knn(ml::KNearest::create());
	knn->train(td);
	knn->findNearest(testfeed, K, res);
	//cout << "Performing test scheme" << endl;
	//cout << res.at<float>(3,0) << endl;
	//int res;
	TP = 0;
	for (int i = 0; i < tlabelfeed.rows; i++)

	{

		//Mat row;
		//testfeed.row(i).copyTo(row);

		//int res = (int)round(svm->predict(row));

		//res = response.at<float>(i,0);
		//cout << (int)res.at<float>(i, 0) << "   " << tlabelfeed.at<int>(i, 0) << endl;
		//confusionMatrix[tlabelfeed.at<int>(i, 0)][(int)res.at<float>(i, 0)]++;
		//confusionMatrix.at<int>((int)tlabelfeed.at<int>(i, 0),res)++;
		if ((int)res.at<float>(i,0) == tlabelfeed.at<int>(i, 0))
			TP++;
		//row.release();
	}

	std::cout << "-------------------------------------" << endl;
	/*
	cout << "----------confusion matrix-----------" << endl;

	for (int i = 0; i<labelNo; i++)

	{

		overall = calcSum(confusionMatrix[i], labelNo);

		for (int j = 0; j<labelNo; j++)

		{

			//double perc = (double)confusionMatrix.at<float>(i,j) * 100 / overall.at<float>(i, 0);
			//double perc = (double)confusionMatrix.at<int>(i, j) * 100 / overall;
			double perc = (double)confusionMatrix[i][j] * 100 / overall;
			cout << perc << "\t";

		} cout << endl;

	} cout << "-------------------------------------" << endl;
	*/
	ACC = (double)TP / (double)testfeed.rows;

	std::cout << "accuracy:" << this->ACC << endl;

}

/*int main(int argc, char * argv[])

{

	MultiTrain inot = MultiTrain::MultiTrain();
	stringstream strStream;
	
	for (int i = 0; i < labelno; i += 1) {
		//strStream << "C:\\Doctor of Philosophy\\brodatz\\" << i << ".xml";
		//inot.loadDataSet(strStream.str());
		//inot.loadDataSet(strStream.str(), 0);
		//strStream.str("");
		strStream << "C:\\Doctor of Philosophy\\outex_tc12\\train\\" << i << ".xml";
		inot.loadDataSet(strStream.str());
		strStream.str("");
		//strStream << "C:\\Doctor of Philosophy\\colour\\R\\" << j << ".xml";
		//inot.loadDataSet(strStream.str());
		//strStream.str("");
	}
	inot.labelMat = inot.labelMat - 1;
	//inot.testResponse = inot.testResponse - 1;
	//for (int i = 0; i < inot.trainDataMat.rows; i++)
	//	normalize(inot.trainDataMat.row(i), inot.trainDataMat.row(i), 1, NORM_L1);
	//for (int i = 0; i < inot.testDataMat.rows; i++)
	//	normalize(inot.testDataMat.row(i), inot.testDataMat.row(i), 1, NORM_L1);
	
	//knn2
	if (inot.labelMat.rows == 1) {
		transpose(inot.labelMat, inot.labelMat);
		//transpose(inot.testResponse, inot.testResponse);
	}
	inot.labelMat.convertTo(inot.labelMat, CV_32SC1);
	//inot.testResponse.convertTo(inot.testResponse, CV_32SC1);
	int K = 1;
	Ptr<TrainData> td = TrainData::create(inot.trainDataMat, ROW_SAMPLE, inot.labelMat);
	//Ptr<TrainData> td = TrainData::create(inot.testDataMat, ROW_SAMPLE, inot.testResponse);
	//Mat response, dist;
	Ptr<ml::KNearest>  knn(ml::KNearest::create());
	knn->setIsClassifier(true);
	knn->setAlgorithmType(KNearest::Types::BRUTE_FORCE);
	//knn->setAlgorithmType(KNearest::Types::KDTREE);
	knn->train(td);
	std::cout << "finish" << endl;
	inot.cleanMap();


	for (int i = 0; i < labelno; i += 1) {
		strStream << "C:\\Doctor of Philosophy\\outex_tc12\\test\\" << i << ".xml";
		inot.loadDataSet(strStream.str());
		strStream.str("");
	}
	inot.labelMat = inot.labelMat - 1;
	transpose(inot.labelMat, inot.labelMat);
	//for (int i = 0; i < inot.trainDataMat.rows; i++)
	//	normalize(inot.trainDataMat.row(i), inot.trainDataMat.row(i), 1, NORM_L1);

	Mat res;
	//cout << inot.labelMat << endl;
	//cout << inot.trainDataMat << endl;
	for (int j = 0; j < 15; j++) {
		//cout << "Data normalized" << endl;
		//knn3
		//knn->findNearest(inot.testDataMat, K, res);
		knn->findNearest(inot.trainDataMat, K, res);
		double TP = 0.0;
		//cout << res << "  " << inot.testResponse<< endl;
		//for (int i = 0; i < inot.testResponse.rows; i++) {
		for (int i = 0; i < inot.labelMat.rows; i++) {
			int a = (int)res.at<float>(i, 0);
			int b = inot.labelMat.at<int>(i, 0);
			//int b = inot.testResponse.at<int>(i, 0); 
			//cout << a << "  " << b << endl;
			if (a==b)
				TP++;
			
		}
		//cout << inot.testResponse.rows << endl;
		//cout << TP << endl;
		double ACC = TP / inot.labelMat.rows;
		//double ACC = TP / inot.testResponse.rows;
		std::cout << ACC << endl;
		
		//inot.createConfusionMatrix(0);
	//inot.trainModel("C:\\Doctor of Philosophy\\outex_tc12\\seg_svm_model_linear.xml");
	}
	inot.cleanMap();
	std::cin.get();
	return 0;

}*/

/*int main(int argc, char * argv[])

{

	MultiTrain inot = MultiTrain::MultiTrain();
	stringstream strStream;
	//for (int j = 5; j <= 15; j++) {
		for (int i = 0; i < labelno; i += 1) {
			strStream << "C:\\Doctor of Philosophy\\brodatzRot\\" << i << ".xml";
			inot.loadDataSet(strStream.str());
			strStream.str("");
			//strStream << "C:\\Doctor of Philosophy\\outex_tc12\\train\\" << i << ".xml";
			//inot.loadDataSet(strStream.str());
			//strStream.str("");
			//strStream << "C:\\Doctor of Philosophy\\colour\\R\\" << j << ".xml";
			//inot.loadDataSet(strStream.str());
			//strStream.str("");
		}
		inot.labelMat = inot.labelMat - 1;
		for (int i = 0; i < inot.trainDataMat.rows; i++)
			normalize(inot.trainDataMat.row(i), inot.trainDataMat.row(i), 1, NORM_L1);

		cout << "Data normalized" << endl;
		for (int i = 0; i < 10; i++)
			inot.createConfusionMatrix(0);
		//inot.trainModel("C:\\Doctor of Philosophy\\outex_tc12\\seg_svm_model_linear.xml");
		inot.cleanMap();
	//}
	cin.get();
	return 0;

}*/

/*int main(int argc, char * argv[])

{

	MultiTrain inot = MultiTrain::MultiTrain();
	stringstream strStream;
	//for (int j = 5; j <= 15; j++) {
	for (int i = 0; i < labelno; i += 1) {
		strStream << "C:\\Doctor of Philosophy\\outex_tc12\\train\\" << i << ".xml";
		inot.loadDataSet(strStream.str());
		strStream.str("");
		//strStream << "C:\\Doctor of Philosophy\\colour\\R\\" << j << ".xml";
		//inot.loadDataSet(strStream.str());
		//strStream.str("");
	}
	inot.labelMat = inot.labelMat - 1;
	
	for (int i = 0; i < inot.trainDataMat.rows; i++)
		normalize(inot.trainDataMat.row(i), inot.trainDataMat.row(i), 1, NORM_L1);

	cout << "Data normalized" << endl;
	//inot.createConfusionMatrix("C:\\Doctor of Philosophy\\umd.xml");
	//inot.trainModel("C:\\Doctor of Philosophy\\outex_tc12");
	inot.testModel("C:\\Doctor of Philosophy\\outex_tc12\\seg_svm_model_linear.xml");
	inot.cleanMap();
	//}
	cin.get();
	return 0;

}*/

//#####################################################################################

using namespace cv;

using namespace std;

void MultiTrain::cleanMap() {
	trainDataMat.release();
	labelMat.release();
}

MultiTrain::MultiTrain()

{
	//this->trainData = 0;
	//this->labels = 0;
	svm->setType(SVM::C_SVC);
	svm->setKernel(SVM::LINEAR);//POLY, LINEAR, LINEAR
	svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 1e6, 0.1));
	knn->setIsClassifier(true);
	knn->setAlgorithmType(KNearest::Types::BRUTE_FORCE);
	//knn->setDefaultK(100);
}

MultiTrain::MultiTrain(Mat traindata, Mat response)
{

	traindata.copyTo(trainDataMat);
	response.copyTo(labelMat);
	trainDataMat.convertTo(trainDataMat, CV_32F);
	labelMat.convertTo(labelMat, CV_32SC1);
	if (labelMat.rows == 1)
		transpose(labelMat, labelMat);
	std::cout << trainDataMat.rows << "    " << labelMat.rows << endl;

	svm->setType(SVM::C_SVC);
	svm->setKernel(SVM::CHI2);//POLY, LINEAR
	svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 1e8, 1.0000000116860974e-7));
}

MultiTrain::MultiTrain(Mat traindata, Mat response, Mat testdata, Mat tlabel) 
{

	traindata.copyTo(trainDataMat);
	response.copyTo(labelMat);
	trainDataMat.convertTo(trainDataMat, CV_32F);
	labelMat.convertTo(labelMat, CV_32SC1);
	if (labelMat.rows == 1)
		transpose(labelMat, labelMat);

	testdata.copyTo(testDataMat);
	tlabel.copyTo(testResponse);
	trainDataMat.convertTo(trainDataMat, CV_32F);
	labelMat.convertTo(labelMat, CV_32SC1);
	if (labelMat.rows == 1)
		transpose(labelMat, labelMat);

	svm->setType(SVM::C_SVC);
	svm->setKernel(SVM::LINEAR);//POLY, CHI2
	svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 1e8, 1.0000000116860974e-7));
}

MultiTrain::~MultiTrain()

{

	//if (this->trainData) cvReleaseMat(&this->trainData);
	trainDataMat.release();
	labelMat.release();
	//if (this->labels) cvReleaseMat(&this->labels);

}

void MultiTrain::loadDataSet(string filename, int a)
//for brodatz or other cases
{
	FileStorage fs;
	fs.open(filename, FileStorage::READ);
	Mat ch_1, ch_2, ch_3/*, ch_4*/;
	fs["data1"] >> ch_1;
	fs["data2"] >> ch_2;
	fs["data3"] >> ch_3;
	//fs["data4"] >> ch_4;
	if (labelMat.empty()) {
		Mat temp1;
		fs["label"] >> temp1;
		temp1.colRange(0, 2).copyTo(labelMat);
		temp1.col(2).copyTo(testResponse);
		//temp1.release();
	}
	else {
		Mat temp;
		fs["label"] >> temp;
		Mat temp1;
		temp.colRange(0, 2).copyTo(temp1);
		Mat matArray[] = { labelMat, temp1 };
		hconcat(matArray, 2, labelMat);
		//delete[] matArray;

		temp.col(2).copyTo(temp1);
		Mat matArray1[] = { testResponse, temp1 };
		hconcat(matArray1, 2, testResponse);
		//delete[] matArray1;
	}
	if (trainDataMat.empty()) {
		Mat matArray[] = { ch_1, ch_2/*, ch_3, ch_4*/ };
		vconcat(matArray, 2, trainDataMat);
		ch_3.copyTo(testDataMat);
		//delete[] matArray;
	}
	else {
		Mat matArray[] = { trainDataMat, ch_1, ch_2/*, ch_3, ch_4*/ };
		vconcat(matArray, 3, trainDataMat);
		//delete[] matArray;
		Mat matArray1[] = { testDataMat, ch_3 };
		vconcat(matArray1, 2, testDataMat);
		//delete[] matArray1;
	}
	trainDataMat.convertTo(trainDataMat, CV_32F);
	labelMat.convertTo(labelMat, CV_32SC1);
	testDataMat.convertTo(testDataMat, CV_32F);
	testResponse.convertTo(testResponse, CV_32SC1);
	//if (labelMat.rows == 1)
	//	transpose(labelMat, labelMat);


	ch_1.release();
	ch_2.release();
	ch_3.release();
	//ch_4.release();
	std::cout << labelMat.cols << endl;
	std::cout << trainDataMat.rows << endl;
	fs.release();
}
/*void MultiTrain::loadDataSet(string filename, int a)
//for brodatz or other cases
{
	FileStorage fs;
	fs.open(filename, FileStorage::READ);
	Mat ch_1, ch_2, ch_3;
	fs["data1"] >> ch_1;
	fs["data2"] >> ch_2;
	fs["data3"] >> ch_3;
	//fs["data4"] >> ch_4;
	if (labelMat.empty()) {
		Mat temp1;
		fs["label"] >> temp1;
		temp1.copyTo(labelMat);
		temp1.col(2).copyTo(testResponse);
		//temp1.release();
	}
	else {
		Mat temp;
		fs["label"] >> temp;
		Mat temp1;
		temp.copyTo(temp1);
		Mat matArray[] = { labelMat, temp1 };
		hconcat(matArray, 2, labelMat);
		//delete[] matArray;

		temp.col(2).copyTo(temp1);
		Mat matArray1[] = { testResponse, temp1 };
		hconcat(matArray1, 2, testResponse);
		//delete[] matArray1;
	}
	if (trainDataMat.empty()) {
		Mat matArray[] = { ch_1, ch_2, ch_3 };
		vconcat(matArray, 3, trainDataMat);
		ch_3.copyTo(testDataMat);
		//delete[] matArray;
	}
	else {
		Mat matArray[] = { trainDataMat, ch_1, ch_2, ch_3 };
		vconcat(matArray, 4, trainDataMat);
		//delete[] matArray;
		Mat matArray1[] = { testDataMat, ch_3 };
		vconcat(matArray1, 2, testDataMat);
		//delete[] matArray1;
	}
	trainDataMat.convertTo(trainDataMat, CV_32F);
	labelMat.convertTo(labelMat, CV_32SC1);
	testDataMat.convertTo(trainDataMat, CV_32F);
	testResponse.convertTo(labelMat, CV_32SC1);
	//if (labelMat.rows == 1)
	//	transpose(labelMat, labelMat);


	ch_1.release();
	ch_2.release();
	ch_3.release();
	//ch_4.release();
	std::cout << labelMat.cols << endl;
	std::cout << trainDataMat.rows << endl;
	fs.release();
}*/

void MultiTrain::loadDataSet(string filename)

{
	FileStorage fs;
	fs.open(filename, FileStorage::READ);
	Mat ch_1, ch_2, ch_3, ch_4;
	fs["data1"] >> ch_1;
	fs["data2"] >> ch_2;
	fs["data3"] >> ch_3;
	fs["data4"] >> ch_4;
	if (labelMat.empty()) {
		Mat temp1;
		fs["label"] >> temp1;
		temp1.copyTo(labelMat);
		temp1.release();
	}
	else {
		Mat temp;
		fs["label"] >> temp;
		Mat matArray[] = { labelMat, temp };
		hconcat(matArray, 2, labelMat);
		temp.release();
	}
	if (trainDataMat.empty()){
		Mat matArray[] = { ch_1, ch_2, ch_3, ch_4 };
		vconcat(matArray, 4, trainDataMat);
	}
	else {
		Mat matArray[] = { trainDataMat, ch_1, ch_2, ch_3, ch_4};
		vconcat(matArray, 5, trainDataMat);
	}
	trainDataMat.convertTo(trainDataMat, CV_32F);
	labelMat.convertTo(labelMat, CV_32SC1);	

	ch_1.release();
	ch_2.release();
	ch_3.release();
	ch_4.release();
	cout << labelMat.cols << endl;
	cout << trainDataMat.rows << endl;
	fs.release();
}

void MultiTrain::trainModel(string outputdir)

{
	//cout << "Preparing training data from legacy" << endl;
	if (labelMat.rows == 1)
		transpose(labelMat, labelMat);
	Ptr<TrainData> td = TrainData::create(trainDataMat, ROW_SAMPLE, labelMat);
	cout << "training the SVM classifier......" << trainDataMat.rows<< "   "<< labelMat.rows<< endl;
	td->shuffleTrainTest();
	svm->trainAuto(td);
	svm->save((outputdir).c_str());

	cout << "SVM model saved to file: " << "_svm_model.xml" << endl;

}

/*void MultiTrain::trainModel(Mat input, Mat response)

{
	cout << "Preparing training data" << endl;

	Ptr<TrainData> td = TrainData::create(trainDataMat, ROW_SAMPLE, labelMat);
	cout << "training the SVM classifier......" << endl;
	svm->train(td);
	//svm->save((outputdir + "/_svm_model.xml").c_str());

	cout << "SVM model trained" <<endl;

}*/

void MultiTrain::loadModel(string filename)

{

	svm = Algorithm::load<SVM>(filename);

}

void MultiTrain::calculateTrainDatacount(int tab[])

{

	for (int i = 0; i< this->labelMat.rows; i++)

	{

		tab[labelMat.at<int>(0,i)]++;

	}

}

void MultiTrain::prepareSets(int class_counts[], int counts[], int part, int parts, CvMat *traindata, CvMat *trainlabels,

	CvMat *testdata, CvMat *testlabels)

{

	int class_integral[6] = { 0 };

	for (int i = 1; i<6; i++)

	{

		class_integral[i] = calcSum(class_counts, 0, i - 1);

	}

	int test_iter, train_iter;

	test_iter = train_iter = 0;

	int type = -1;

	for (int i = 0; i<this->trainDataMat.rows; i++)

	{

		if (i < (class_integral[1] + class_counts[1]))

			type = 1;

		else if (i < (class_integral[2] + class_counts[2]))

			type = 2;

		else if (i < (class_integral[3] + class_counts[3]))

			type = 3;

		else if (i < (class_integral[4] + class_counts[4]))

			type = 4;

		else if (i < (class_integral[5] + class_counts[5]))

			type = 5;

		else if (i < (class_integral[6] + class_counts[6]))

			type = 6;

		//else if(i < (class_integral[6]+class_counts[6]))

		//type = 6;

		if (type >= 0)

		{

			//CvMat *r = cvCreateMat(1, 656, CV_32FC1);
			Mat r;

			//cvGetRow(this->trainData, r, i);
			trainDataMat.row(i).copyTo(r);

			if ((i >= (part*counts[type] + class_integral[type])) && (i<((part + 1)*counts[type]

				+ class_integral[type])))

			{

				for (int j = 0; j< r.cols; j++)

				{

					//CV_MAT_ELEM(*testdata, float, test_iter, j) = r->data.fl[j];
					CV_MAT_ELEM(*traindata, float, train_iter, j) = r.at<int>(0, j);
					CV_MAT_ELEM(*testlabels, int, test_iter, 0) = type;

				}

				test_iter++;

			}
			else

			{

				for (int j = 0; j<r.cols; j++)

				{

					//CV_MAT_ELEM(*traindata, float, train_iter, j) = r->data.fl[j];
					CV_MAT_ELEM(*traindata, float, train_iter, j) = r.at<int>(0,j);
					CV_MAT_ELEM(*trainlabels, int, train_iter, 0) = type;

				}

				train_iter++;

			}

		}

	}

}

void MultiTrain::performCrossValidation(int parts)

{

	float sum = 0, av;

	vector<double> test_results;

	int class_counts[6] = { 0 };

	int counts[6] = { 0 };

	if (parts != 0)

	{

		calculateTrainDatacount(class_counts);

		for (int i = 0; i<6; i++)

		{

			counts[i] = (class_counts[i] / parts);

		}

	}

	if (parts == 0 || parts == 1) cout << "Cross validation cannot be performed for such input values" << endl;

	else if (calcSum(counts, 6)<6) cout << "The database is too small for performing the " << parts << "-fold crossvalidation." << endl;

	else

	{

		//MAIN LOOP

		for (int p = 0; p<parts; p++)

		{

			//CREATE SETS

			CvMat *traindata = 0;

			CvMat *trainlabels = 0;

			CvMat *testdata = 0;

			CvMat *testlabels = 0;

			testdata = cvCreateMat(calcSum(counts, 6), 656, CV_32FC1);

			testlabels = cvCreateMat(calcSum(counts, 6), 1, CV_32SC1);

			traindata = cvCreateMat(this->trainDataMat.rows - calcSum(counts, 6), 656, CV_32FC1);

			trainlabels = cvCreateMat(this->trainDataMat.rows - calcSum(counts, 6), 1, CV_32SC1);

			//PREPARE SETS

			this->prepareSets(class_counts, counts, p, parts, traindata, trainlabels, testdata, testlabels);

			//PERFORM TRAINING

			cout << "Training the SVM classifier...part#" << p << endl;

			//Mat trainDataMat = cvarrToMat(trainData);
			//Mat labelsMat = cvarrToMat(labels);
			Ptr<TrainData> td = TrainData::create(trainDataMat, ROW_SAMPLE, labelMat);
			cout << "training the SVM classifier......" << endl;
			svm->train(td);

			//PERFORM TESTING

			int TP = 0;

			for (int i = 0; i<(int)traindata->rows; i++)

			{

				CvMat *row = cvCreateMat(1, 656, CV_32FC1);

				cvGetRow(traindata, row, i);

				int res = (int)svm->predict(cvarrToMat(row));

				if (res == trainlabels->data.i[i])

					TP++;

			}

			double accuracy = (double)TP / (double)traindata->rows;

			cout << "accuracy for part#" << p << " : " << accuracy << endl;

			sum = sum + accuracy;

			//RELEASE SETS

			cvReleaseMat(&traindata);

			cvReleaseMat(&trainlabels);

			cvReleaseMat(&testdata);

			cvReleaseMat(&testlabels);

			svm->clear();

			cout << "-----------------------------------------" << endl;

		}

	}

	//av = sum/10;

	//cout << "av=" << av << endl;

}

void MultiTrain::testModel(string filename)

{

	this->loadModel(filename);
	//int confusionMatrix[labelno][labelno] = { 0 };
	float overall;

	int TP = 0; //true prediction counter


	for (int i = 0; i < trainDataMat.rows; i++)

	{

		Mat row;
		trainDataMat.row(i).copyTo(row);

		int res = (int)round(svm->predict(row));

		//res = response.at<float>(i,0);
		cout << res << "   " << labelMat.at<int>(0, i) << endl;
		//confusionMatrix[labelMat.at<int>(0, i)-1][res]++;
		//confusionMatrix.at<int>((int)tlabelfeed.at<int>(i, 0),res)++;
		if (res == labelMat.at<int>(0, i))
			TP++;
		row.release();
	}
	/*
	cout << "-------------------------------------" << endl;

	cout << "----------confusion matrix-----------" << endl;

	for (int i = 0; i<labelno; i++)

	{

		overall = calcSum(confusionMatrix[i], labelno);

		for (int j = 0; j<labelno; j++)

		{

			//double perc = (double)confusionMatrix.at<float>(i,j) * 100 / overall.at<float>(i, 0);
			//double perc = (double)confusionMatrix.at<int>(i, j) * 100 / overall;
			double perc = (double)confusionMatrix[i][j] * 100 / overall;
			cout << perc << "\t";

		} cout << endl;

	} cout << "-------------------------------------" << endl;*/

	ACC = (double)TP / (double)labelMat.cols;

	cout << "accuracy:" << this->ACC << endl;

}

void MultiTrain::createConfusionMatrix(string filename)

{
	const int labelNo = labelno;
	int slice = 10;//trainDataMat.rows;
	//Mat confusionMatrix = Mat::zeros(labelNo, labelNo, DataType<int>::type);
	int confusionMatrix[labelNo][labelNo] = { 0 };
	float overall;
	int TP = 0;

	//if (trainDataMat.empty())
	//	this->loadDataSet(filename);
	cout << "predict start" << endl;

	//Mat shuffle_Mat;
	if (labelMat.rows == 1)
		transpose(labelMat, labelMat);
	labelMat.convertTo(labelMat, CV_32SC1);
	//shuffle_labelMat.convertTo(shuffle_labelMat, CV_32F);
	//hconcat(trainDataMat, shuffle_labelMat, shuffle_Mat);
	//Ptr<TrainData> whole = TrainData::create(trainDataMat, ROW_SAMPLE, labelMat);
	Mat shuffle_trainDataMat = shuffleRows(trainDataMat, labelMat);
	
	//shuffle_trainDataMat.col((shuffle_trainDataMat.cols-1)).copyTo(shuffle_labelMat);
	//shuffle_trainDataMat(Range(0, shuffle_trainDataMat.rows), Range(0, shuffle_trainDataMat.cols - 1)).copyTo(shuffle_trainDataMat);
	//shuffle_labelMat.convertTo(shuffle_labelMat, CV_32SC1);
	//whole->shuffleTrainTest();
	//whole->setTrainTestSplitRatio((double)slice);
	
	cout << "shuffle data " <<shuffle_trainDataMat.rows<< endl;
	
Mat trainfeed, labelfeed, testfeed, tlabelfeed;
	int totalrow = trainDataMat.rows;
	int separate = totalrow / (slice);
	//whole->getTrainSamples().copyTo(trainfeed);
	//whole->getTrainResponses().copyTo(labelfeed);
	//whole->getTest.copyTo(testfeed);
	//whole->getTestResponses().copyTo(tlabelfeed);
	shuffle_trainDataMat(Range(0, separate), Range(0, shuffle_trainDataMat.cols-1)).copyTo(testfeed);
	shuffle_trainDataMat(Range(0, separate), Range(shuffle_trainDataMat.cols - 1, shuffle_trainDataMat.cols)).copyTo(tlabelfeed);
	shuffle_trainDataMat(Range(separate, shuffle_trainDataMat.rows), Range(0, shuffle_trainDataMat.cols-1)).copyTo(trainfeed);
	shuffle_trainDataMat(Range(separate, shuffle_trainDataMat.rows), Range(shuffle_trainDataMat.cols - 1, shuffle_trainDataMat.cols)).copyTo(labelfeed);
	//shuffle_labelMat.release();
	shuffle_trainDataMat.release();
//cout << shuffle_trainDataMat(Range(0, separate), Range(shuffle_trainDataMat.cols - 1, shuffle_trainDataMat.cols)) << endl;
	cout << "Training cross validation" << endl;
	cout << trainfeed.rows << endl;
	cout << labelfeed.rows << endl;
	cout << testfeed.rows << endl;
	cout << tlabelfeed.rows << endl;
	labelfeed.convertTo(labelfeed, CV_32SC1);
	tlabelfeed.convertTo(tlabelfeed, CV_32SC1);
	Ptr<TrainData> td = TrainData::create(trainfeed, ROW_SAMPLE, labelfeed);
	//td->setTrainTestSplitRatio((double)slice, true);
	//td->setTrainTestSplit(10.0,true);
	//Mat trainD = td->getTrainSamples();
	//Ptr<TrainData> trainPtr = TrainData::create(td->getTrainSamples(), ROW_SAMPLE, td->getTrainSampleIdx());
	svm->trainAuto(td);

	//Mat testLabel32F;
	//tlabelfeed.convertTo(testLabel32F, CV_32F);
	Ptr<TrainData> data = 
		TrainData::create(testfeed, 
			ROW_SAMPLE, 
			tlabelfeed);
	Mat res;
	//Ptr<TrainData> testPtr = TrainData::create(td->getTestResponses(), ROW_SAMPLE, td->getTestSampleIdx());
	float err = svm->calcError(data, true, res);
	cout << 100-err << endl;
	cout << res << endl;
	cout << tlabelfeed << endl;
	/*
	//KNN
	int K = 1;
	Mat response, dist;
	Ptr<ml::KNearest>  knn(ml::KNearest::create());
	knn->train(trainfeed, ROW_SAMPLE, labelfeed);
	knn->findNearest(testfeed, K, noArray(), response, dist);
	cout << "Performing test scheme" << endl;
	int res;*/
	for (int i = 0; i < testfeed.rows; i++)

	{

		Mat row;
		testfeed.row(i).copyTo(row);

		int res = (int)round(svm->predict(row));
		
		//res = response.at<float>(i,0);
		cout << res << "   " << tlabelfeed.at<int>(i, 0) << endl;
		confusionMatrix[tlabelfeed.at<int>(i, 0)][res]++;
		//confusionMatrix.at<int>((int)tlabelfeed.at<int>(i, 0),res)++;
		if (res == tlabelfeed.at<int>(i, 0))
			TP++;
		row.release();
	}
	
	cout << "-------------------------------------" << endl;

	cout << "----------confusion matrix-----------" << endl;

	for (int i = 0; i<labelNo; i++)

	{

		overall = calcSum(confusionMatrix[i], labelNo);

		for (int j = 0; j<labelNo; j++)

		{

			//double perc = (double)confusionMatrix.at<float>(i,j) * 100 / overall.at<float>(i, 0);
			//double perc = (double)confusionMatrix.at<int>(i, j) * 100 / overall;
			double perc = (double)confusionMatrix[i][j] * 100 / overall;
			cout << perc << "\t";

		} cout << endl;

	} cout << "-------------------------------------" << endl;
	
	ACC = (double)TP / (double)testfeed.rows;
	
	cout << "accuracy:" << this->ACC << endl;
	
}

float MultiTrain::calcSum(Mat tab, int n)

{

	float rlt = 0.0;

	rlt = sum(tab)[0];

	return rlt;

}

float MultiTrain::calcSum(int *tab, int n)

{

	float sum = 0.0;

	for (int i = 0; i<n; i++)

		sum += tab[i];

	return sum;

}

float MultiTrain::calcSum(int *tab, int idx1, int idx2)

{

	float sum = 0.0;

	for (int i = idx1; i <= idx2; i++)

		sum += tab[i];

	return sum;

}

cv::Mat  MultiTrain::shuffleRows(const cv::Mat& mat, Mat& label)
{
	Mat temp;
	
	mat.convertTo(mat, CV_32F);
	label.convertTo(label, CV_32F);
	hconcat(mat, label, temp);
	

	vector<Mat> vm;
	for (int cont = 0; cont < mat.rows; cont++)
		vm.push_back(temp.row(cont).clone());
	temp.release();

	random_device rd;
	mt19937 g(rd());
	shuffle(vm.begin(), vm.end(), g);

	Mat output;
	for (auto iterator = vm.begin(); iterator != vm.end();)
	{
		output.push_back(vm.front()); 
		iterator = vm.erase(iterator);
	}

	return output;
}