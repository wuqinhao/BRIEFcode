int main(int argc, char** argv)
{
	
	Mat image;
	Mat image1;
	Mat dataAssamble = Mat::zeros(40, 512, DataType<int>::type);
	//Vector<Mat> out;
	stringstream strStream;
	string * location1 = new string[40];
	string * location2 = new string[40];
	string * location3 = new string[25];
	//string number;
	map<int, int>::iterator h_It;
	//map<int, int>::iterator s1_It;
	//map<int, int>::iterator s2_It;
	//double rt[256][80];
	map<int, int> temp, texture1, texture2;
	//map<int, int> sum2;
	map<int, int> m1;
	map<int, int> m2;
	BRIEF brief;
	//double b1 = 0;
	//double b2 = 0;

	//String loca = "C:\\Users\\wuq\\Documents\\Exp_distribution\\Result\\G1\\T01\\T01_sum.txt";
	//sum1 = brief.readFile(loca, sum1);
	
	/*
	ifstream fin("C:\\Users\\wuq\\Documents\\Exp_distribution\\Result\\G1\\T01\\T01_sum.txt");
	if (!fin) 
	{
		std::cerr << "Can't open file " << "C:\\Users\\wuq\\Documents\\Exp_distribution\\Result\\G1\\T01\\T01_sum.txt" << std::endl;
		std::exit(-1);
	}

	string key, value; 
	while (getline(fin, key))
	{
		if (key.empty())
			break;
		stringstream strStream(key);
		strStream >> key >> value;
		cout << stoi(key) << "\t" << stoi(value) << endl;
		sum1.insert(pair <int, int>(stoi(key), stoi(value)));
	}
	//cout << sum1.at(stoi(key)) << endl;
		
	
	fin.close();*/
	
	for (int j = 1; j <= 1; j++)
	{
		strStream.clear();
		if (j <= 9) {
			strStream << "C:\\Users\\wuq\\Documents\\Exp_distribution\\Result\\G2\\T0" << j << "\\T0" << j << "_cbresultTTTTTTTTTTTTT.txt";
			location3[j] = strStream.str();
		}
		else {
			strStream << "C:\\Users\\wuq\\Documents\\Exp_distribution\\Result\\G2\\T" << j << "\\T" << j << "_cbresult20.txt";
			location3[j] = strStream.str();
		}
		strStream.str("");

		if (j <= 9)
		{
			for (int i = 0; i < 40; i++)
			{
				if (i >= 9)
				{
					strStream << "C:\\Users\\wuq\\Documents\\ImageDatabases\\Texture Datasets\\UIUCTex\\T0" << j << "\\T0" << j << "_" << i + 1 << ".jpg";
					location1[i] = strStream.str();
					strStream.str(""); // clean Stringstream
					strStream << "C:\\Users\\wuq\\Documents\\ImageDatabases\\Texture Datasets\\UIUCTex\\T0" << j << "\\T0" << j << "_" << i + 1 << ".jpg";
					location2[i] = strStream.str();
					strStream.str("");
				}
				else
				{
					strStream << "C:\\Users\\wuq\\Documents\\ImageDatabases\\Texture Datasets\\UIUCTex\\T0" << j << "\\T0" << j << "_0" << i + 1 << ".jpg";
					location1[i] = strStream.str();
					strStream.str(""); // clean Stringstream
					strStream << "C:\\Users\\wuq\\Documents\\ImageDatabases\\Texture Datasets\\UIUCTex\\T0" << j << "\\T0" << j << "_0" << i + 1 << ".jpg";
					location2[i] = strStream.str();
					strStream.str("");
				}
			}
			strStream.clear();
		}
		else
		{
			for (int i = 0; i < 40; i++)
			{
				if (i >= 9)
				{
					strStream << "C:\\Users\\wuq\\Documents\\ImageDatabases\\Texture Datasets\\UIUCTex\\T" << j << "\\T" << j << "_" << i + 1 << ".jpg";
					location1[i] = strStream.str();
					strStream.str(""); // clean Stringstream
					strStream << "C:\\Users\\wuq\\Documents\\ImageDatabases\\Texture Datasets\\UIUCTex\\T" << j << "\\T" << j << "_" << i + 1 << ".jpg";
					location2[i] = strStream.str();
					strStream.str("");
				}
				else
				{
					strStream << "C:\\Users\\wuq\\Documents\\ImageDatabases\\Texture Datasets\\UIUCTex\\T" << j << "\\T" << j << "_0" << i + 1 << ".jpg";
					location1[i] = strStream.str();
					strStream.str(""); // clean Stringstream
					strStream << "C:\\Users\\wuq\\Documents\\ImageDatabases\\Texture Datasets\\UIUCTex\\T" << j << "\\T" << j << "_0" << i + 1 << ".jpg";
					location2[i] = strStream.str();
					strStream.str("");
				}
			}
			strStream.clear();
		}
		/*
		m1 = brief.readFile("C:\\Users\\wuq\\Documents\\Exp_distribution\\Result\\G1\\T01\\T01_01.txt", m1);
		m2 = brief.readFile("C:\\Users\\wuq\\Documents\\Exp_distribution\\Result\\G1\\T01\\T01_02.txt", m2);
		temp = brief.histogramAdd(m1, m2);
		m2.clear();
		m1.clear();
		m1 = brief.readFile("C:\\Users\\wuq\\Documents\\Exp_distribution\\Result\\G1\\T01\\T01_04.txt", m1);
		m2 = brief.histogramAdd(m1, temp);
		m1.clear();
		temp.clear();
		m1 = brief.readFile("C:\\Users\\wuq\\Documents\\Exp_distribution\\Result\\G1\\T01\\T01_09.txt", m1);
		texture1 = brief.histogramAdd(m1, m2);
		temp.clear();
		m1.clear();
		m2.clear();
		brief.writeFile(texture1, "C:\\Doctor of Philosophy\\histogram.txt");

		m1 = brief.readFile("C:\\Users\\wuq\\Documents\\Exp_distribution\\Result\\G1\\T09\\T09_01.txt", m1);
		m2 = brief.readFile("C:\\Users\\wuq\\Documents\\Exp_distribution\\Result\\G1\\T09\\T09_02.txt", m2);
		temp = brief.histogramAdd(m1, m2);
		m1.clear();
		m2.clear();
		m1 = brief.readFile("C:\\Users\\wuq\\Documents\\Exp_distribution\\Result\\G1\\T09\\T09_03.txt", m1);
		m2 = brief.histogramAdd(m1, temp);
		m1.clear();
		temp.clear();
		m1 = brief.readFile("C:\\Users\\wuq\\Documents\\Exp_distribution\\Result\\G1\\T09\\T09_06.txt", m1);
		texture2 = brief.histogramAdd(m2, m1);
		brief.writeFile(texture2, "C:\\Doctor of Philosophy\\histogram.txt");
		


		brief.compareHistogramNB(texture1, texture2);*/
		//brief.writeFile(brief.NBresult,"C:\\Doctor of Philosophy\\histogram.txt");
		brief.readFile("C:\\Users\\wuq\\Documents\\Exp_distribution\\Gsample.txt");
		for (int i = 0; i < 40; i++)
		{
			image = imread(location1[i], CV_LOAD_IMAGE_GRAYSCALE);
			cout << j << location1[i] << endl;
		/*
			image1 = imread(location2[4], CV_LOAD_IMAGE_GRAYSCALE);
			cout << j << location2[4] << endl;
			
			Mat temp1, temp2;
			temp1 = Mat::zeros(image.cols / 2, image.rows, DataType<unsigned>::type);
			temp2 = Mat::zeros(image.cols / 2, image.rows, DataType<unsigned>::type);
			image(Rect(image.cols / 2, 0, image.cols / 2, image.rows)).copyTo(temp1);
			image1(Rect(0, 0, image.cols / 2, image.rows)).copyTo(temp2);
			brief.image = temp1.clone();
			brief.calBRIEF(7, 6, 0);
			texture1 = brief.accHistogram();
			brief.cleanMap();

			brief.image = temp2.clone();
			brief.calBRIEF(7, 6, 0);
			texture2 = brief.accHistogram();
			brief.cleanMap();
			brief.compareHistogramNB(texture1, texture2);
			
			namedWindow("temp1", WINDOW_AUTOSIZE);
			imshow("temp1", temp1);
			namedWindow("temp2", WINDOW_AUTOSIZE);
			imshow("temp2", temp2);
			namedWindow("test", WINDOW_AUTOSIZE);
			imshow("test", brief.image);
			*/
			//brief.combine_picture(image, image1, 0);

			brief.calBRIEF(7, 9, 0);
			m1 = brief.accHistogram();
			/*brief.recreate_NBpicture();
			namedWindow("test", WINDOW_AUTOSIZE);
			imshow("test", brief.image);
			namedWindow("result", WINDOW_AUTOSIZE);
			imshow("result", brief.re_image);*/
			for (int k = 0; k < 512; k++)
			{
				h_It = m1.find(k);
				if (h_It != m1.end())
					dataAssamble.at<int>(i, k) = h_It->second;
			}
			m1.clear();
			//imwrite("C:/Doctor of Philosophy/NBresult.jpg", brief.re_image);
			//brief.writeFile(sum1, location3[j]);
		}
		brief.writeFile(dataAssamble);
		//brief.writeFile(sum1, location3[j]);
		//brief.histogramS.clear();

	}


	/*
	BRIEF test;
	Mat image1 = imread("C:\\Users\\wuq\\Documents\\ImageDatabases\\Texture Datasets\\UIUCTex\\T03\\T03_03.jpg", CV_LOAD_IMAGE_GRAYSCALE);
	Mat image2 = imread("C:\\Users\\wuq\\Documents\\ImageDatabases\\Texture Datasets\\UIUCTex\\T13\\T13_01.jpg", CV_LOAD_IMAGE_GRAYSCALE);


	//BRIEF test1(image1, result);
	Mat cp = test.combine_picture(image1, image2, 1);
	//test1.readFile("C:\\Doctor of Philosophy\\sample.txt");

	if (!image1.data)                              // Check for invalid input
	{
		cout << "Could not open or find the image" << std::endl;
		return -1;
	}


	//test.combine_picture(image1, image2, 1);
	namedWindow("w", WINDOW_AUTOSIZE);
	imshow("w", cp);
	cout<<"Fin!"<<endl;*/
	
	cv::waitKey(0);
	return 0;

}