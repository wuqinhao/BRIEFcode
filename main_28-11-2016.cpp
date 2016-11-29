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
	int Patch_Size = 5;
	int Sample_Number = 2;

	for (int it_it = 3; it_it < 4; it_it++)
	{

		for (int j = 6; j < 7; j++)
		{
			for (Sample_Number = 11; Sample_Number <= 15; Sample_Number++) 
			{
				strStream.clear();
				for (int i = 0; i < 40; i++)
				{
					strStream << "C:\\Users\\wuq\\Documents\\Exp_distribution\\Result\\Patch_Sample\\T" << it_number[it_it] << it_number[j] << "_" << Sample_Number << "_" << i << "segmentation.txt";
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