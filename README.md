# BRIEFcode
Binary string contribution now is when (p1-p2)>3*sigma. Here, sigma means the noise standard deviation. 
The binary string is processed in BRIEF::cal_window_sample.

Also, this BRIEF code offers the change of BRIEF patch size and sample pair number. The main output is
the BRIEF histogram of the whold image. The outcome can be printed in txt file.

28-11-2016 update

1. Bug fixed
	*BRIEF number calculation for each patch
	*Mat datatype are now mainly <unsigned>, which cannot be 	 shown by imshow(). Alternatively, use BRIEF::writeFile() 	 to pull out Mat matrix.
	*Sample generated can overlap now.

2. New method add on
	*calBRIEFOverlap() - calculate BRIEF patch pixel by 	pixel;
	*compareHistogramNB() - create Naive Bayes classifier 	based on two texture's model histograms;
	*imageSegmentation() - image segmentation and generated 	result using Mat, based on different methods; (now is 	mainly based on Chi-squared);
	*combine_picture() - combine two images together based 	on two masks: circle and square;
	*recreate_NBpicture() - recreate result picture after 	segmentation;
	*histogramAdd() & histogramMinus() - add and minus 	calculation for two histograms.

3. Main function
	*It is used to evaluate how Sample_Number affecting 	segmentation performance based on fixed Patch_Size and 
        Sample_Number. The result is shown as a .txt file.
