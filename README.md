# BRIEFcode

BRIEF class is packed and prepared in the BRIEF.h file. 

All presented as the raw implementation in C++ with OpenCV (v3.0) and OpenMP (v4.5), so that it could be modified by different needs. The pipeline would be shown in the main function btest.cpp. The accept image type is the general opencv accepting types. Any further updates will be listed below.

Binary string contribution now is when (p1-p2)> Threshold, calculated in BRIEF::cal_threshold.

The binary string is processed in BRIEF::cal_window_sample.

Also, this BRIEF code offers the change of BRIEF patch size and sample pair number. The main output is
the BRIEF histogram of the whole image. The outcome can be printed as the txt file. SVM and Nearest Neighbour classifier is offered in svm_multiple.cpp. The files are saved in xml format. 

07-02-2019

Upload BRIEF.h including class implementation.
Threshold calculation replaced by 3*sigma+abs(mean): higher threshold for better classification result.
btest.cpp archived: legacy code.

12-12-2018

Process improved: functions were updated by using OpenCV APIs.

24-02-2017

Bug fixed: including overlapping patch, test data mask, and parallel computing. Fixed threshold bug: all using opencv methods now.
