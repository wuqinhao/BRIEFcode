# BRIEFcode

Did not offer the .h file. All presented as the raw implementation, so that it could be modified by different needs. The pipeline would be shown in the main function.

Binary string contribution now is when (p1-p2)>3*sigma. Here, sigma means the noise standard deviation. 
The binary string is processed in BRIEF::cal_window_sample.

Also, this BRIEF code offers the change of BRIEF patch size and sample pair number. The main output is
the BRIEF histogram of the whole image. The outcome can be printed as the txt file.

12-12-2018

Process improved: functions were updated by using OpenCV APIs.

24-02-2017

Bug fixed: including overlapping patch, test data mask, and parallel computing. Fixed threshold bug: all using opencv methods now.
