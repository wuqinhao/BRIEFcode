# BRIEFcode
Binary string contribution now is when (p1-p2)>3*sigma. Here, sigma means the noise standard deviation. 
The binary string is processed in BRIEF::cal_window_sample.

Also, this BRIEF code offers the change of BRIEF patch size and sample pair number. The main output is
the BRIEF histogram of the whold image. The outcome can be printed in txt file.

23-12-2016 update

1. BRIEF class changed stored in Threshold.cpp. It is the 	quick fixed version to set up threshold as a parameter in 	the class.