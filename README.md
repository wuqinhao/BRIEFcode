# BRIEFcode
Binary string contribution now is when (p1-p2)>3*sigma. Here, sigma means the noise standard deviation. 
The binary string is processed in BRIEF::cal_window_sample.

Also, this BRIEF code offers the change of BRIEF patch size and sample pair number. The main output is
the BRIEF histogram of the whold image. The outcome can be printed in txt file.

24-02-2017

Debug version of BRIEF project: including overlapping patch, test data mask, and parallel computing. Fixed threshold bug: all using opencv methods now.