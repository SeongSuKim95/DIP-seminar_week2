# DIP_seminar_week2
Digital Image Processing 2주차 내용을 python으로 구현한 code로, 다음과 같은 항목들을 구현하였음.

1. Spatial domain filtering
  1) convolution 구현
  2) 구현한 convolution 함수를 이용해서 다음을 구현
    (1) Box filtering
    (2) Low-pass Gaussian filtering
    (3) Laplacian
    (4) Unsharp masking and highboost filtering
    (5) Sobel operator 


2. Frequency domain filtering

 1) 2D-Fourier transform 구현
  
 2) FFT는 python package에 구현된 것으로 사용해서 다음을 구현
    (1) Ideal low&high-pass filter
    (2) Gaussian low&high-pass filter
    (3) Butterworth low&high-pass filter
    (4) Notch filter

3. Restoration
  1) Spatial domain restoration
    (1) Median filter 
    (2) Min-max filter
    (3) Midpoint filter
    (4) Alpha-trimmed mean filter
    (5) Adaptive median fitler
  2) Frequency domain restoration
    (1) Inverse filtering
    (2) Wiener filtering
