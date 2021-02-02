# DIP_various_filter
## This repository is python code of "Digital Image Processing, Rafael C.Gonzalez, 3rd edition" Chapter 3 & 4

* Spatial domain filtering
  * 2D Convolution 
  * Spatial filtering    
    * Box filter (Smoothing linear filter) 
      * Original image
      
      ![original](https://user-images.githubusercontent.com/62092317/106533707-01eed200-6536-11eb-9791-48681df419d2.PNG)
      * Low pass filter
      ![low_box](https://user-images.githubusercontent.com/62092317/106533689-fdc2b480-6535-11eb-9bfd-5ed3490adc85.PNG)
      ```python

      ```
      * High pass filter
      ```python
      
      ```
    * Laplacian filter
      * Before Normalizing
      ![Laplacian](https://user-images.githubusercontent.com/62092317/106533685-fd2a1e00-6535-11eb-926c-bd97658ffbbe.PNG)
      ```python
      
      ```
      * After Nomalizing
      ![Laplacian_Scaling](https://user-images.githubusercontent.com/62092317/106533688-fd2a1e00-6535-11eb-8d59-dd7bcc6e67b9.PNG)
      ```python
      
      ```
      * Sharpening with normal laplacian filter
      ![Sharpening1](https://user-images.githubusercontent.com/62092317/106533713-02876880-6536-11eb-9ad1-77bbd3a69897.PNG)
      ```python
      
      ```
      * Sharpening with diagonal laplacian filter
      ![Sharpening2](https://user-images.githubusercontent.com/62092317/106533716-031fff00-6536-11eb-854c-b817cba66683.PNG)
      ```python
      
      ```
      * Comparison between normal & diagonal laplacian filter
      ![Sharpening](https://user-images.githubusercontent.com/62092317/106533709-01eed200-6536-11eb-9e4f-53dca2d3b200.PNG)
      ```python
      
      ``
    * Sobel operator
      ![Sobel](https://user-images.githubusercontent.com/62092317/106533719-03b89580-6536-11eb-96e5-c9ccb0a80ffb.PNG)
      ![Sobel_vertical_horizontal](https://user-images.githubusercontent.com/62092317/106533722-04512c00-6536-11eb-9b3b-a70fab8fe350.PNG)
      ```python

      ```
    
    * Unsharp masking and highboost filtering
      * Unsharp masking
      ![Unsharpmasking](https://user-images.githubusercontent.com/62092317/106533724-04e9c280-6536-11eb-8fba-da6128d47ed9.PNG)
      ```python

      ```
      * High boost by Unsharpmasking + Sobel filter
      ![Unsharpmasking+Sobel](https://user-images.githubusercontent.com/62092317/106533728-05825900-6536-11eb-9f22-4e46e9beeea5.PNG)
      ```python
      
      ```

* Frequency domain filtering
  * Ideal low&high-pass filter
    * Low pass filter
    ![Frequency_LPF](https://user-images.githubusercontent.com/62092317/106533676-fa2f2d80-6535-11eb-9a3d-cdd4fec60411.PNG)
    ```python

    ```
    * High pass filter
    ![Frequency_HPF](https://user-images.githubusercontent.com/62092317/106533749-09ae7680-6536-11eb-8344-b3f9b46461f0.PNG)
    ```python

    ```

  * Gaussian low&high-pass filter
    * Low pass filter
    ![Gaussian_LPF](https://user-images.githubusercontent.com/62092317/106533683-fbf8f100-6535-11eb-971b-92a339371d4a.PNG)
    ```python

    ```
    * High pass filter
    ![Gaussian_HPF](https://user-images.githubusercontent.com/62092317/106533680-fbf8f100-6535-11eb-96e5-33c2bdffba38.PNG)
    ```python

    ```
  * Butterworth low&high-pass filter
   * Low pass filter
   ```python

   ```
   * High pass filter
   ![ButterWorth_HPF](https://user-images.githubusercontent.com/62092317/106533748-0915e000-6536-11eb-8e0b-2797eef4dbbf.PNG)
   ```python

   ```
  * Notch filter
    * Frequency components of original image & Required notch filter
    ![Notch_filter](https://user-images.githubusercontent.com/62092317/106533699-00250e80-6536-11eb-950e-bc9c26954890.PNG)
    * Filtered image
    ![Notch_filter_1](https://user-images.githubusercontent.com/62092317/106533701-00bda500-6536-11eb-9142-656d0226afca.PNG)
    * Noise of the image ( Extracted by Inverse notch filter)
    ![Notch_filter_2](https://user-images.githubusercontent.com/62092317/106533704-01563b80-6536-11eb-960a-8a72774c82ff.PNG)
    
    ```python
    
    ```

* Restoration
  * Spatial domain restoration
    * Median filter
    ![median_filter](https://user-images.githubusercontent.com/62092317/106533692-fe5b4b00-6535-11eb-8169-186ddca92125.PNG)
    ```python
    
    ```
    * Min-max filter
      * Min filter
      ![min_filter](https://user-images.githubusercontent.com/62092317/106533697-ff8c7800-6535-11eb-81a3-d202f0a5bc86.PNG)
      ```python
      
      ```
      * Max filter
      ![max_filter](https://user-images.githubusercontent.com/62092317/106533690-fdc2b480-6535-11eb-944b-4eefceba55c6.PNG)
      ```python
      
      ```
    * Midpoint filter
    ![mid_point_filter](https://user-images.githubusercontent.com/62092317/106533693-fef3e180-6535-11eb-8516-6b490d66e4a2.PNG)
    ```python
    
    ```
    * Alpha-trimmed mean filter
    ![Alpha_trimmed_mean_filter](https://user-images.githubusercontent.com/62092317/106533743-087d4980-6536-11eb-8add-b9f638ae5ac7.PNG)
    ```python
    
    ```
    * Adaptive median fitler
    ![Adaptive_median_filter](https://user-images.githubusercontent.com/62092317/106533738-074c1c80-6536-11eb-9fc9-58109823a3ca.PNG)
    ```python
    
    ```
  * Frequency domain restoration
    * Inverse filtering
    ![Inverse_filter](https://user-images.githubusercontent.com/62092317/106533684-fc918780-6535-11eb-83a5-842641c1fdee.PNG)
    ```python
    
    ```
    * Wiener filtering
    ![Wiener_filter](https://user-images.githubusercontent.com/62092317/106533732-061aef80-6536-11eb-8ad8-fb6a4dee5434.PNG)
    ```python

    ```
