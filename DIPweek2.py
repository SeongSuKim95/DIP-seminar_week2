import numpy as np
from numpy import pi, exp, sqrt
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl
from scipy.fftpack import fft2, ifft2

def main():
    img = cv2.imread('image_letter.tif', cv2.IMREAD_GRAYSCALE)
    #img2 = cv2.imread('Skeleton_sharpening.jpg',cv2.IMREAD_GRAYSCALE)
    cv2.imshow('Original',img)
    cv2.waitKey(0)
    cv2.imshow('Laplacian zero scaling',S_Laplacian_filter(img))
    cv2.imwrite('letter.jpg',Fourier_transform(img))
    cv2.waitKey(0)

    #test_filter1 = np.array([[0,0,0,0,0],[0,0,0,0,0],[0,0,1,0,0],[0,0,0,0,0],[0,0,0,0,0]])
    #test_filter2 = np.ones((3,3))*(1/9)
    #test_filter3 = np.ones((5,5))*(1/25)

    #test = S_Gaussian_LPF(img,5,1)

    #test = S_smoothing_linear(img,15)
    #test = S_Hpf(img,3)
    #test = S_Laplacian_filter(img)

    #test = S_Unsharp_masking(img,5,1.5)
    #test1 = S_Sobel_horizontal(img)
    #test2 = S_Sobel_vertical(img)
    #test = Median_filter(img,7)
    #test = Min_filter(img,3)
    #test = Max_filter(img,3)
    #test = Midpoint_filter(img,5)
    #test = Alpha_trimmed_mean_filter(img,5,1)
    #test = Adaptive_median_filter(img)
    #test = np.hstack((test1,test2))
    #test = Spatial_convolution(img,test_filter1)

    #test = Midpoint_filter(img,25)
    #test= F_LPF_square(img,20)
    #test= Butterworth_HPF(img,20,2)
    #test= F_Gaussian_HPF(img,20)
    #test = F_Gaussian_LPF(img,20)
    #test = Fourier_transform(img)
    #test = F_LPF_round(img,20)
    #test = F_HPF_round(img,20)
    #test = F_HPF_square(img,20)
    #test = Notch_round_filter(img,6)
    #test = Bluring_Noise(img,0.0025)
    #test = Inverse_filter(img,0.0025)
    #test = Wiener_filter(img,0.0025,0.0001)
    #test = Wiener_with_Butterworth_filter(img,0.0025,0.0001,85)
    #test = Inverse_with_Butterworth_filter(img,0.0025,85)

    #print(img.shape)
    #x = np.linspace(0,img.shape[1]-1,img.shape[1])
    #y = np.linspace(0,img.shape[0]-1,img.shape[0])


    #xx,yy =np.meshgrid(x,y)
    #fig = plt.figure()
    #ax = plt.axes(projection ='3d')
    #ax.plot_surface(yy,xx,test)
    #plt.show()
##Spatial domain

def Spatial_convolution(img, filter):
    m, n = filter.shape
    if (m == n):
        y, x = img.shape
        #y = y - m + 1
        #x = x - m + 1
        zp = int((m - 1) / 2)
        result = np.zeros((y, x))
        image_zero_padding = np.zeros((y+n-1,x+m-1))
        for i in range(y):
            for j in range(x):
                image_zero_padding[i+zp][j+zp] = img[i][j]

        for i in range(y):
            for j in range(x):
                result[i][j] = np.sum(image_zero_padding[i:i+m,j:j+n] * filter)
                if result[i][j] <0:
                  result[i][j] = 0
                #result_zero_padding[i+zp][j+zp] = result[i][j]

        #min = np.amin(result)
        #result = result -min
        #max = np.amax(result)
        #result = (255 / max) * result
        #result = np.array(result, dtype=np.uint8)

        #result = result + img
        result = np.array(result,dtype=np.uint8)

        #result_zero_padding = img - result_zero_padding
        #b,a =result_zero_padding.shape
        #for i in range(b):
        #   for j in range(a):
        #     if result_zero_padding[i][j] <0:
        #         result_zero_padding[i][j]=0

        #result_zero_padding = np.array(result_zero_padding,dtype=np.uint8)
    return result

def S_smoothing_linear(img,c):

    smoothing_filter = np.ones((c,c))*(1/c**2)
    result = Spatial_convolution(img,smoothing_filter)
    result = np.array(result,dtype=np.uint8)
    return  result

def S_weighted_average(img):

    weighted_filter = np.array([[1,2,1],[2,4,2],[1,2,1]])*(1/16)
    result = Spatial_convolution(img,weighted_filter)
    result = np.array(result,dtype=np.uint8)
    return  result

def S_Gaussian_LPF(img,size,sigma):

    from numpy import pi, exp, sqrt
    # generate a (2*size+1)x(2*size+1) gaussian kernel with mean=0 and sigma
    probs = [exp(-z * z / (2 * sigma * sigma)) / sqrt(2 * pi * sigma * sigma) for z in range(-size, size + 1)]
    Gaussian = np.outer(probs, probs)
    Gaussian = np.array(Gaussian,dtype=np.float)/Gaussian.sum(dtype=np.float)

    result = Spatial_convolution(img,Gaussian)
    result = np.array(result,dtype=np.uint8)
    return result

def S_Laplacian_filter(img):
    laplacian = np.array([[0,1,0],[1,-4,1],[0,1,0]])
    laplacian_diagonal = np.array([[1,1,1],[1,-8,1],[1,1,1]])
    #for x in range(0,result1.shape[1]):
    #    for y in range(0,result1.shape[0]):
    #        if result1[y][x] < 0 :
    #            result1[y][x] == 0

    #result1 = Spatial_convolution(img,laplacian)
    result2 = Spatial_convolution(img,laplacian_diagonal)
    #result = np.hstack((result1,result2))
    return result2

def S_Hpf(img,c):
    identity = np.zeros((c,c))
    center = int((c-1)/2)
    identity[center][center] = 1
    high_pass_filter = identity - np.ones((c,c))*(1/c**2)
    result = Spatial_convolution(img, high_pass_filter)

    result1 = np.array(result,dtype=np.uint8)
    result = logtransformation(result1)
    return result

def S_Unsharp_masking(img,c,a,k):

    a,b = img.shape
    #hpf_image = S_Hpf(img,c)
    lpf_image = S_Gaussian_LPF(img,c,a)
    mask_image = np.empty_like(img)
    mask_image = np.array(mask_image,dtype=float)
    img = np.array(img,dtype=float)
    lpf_image = np.array(lpf_image,dtype=float)
    for i in range(a):
        for j in range(b):
            mask_image[i][j] = img[i][j] - lpf_image[i][j]




    #mask_image = 255 - np.array(mask_image, dtype=np.uint8)


    result = np.empty_like(img)
    result = np.array(result, dtype=float)
    for i in range(a):
        for j in range(b):
            result[i][j] = img[i][j] + k*mask_image[i][j]
    min = np.amin(result)
    result = result - min
    max = np.amax(result)
    result = (255 / max) * result
    result = np.array(result, dtype=np.uint8)
    #result = img + k*mask_image
    #result = img + k * hpf_image
    #result = np.array(result,dtype = np.uint8)
    #result = 255 - np.array(result, dtype=np.uint8)

    return result

def S_Sobel_horizontal(img):
    sobel_h = np.array([[-1,-2,-1],[0,0,0],[1,2,1]])
    result = Spatial_convolution(img,sobel_h)
    min = np.amin(result)
    result = result -min
    max = np.amax(result)
    result = (255 / max) * result
    result = np.array(result, dtype=np.uint8)
    return  result

def S_Sobel_vertical(img):
    sobel_v = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
    result = Spatial_convolution(img,sobel_v)
    min = np.amin(result)
    result = result -min
    max = np.amax(result)
    result = (255 / max) * result
    result = np.array(result, dtype=np.uint8)
    return result

# Restoration
# nonlinear

def Median_filter(img,c):

    zp = int((c-1)/2)
    y, x = img.shape
    image_zero_padding = np.zeros((y + c- 1, x + c - 1))
    for i in range(y):
        for j in range(x):
            image_zero_padding[i+zp][j+zp] = img[i][j]

    image_zero_padding = np.array(image_zero_padding, dtype=np.uint8)

    filter = np.zeros((c, c))
    result = np.zeros((y, x))

    for i in range(y):
        for j in range(x):
            filter = image_zero_padding[i :i+2*zp+1, j:j+2*zp+1]
            result[i][j] = np.median(filter)

    result = np.array(result, dtype=np.uint8)

    return result

def Min_filter(img,c):
    zp = int((c-1)/2)
    y, x = img.shape
    image_zero_padding = np.zeros((y + c- 1, x + c - 1))
    for i in range(y):
        for j in range(x):
            image_zero_padding[i+zp][j+zp] = img[i][j]

    image_zero_padding = np.array(image_zero_padding, dtype=np.uint8)

    filter = np.zeros((c, c))
    result = np.zeros((y, x))

    for i in range(y):
        for j in range(x):
            filter = image_zero_padding[i :i+2*zp+1, j:j+2*zp+1]
            result[i][j] = np.amin(filter)

    result = np.array(result, dtype=np.uint8)

    return result

def Max_filter(img,c):
    zp = int((c-1)/2)
    y, x = img.shape
    image_zero_padding = np.zeros((y + c- 1, x + c - 1))
    for i in range(y):
        for j in range(x):
            image_zero_padding[i+zp][j+zp] = img[i][j]

    image_zero_padding = np.array(image_zero_padding, dtype=np.uint8)

    filter = np.zeros((c, c))
    result = np.zeros((y, x))

    for i in range(y):
        for j in range(x):
            filter = image_zero_padding[i :i+2*zp+1, j:j+2*zp+1]
            result[i][j] = np.amax(filter)

    result = np.array(result, dtype=np.uint8)

    return result

def Midpoint_filter(img,c):
    zp = int((c - 1) / 2)
    y, x = img.shape
    image_zero_padding = np.zeros((y + c - 1, x + c - 1))
    for i in range(y):
        for j in range(x):
            image_zero_padding[i + zp][j + zp] = img[i][j]

    image_zero_padding = np.array(image_zero_padding, dtype=np.uint8)

    filter = np.zeros((c, c))
    result = np.zeros((y, x))

    for i in range(y):
        for j in range(x):
            filter = image_zero_padding[i:i + 2 * zp + 1, j:j + 2 * zp + 1]
            result[i][j] = (1/2)*np.amax(filter)+(1/2)*np.amin(filter)
            a= np.amax(filter)
            b= np.amin(filter)
    result = np.array(result, dtype=np.uint8)

    return result

def Alpha_trimmed_mean_filter(img,c,a):

    zp = int((c - 1) / 2)
    y, x = img.shape
    image_zero_padding = np.zeros((y + c - 1, x + c - 1))
    for i in range(y):
        for j in range(x):
            image_zero_padding[i + zp][j + zp] = img[i][j]

    image_zero_padding = np.array(image_zero_padding, dtype=np.uint8)

    filter = np.zeros((c, c))
    result = np.zeros((y, x))

    for i in range(y):
        for j in range(x):
            filter = image_zero_padding[i:i + 2 * zp + 1, j:j + 2 * zp + 1]
            ordered_filter = np.array(filter).reshape(c**2,)
            ordered_filter = np.sort(ordered_filter)
            ordered_trimmed_filter =  ordered_filter[a:c**2-a]
            result[i][j]=(1/(c**2 - 2*a))*np.sum(ordered_trimmed_filter)

    result = np.array(result, dtype=np.uint8)

    return result

def Adaptive_median_filter(img):
        Smax = 7
        zpm = 3
        y, x = img.shape
        image_zero_padding = np.zeros((y + Smax - 1, x + Smax - 1))
        for i in range(y):
            for j in range(x):
                image_zero_padding[i+zpm][j+zpm] = img[i][j]

        image_zero_padding = np.array(image_zero_padding,dtype =np.uint8)

        filter = np.zeros((3,3))
        filter1 = np.zeros((5,5))
        filter2 = np.zeros((7,7))

        result = np.zeros((y, x))
        for i in range(y):
            for j in range(x):
                filter = image_zero_padding[i+2:i+5,j+2:j+5]
                if (np.median(filter) - np.amin(filter)) > 0 and (np.median(filter) - np.amax(filter)) < 0:
                    if image_zero_padding[i + 3][j + 3] - np.amin(filter) > 0 and image_zero_padding[i + 3][j + 3] - np.amax(filter) < 0:
                        result[i][j] = image_zero_padding[i + 3][j + 3]
                    else:
                        result[i][j] = np.median(filter)
                else:
                    filter1 =image_zero_padding[i+1:i+6,j+1:j+6]
                    if (np.median(filter1)-np.amin(filter1))>0 and (np.median(filter1)-np.amax(filter1)) < 0 :
                        if image_zero_padding[i+3][j+3] - np.amin(filter1) >0 and image_zero_padding[i+3][j+3] -np.amax(filter1) <0:
                            result[i][j] = image_zero_padding[i+3][j+3]
                        else :
                            result[i][j] = np.median(filter1)
                    else :
                        filter2 = image_zero_padding[i:i+7,j:j+7]
                        if (np.median(filter2) - np.amin(filter2)) > 0 and (np.median(filter2) - np.amax(filter2)) < 0:
                            if image_zero_padding[i + 3][j + 3] - np.amin(filter2) > 0 and image_zero_padding[i + 3][j + 3] - np.amax(filter2) < 0:
                                result[i][j] = image_zero_padding[i + 3][j + 3]
                            else:
                                result[i][j] = np.median(filter2)


        result = np.array(result,dtype=np.uint8)

        return result

##Frequency domain

def Fourier_transform(img):
    FT_transformed = np.fft.fft2(img)
    fft_shift = np.fft.fftshift(FT_transformed)
    fft_shift = np.asarray(fft_shift)

    magnitude_spectrum = 20*np.log(np.abs(fft_shift))/np.log(5)
    magnitude_spectrum = np.asarray(magnitude_spectrum,dtype=np.uint8)

    return magnitude_spectrum

def F_LPF_round(img,d):
    
    Lowpassfilter = np.zeros_like(img)
    a,b = Lowpassfilter.shape
    for x in range(0,b):
        for y in range(0,a):
            if int(sqrt((x-b/2)**2 + (y-a/2)**2)) <= d:
                Lowpassfilter[y,x] = 1

    Lowpassfilter = np.array(Lowpassfilter,dtype= float)
    img = Fourier_transform(img)
    result = img * Lowpassfilter
    result1 = np.fft.ifft2(result)
    result2 = np.absolute(result1)

    max = np.amax(result2)
    result3 = (255 / max) * result2
    result4 = np.array(result3, dtype=np.uint8)
    # print(img.shape)
    xx = np.linspace(0,img.shape[1]-1,img.shape[1])
    yy = np.linspace(0,img.shape[0]-1,img.shape[0])

    xxx,yyy =np.meshgrid(xx,yy)
    fig = plt.figure()
    ax = plt.axes(projection ='3d')
    ax.plot_surface(yyy,xxx,Lowpassfilter)
    plt.show()

    return result4

def F_LPF_square(img,c):

    Lowpassfilter = np.zeros_like(img)
    a,b = Lowpassfilter.shape
    if (a%2) == 0:
        if (b%2) == 0:
            Lowpassfilter[int(b/2)-c:int(b/2)+1+c,int(a/2)-c:int(a/2)+1+c] = 1
        elif (b%2) == 1:
            Lowpassfilter[int((b-1)/2)-c:int((b-1)/2)+2+c, int(a/2)-c:int(a/2)+1+c] = 1
    elif (a%2) == 1:
        if (b%2) == 0:
            Lowpassfilter[int(b/2)-c:int(b/2)+1+c,int((a-1)/2)-c:int((a-1)/2)+2+c]=1
        elif (b%2) ==1:
            Lowpassfilter[int((b-1)/2)-c:int((b-1)/2)+2+c,int((a-1)/2)-c:int((a-1)/2)+2+c] = 1

    Lowpassfilter = np.array(Lowpassfilter,dtype=float)
    img = Fourier_transform(img)
    result = img*Lowpassfilter
    result1 = np.fft.ifft2(result)
    result2 = np.absolute(result1)
    max = np.amax(result2)
    result3 = (255/max)*result2
    result4 = np.array(result3,dtype=np.uint8)

    return result4

def F_Gaussian_LPF(img,sigma):

    a,b = img.shape

    Gaussian_height = [exp(-(z-int(a/2))*(z-int(a/2)) / (2 * sigma * sigma))  for z in range(0, a)]
    Gaussian_width = [exp(-(z-int(b/2))*(z-int(b/2)) / (2 * sigma * sigma))  for z in range(0, b)]

    Gaussian_filter = np.outer(Gaussian_height, Gaussian_width)
    Gaussian_filter = np.array(Gaussian_filter,dtype=np.float)/Gaussian_filter.sum(dtype=np.float)
    img = Fourier_transform(img)
    result = img*Gaussian_filter
    result1 = np.fft.ifft2(result)
    result2 = np.absolute(result1)
    max = np.amax(result2)
    result3 = (255/max)*result2
    result4 = np.array(result3,dtype=np.uint8)

    # print(img.shape)
    xx = np.linspace(0,img.shape[1]-1,img.shape[1])
    yy = np.linspace(0,img.shape[0]-1,img.shape[0])

    xxx,yyy =np.meshgrid(xx,yy)
    fig = plt.figure()
    ax = plt.axes(projection ='3d')
    ax.plot_surface(yyy,xxx,Gaussian_filter)

    plt.show()
    return result4

def F_HPF_round(img,d):
    Highpassfilter = np.ones_like(img)
    a,b = Highpassfilter.shape
    for x in range(0,b):
        for y in range(0,a):
            if int(sqrt((x-b/2)**2 + (y-a/2)**2)) <= d:
                Highpassfilter[y,x] = 0

    Highpassfilter = np.array(Highpassfilter,dtype= float)
    img = Fourier_transform(img)
    result = img * Highpassfilter
    result1 = np.fft.ifft2(result)
    result2 = np.absolute(result1)
    max = np.amax(result2)
    result3 = (255 / max) * result2
    result4 = np.array(result3, dtype=np.uint8)

    # print(img.shape)
    xx = np.linspace(0,img.shape[1]-1,img.shape[1])
    yy = np.linspace(0,img.shape[0]-1,img.shape[0])

    xxx,yyy =np.meshgrid(xx,yy)
    fig = plt.figure()
    ax = plt.axes(projection ='3d')
    ax.plot_surface(yyy,xxx,Highpassfilter)
    plt.show()

    return result4

def F_HPF_square(img,c):
    Highpassfilter = np.ones_like(img)
    a, b = Highpassfilter.shape
    if (a % 2) == 0:
        if (b % 2) == 0:
            Highpassfilter[int(b / 2) - c:int(b / 2) + 1 + c, int(a / 2) - c:int(a / 2) + 1 + c] = 0
        elif (b % 2) == 1:
            Highpassfilter[int((b - 1) / 2) - c:int((b - 1) / 2) + 2 + c, int(a / 2) - c:int(a / 2) + 1 + c] = 0
    elif (a % 2) == 1:
        if (b % 2) == 0:
            Highpassfilter[int(b / 2) - c:int(b / 2) + 1 + c, int((a - 1) / 2) - c:int((a - 1) / 2) + 2 + c] = 0
        elif (b % 2) == 1:
            Highpassfilter[int((b - 1) / 2) - c:int((b - 1) / 2) + 2 + c, int((a - 1) / 2) - c:int((a - 1) / 2) + 2 + c] = 0

    Highpassfilter = np.array(Highpassfilter, dtype=float)
    img = Fourier_transform(img)
    result = img * Highpassfilter
    result1 = np.fft.ifft2(result)
    result2 = np.absolute(result1)
    max = np.amax(result2)
    result3 = (255 / max) * result2
    result4 = np.array(result3, dtype=np.uint8)
    return result4

def F_Gaussian_HPF(img,sigma):
    from numpy import pi, exp, sqrt
    a, b = img.shape

    Gaussian_height = [(exp(-(z-int(a/2))*(z-int(a/2))/(2*sigma*sigma))) for z in range(0, a)]
    Gaussian_width = [(exp(-(z-int(b/2))*(z-int(b/2))/(2*sigma*sigma))) for z in range(0, b)]

    Gaussian_filter = np.outer(Gaussian_height, Gaussian_width)
    Gaussian_filter = np.array(Gaussian_filter, dtype=np.float)
    Gaussian_filter = np.array(1-Gaussian_filter,dtype=np.float)
    img = Fourier_transform(img)
    result = img * (Gaussian_filter)
    result1 = np.fft.ifft2(result)
    result2 = np.absolute(result1)
    max = np.amax(result2)
    result3 = (255 / max) * result2
    result4 = np.array(result3, dtype=np.uint8)
    # print(img.shape)

    xx = np.linspace(0,img.shape[1]-1,img.shape[1])
    yy = np.linspace(0,img.shape[0]-1,img.shape[0])

    xxx,yyy =np.meshgrid(xx,yy)
    fig = plt.figure()
    ax = plt.axes(projection ='3d')
    ax.plot_surface(yyy,xxx,Gaussian_filter)
    plt.show()

    return result4

def Butterworth_LPF(img,d,order):

    Lowpassfilter = np.zeros_like(img)
    Lowpassfilter = np.array(Lowpassfilter,dtype=float)
    a,b = Lowpassfilter.shape
    for x in range(0,b):
        for y in range(0,a):
                distance = sqrt((x-b/2)**2 + (y-a/2)**2)
                Lowpassfilter[y,x] = 1/(1+(distance/d)**(2*order))

    img = Fourier_transform(img)
    result = img * Lowpassfilter
    result1 = np.fft.ifft2(result)
    result2 = np.absolute(result1)
    max = np.amax(result2)
    result3 = (255 / max) * result2
    result4 = np.array(result3, dtype=np.uint8)

    xx = np.linspace(0,img.shape[1]-1,img.shape[1])
    yy = np.linspace(0,img.shape[0]-1,img.shape[0])

    xxx,yyy =np.meshgrid(xx,yy)
    fig = plt.figure()
    ax = plt.axes(projection ='3d')
    ax.plot_surface(yyy,xxx,Lowpassfilter)
    plt.show()
    return result4

def Butterworth_HPF(img,d,order):
    Highpassfilter = np.zeros_like(img)
    Highpassfilter = np.array(Highpassfilter,dtype=float)
    a,b = Highpassfilter.shape
    for x in range(0,b):
        for y in range(0,a):
            distance = sqrt((x-b/2)**2+(y-a/2)**2)
            if distance==0 :
                Highpassfilter[y,x] = 0
            else :
                Highpassfilter[y,x] = 1/(1+(d/distance)**(2*order))

    img = Fourier_transform(img)
    result = img * Highpassfilter
    result1 = np.fft.ifft2(result)
    result2 = np.absolute(result1)
    max = np.amax(result2)
    result3 = (255 / max) * result2
    result4 = np.array(result3, dtype=np.uint8)
    # print(img.shape)
    xx = np.linspace(0,img.shape[1]-1,img.shape[1])
    yy = np.linspace(0,img.shape[0]-1,img.shape[0])

    xxx,yyy =np.meshgrid(xx,yy)
    fig = plt.figure()
    ax = plt.axes(projection ='3d')
    ax.plot_surface(yyy,xxx,Highpassfilter)
    plt.show()

    return result4

def Notch_round_filter(img,d):
    a, b = img.shape
    Notchfilter = np.ones_like(img)

    for x in range(0,b):
        for y in range(0,a):
            if (sqrt((x-111)**2 + (y-81)**2)) <= d or (sqrt((x-55)**2 + (y-85)**2)) <= d or (sqrt((x-57)**2 + (y-165)**2)) <= d or (sqrt((x-113)**2 + (y-161)**2)) <= d or (sqrt((x-55)**2 + (y-44)**2)) <= d or (sqrt((x-111)**2 + (y-40)**2)) <= d or (sqrt((x-57)**2 + (y-206)**2)) <= d or(sqrt((x-113)**2 + (y-202)**2)) <= d :
             Notchfilter[y,x] = 0

    Notchfilter = np.array(Notchfilter,dtype= float)
    img = Fourier_transform(img)
    result = img * Notchfilter
    result1 = np.fft.ifft2(result)
    result2 = np.absolute(result1)
    max = np.amax(result2)
    result3 = (255 / max) * result2
    result4 = np.array(result3, dtype=np.uint8)

    xx = np.linspace(0,img.shape[1]-1,img.shape[1])
    yy = np.linspace(0,img.shape[0]-1,img.shape[0])

    xxx,yyy =np.meshgrid(xx,yy)
    fig = plt.figure()
    ax = plt.axes(projection ='3d')
    ax.plot_surface(yyy,xxx,Notchfilter)
    plt.show()

    return result4
#Restoration

def Bluring_Noise(img,k):

    Noisefilter = np.zeros_like(img)
    Noisefilter = np.array(Noisefilter, dtype=float)
    a, b = Noisefilter.shape

    for x in range(0, b):
        for y in range(0, a):
                Noisefilter[y, x] = exp((-1)*k*((y-b/2)**2 +(x-a/2)**2)**(5/6))

    img = Fourier_transform(img)
    result = img * Noisefilter
    result1 = np.fft.ifft2(result)
    result2 = np.absolute(result1)
    #max = np.amax(result2)
    #result3 = (255 / max) * result2
    #result4 = np.array(result3, dtype=np.uint8)
    result2 = np.array(result2,dtype=np.uint8)

    return result2

def Inverse_filter(img,k):
    Inverse_filter = np.zeros_like(img)
    Inverse_filter = np.array(Inverse_filter, dtype=float)
    a, b = Inverse_filter.shape

    for x in range(0, b):
        for y in range(0, a):
            Inverse_filter[y,x] =1 /(exp((-1)*k*((y-b/2)**2+(x-a/2)**2)**(5/6)))

    img = Fourier_transform(img)
    result = img * Inverse_filter
    result1 = np.fft.ifft2(result)
    result2 = np.absolute(result1)

    max = np.amax(result2)
    result3 = (255 / max ) * result2
    result4 = np.array(result3, dtype=np.uint8)

    return result4

def Wiener_filter(img,k,K):
    Wiener_filter = np.zeros_like(img)
    Noise_filter = np.zeros_like(img)
    Noise_filter = np.array(Noise_filter, dtype=float)

    a, b = Noise_filter.shape

    for x in range(0, b):
        for y in range(0, a):
            Noise_filter[y, x] = (exp((-1)*k* ((y - b / 2) ** 2 + (x - a / 2) ** 2) ** (5 / 6)))
    Noise_abs = np.absolute(Noise_filter)
    Noise_abs_square  =np.square(Noise_abs)

    Wiener_filter = (1/Noise_filter)*(Noise_abs_square/(Noise_abs_square+K))
    img = Fourier_transform(img)
    result = img * Wiener_filter
    result1 = np.fft.ifft2(result)
    result2 = np.absolute(result1)

    max = np.amax(result2)
    result3 = (255 / max ) * result2
    result4 = np.array(result3, dtype=np.uint8)

    return result4

def Inverse_with_Butterworth_filter(img,k,r):
    Inverse_filter = np.zeros_like(img)
    Inverse_filter = np.array(Inverse_filter, dtype=float)
    a, b = Inverse_filter.shape

    for x in range(0, b):
        for y in range(0, a):
            Inverse_filter[y, x] =1 /(exp((-1)*k* ((y - b / 2) ** 2 + (x - a / 2) ** 2) ** (5 / 6)))

    Lowpassfilter = np.zeros_like(img)
    Lowpassfilter = np.array(Lowpassfilter,dtype=float)
    c,d = Lowpassfilter.shape

    for x in range(0,d):
        for y in range(0,c):
                distance = sqrt((x-d/2)**2 + (y-c/2)**2)
                Lowpassfilter[y,x] = 1/(1+(distance/r)**(2*10))

    img = Fourier_transform(img)
    result = img * Inverse_filter * Lowpassfilter
    result1 = np.fft.ifft2(result)
    result2 = np.absolute(result1)

    max = np.amax(result2)
    result3 = (255 / max ) * result2
    result4 = np.array(result3, dtype=np.uint8)

    return result4

def Wiener_with_Butterworth_filter(img,k,K,r):
    Wiener_filter = np.zeros_like(img)
    Noise_filter = np.zeros_like(img)
    Noise_filter = np.array(Noise_filter, dtype=float)

    a, b = Noise_filter.shape

    for x in range(0, b):
        for y in range(0, a):
            Noise_filter[y, x] = (exp((-1) * k * ((y - b / 2) ** 2 + (x - a / 2) ** 2) ** (5 / 6)))
    Noise_abs = np.absolute(Noise_filter)
    Noise_abs_square = np.square(Noise_abs)

    Wiener_filter = (1 / Noise_filter) * (Noise_abs_square / (Noise_abs_square + K))

    Lowpassfilter = np.zeros_like(img)
    Lowpassfilter = np.array(Lowpassfilter,dtype=float)
    c,d = Lowpassfilter.shape

    for x in range(0,d):
        for y in range(0,c):
                distance = sqrt((x-d/2)**2 + (y-c/2)**2)
                Lowpassfilter[y,x] = 1/(1+(distance/r)**(2*10))

    img = Fourier_transform(img)
    result = img * Wiener_filter * Lowpassfilter
    result1 = np.fft.ifft2(result)
    result2 = np.absolute(result1)

    max = np.amax(result2)
    result3 = (255 / max ) * result2
    result4 = np.array(result3, dtype=np.uint8)

    return result4

### log transformation

def logtransformation(img):
    c=255/np.log(256)/np.log(5)
    #c=(1/2)*255/np.log(1+np.max(img))/np.log(np.sqrt(10))
    log_transform = c*np.log(1+img)/np.log(5)
    result = np.array(log_transform, dtype=np.uint8)
    return result

def Thresholding(img,p):
    img = np.array(img,dtype = np.uint8)
    max_intensity = np.amax(img)
    for x in range(0,img.shape[1]):
        for y in range(0,img.shape[0]):
            if img[y,x] < int(max_intensity * p):
                img[y,x] = 0

    return img

main()

