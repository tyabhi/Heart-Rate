#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
                ***********  CONTACTLESS HEART RATE DETECTION  ***********
Created on 18-01-2021

@author: Abhishek Tyagi
         MTech CDS I year
         Sr No - 17907
"""
import datetime
import time
import traceback
import cv2
import numpy
import scipy.fftpack as fftpack
from scipy import signal
from matplotlib import pyplot
# Face detection 
faceCascade = cv2.CascadeClassifier("/haarcascades/haarcascade_frontalface_alt0.xml")

# Converting the input video into set of frames
def read_video(path):
    cap = cv2.VideoCapture(path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    video_frames = []
    face_rects = ()
    while cap.isOpened():
        ret,img = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
        roi_frame = img
        # Face detection
        if len(video_frames) == 0:
            face_rects = faceCascade.detectMultiScale(gray, 1.3,5)
        # Select Region of interest (ROI)
        if len(face_rects) > 0:
            for(x,y,w,h) in face_rects:
                roi_frame = img[y:y+h,x:x+w]
            if roi_frame.size != img.size:
                roi_frame = cv2.resize(roi_frame,(500,500))
                frame = numpy.ndarray(shape = roi_frame.shape,dtype="float")
                frame[:] = roi_frame * (1. /255)
                video_frames.append(frame)
    frame_ct = len(video_frames)
    cap.release()
    return video_frames, frame_ct, fps


def build_gaussian_pyramid(img,levels):
    float_img = numpy.ndarray(img.shape,dtype = "float")
    float_img[:] = img
    pyramid = [float_img]
    for i in range(levels-1):
        float_img = cv2.pyrDown(float_img)
        pyramid.append(float_img)
    return pyramid
    
    
def build_laplacian_pyramid(img,levels):
    gaussian_pyramid = build_gaussian_pyramid(img, levels)
    laplacian_pyramid = []
    for i in range(levels-1):
        upsampled = cv2.pyrUp(gaussian_pyramid[i+1])
        (height,width,depth) = upsampled.shape
        gaussian_pyramid[i] = cv2.resize(gaussian_pyramid[i],(height,width))
        diff = cv2.subtract(gaussian_pyramid[i], upsampled)
        laplacian_pyramid.append(diff)
    laplacian_pyramid.append(gaussian_pyramid[-1])
    return laplacian_pyramid


def build_video_pyramid(frames):
    lap_video = []
    for i ,frame in enumerate(frames):
        pyramid = build_laplacian_pyramid(frame,3)
        
        for j in range(3):
            if i == 0:
                lap_video.append(numpy.zeros((len(frames),pyramid[j].shape[0],pyramid[j].shape[1],3)))
            lap_video[j][i] = pyramid[j]
            #fig = pyplot.figure()
            #pyplot.imshow(lap_video[j][i])       
    #pyplot.show()
    return lap_video


# Bandpass filter with Fast-fourier transform
def fft_filter(video,freq_min,freq_max,fps):
    fft = fftpack.fft(video,axis=0)
    frequencies = fftpack.fftfreq(video.shape[0], d=1.0/fps)
    bound_low = (numpy.abs(frequencies-freq_min)).argmin()
    bound_high = (numpy.abs(frequencies - freq_max)).argmin()
    # All unwanted frequencies amp = 0
    fft[:bound_low] = 0
    fft[bound_high:-bound_high] = 0
    fft[-bound_low:] = 0
    ifft = fftpack.ifft(fft,axis=0)
    result=numpy.abs(ifft)
    #Amplication factor = 100
    result *= 100
    return result,fft,frequencies


def find_heart_rate(fft,freqs,freq_min,freq_max):
    fft_maximus = []
    for i in range(fft.shape[0]):
        if freq_min <= freqs[i] <= freq_max:
            fftMap = abs(fft[i])
            fft_maximus.append(fftMap.max())
        else:
            fft_maximus.append(0)
    peaks,properties = signal.find_peaks(fft_maximus)
    max_peak = -1
    max_freq = 0
    
    #Find frequency with max amplitude in peaks 
    for peak in peaks:
        if fft_maximus[peak] > max_freq:
            max_freq = fft_maximus[peak]
            max_peak = peak
            
    return freqs[max_peak]*60 # Multiplied by 60 to convert Hz in bmp


def main():
    # 1 Hz = 60 Beats per minute (bpm)
    # 1.8 Hz = 108 Beats per minute (bpm)
    freq_min = 1
    freq_max = 1.8
    
    #Preprocessing phase
    print("Reading and preprocessing video ...")
    video_frames, frame_ct, fps = read_video("/videos/test.mov")
    
    #Pyramiding phase
    print("Building laplacian video pyramid")
    lap_video = build_video_pyramid(video_frames)
    for i, video in enumerate(lap_video):
        if i == 0 or i == len(lap_video)-1:
            continue
        #Eulerian magnification with temporal FFT filtering
        print("Running FFT and Eulerian magnification ...")
        result,fft,frequencies = fft_filter(video,freq_min,freq_max,fps)
        lap_video[i] += result
        
        #Calculate heart rate 
        print("Calculating heart rate ...")
        heart_rate = find_heart_rate(fft,frequencies,freq_min,freq_max)

    #Output heart rate and fial video
    print("Heart rate: ", heart_rate, "bpm") 
    return


if __name__ == '__main__':
    print('Program started at '+datetime.datetime.now().strftime('%d/%m/%Y %I:%M:%S %p'))
    start_time = time.time()
    try:
        main()
    except Exception as e:
        print(e)
        traceback.print_exc()
    end_time = time.time()
    print('Program ended at '+datetime.datetime.now().strftime('%d/%m/%Y %I:%M:%S %p'))
    print('Execution time: ' +str(datetime.timedelta(seconds=end_time-start_time)))