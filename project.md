---
layout: page
title: Project
permalink: /project/
---

How to start a project in machine learning? How can I leverage all those techniques into real world challenge? There are my projects and competitions Jupyter notebooks for you as a starting points.


  0. Enlighten Segmentation (new)
      
      This is a project which build up a pipeline line for everyone to train his own Capsule Nets or SegCaps from scratch by Microsoft Common Objects in COntext (MS COCO) 2D image dataset.

      The original paper for SegCaps can be found at [https://arxiv.org/abs/1804.04241](https://arxiv.org/abs/1804.04241). 
      
      The official source code can be found at [https://github.com/lalonderodney/SegCaps](https://github.com/lalonderodney/SegCaps) 
      
      Author's project page for this work can be found at [https://rodneylalonde.wixsite.com/personal/research-blog/capsules-for-object-segmentation](https://rodneylalonde.wixsite.com/personal/research-blog/capsules-for-object-segmentation).

      The project delivery includes:
      1. Microsoft COCO dataset crawler program to automatic generate training data set for any class.
        - You can choose any class of images from MS COCO dataset, the specific class of mask will also generate for segmentation task.
        - You can base on image IDs to download image files and specify the annotation class you want for mask data.
      2. Improve programs not only take computed tomography (CT) scan images, but also support 2D color images training and testing.
      3. A program captures image from video stream via a webcam for segmentation task.

      Project address: [https://github.com/Cheng-Lin-Li/SegCaps](https://github.com/Cheng-Lin-Li/SegCaps)

    
  
  
  1. Stock Price Forecasting by Stock Selections: Python/Tensorflow
    
      This is a project which implemented Neural Network and Long Short Term Memory (LSTM) for stock price predictions. These models beat DJIA performance based on 1 quarter of weekly price, return rate of the DJIA components plus assistant indices to predict the highest increasing rate stock for the next quarter.

      Project Report: [Multi-Layer Perceptron (MLP), and Long-Short Term Memory (LSTMs) for stock price forecasting](https://github.com/Cheng-Lin-Li/MachineLearning/blob/master/TensorFlow/ProjectReport.pdf)

      Github address: [https://github.com/Cheng-Lin-Li/MachineLearning/tree/master/TensorFlow](https://github.com/Cheng-Lin-Li/MachineLearning/tree/master/TensorFlow)

  2. Market Trend Prediction with Social Media Listening: Python/Keras/Facebook Graph API/Twitter API/ACHE Crawler/Beautifulsoup
      
      This project leveraged 1.5 years of 30 historical stock prices, Dow Jones Industrial Average(DJIA) index, with semantic information from social media (Facebook and Twitter) on T day to provide better one to many DJIA trend classifications for T+1/T+30 days than the model without social media info by LSTM in python a Keras.

      Project Jupyter Notebook: 
      
      a. [Dow Jones Industrial Average (DJIA) Prediction with Social Media Information](https://github.com/Cheng-Lin-Li/Market-Trend-Prediction/blob/master/source/Dow%20Jones%20Industrial%20Average%20Prediction%20with%20Media%20Channel%20Info-with%20Social%20Info.ipynb)

      b. [Dow Jones Industrial Average (DJIA) Prediction
 with NO Social Media information](https://github.com/Cheng-Lin-Li/Market-Trend-Prediction/blob/master/source/Dow%20Jones%20Industrial%20Average%20Prediction%20without%20Social%20media%20data.ipynb)

      Github address: [https://github.com/Cheng-Lin-Li/Market-Trend-Prediction](https://github.com/Cheng-Lin-Li/Market-Trend-Prediction)
  3. Information Visualization Project - Business Cycle Introduction
      
      Build up a web application to introduce what business cycle is and how it will impact to us.

      Project live demo site:[Business Cycle Introduction](https://cheng-lin-li.github.io/assets/InformationVisualization/BusinessCycle/dist/index.html)

      Github address: [https://github.com/Cheng-Lin-Li/InformationVisualization/tree/master/BusinessCycle](https://github.com/Cheng-Lin-Li/InformationVisualization/tree/master/BusinessCycle)

  4. Number of vehicles Prediction: scikit-learn/Keras
      This task is to perform prediction for number of vehicles by given data. This is a demo program to leverage four models (SVR, NN, LSTM, GRU) from existing libraries in one challenge. The final result can be improved by some emsemble techniques like Bootstrap aggregating (bagging), boosting, and stacking to get better performance.
      
      Jupyter Notebook: [Prediction Number of Vehicles](https://github.com/Cheng-Lin-Li/MachineLearning/blob/master/Competition/McKinseyAnalyticsPrediction/NumberOfVehiclesPrediction.ipynb)

      Github address: [https://github.com/Cheng-Lin-Li/MachineLearning/tree/master/Competition/McKinseyAnalyticsPrediction](https://github.com/Cheng-Lin-Li/MachineLearning/tree/master/Competition/McKinseyAnalyticsPrediction)

  5. Recommendation System: scikit-learn/Surprise (under constructing)
      This task leverages Content Based Filtering and Singular Value decomposition (SVD) to perform recommendation system build up.

      Jupyter Notebook: [https://github.com/Cheng-Lin-Li/MachineLearning/blob/master/Competition/McKinseyAnalyticsRecommendation/Recommendation.ipynb](https://github.com/Cheng-Lin-Li/MachineLearning/blob/master/Competition/McKinseyAnalyticsRecommendation/Recommendation.ipynb)


  6. Objects detection and segmentation: Keras/Tensorflow/OpenCV(under constructing)
      This task is based on Mask RCNN (extends Faster R-CNN by adding a branch for predicting an object mask in parallel with the existing branch for bounding box recognition) to perform transfer learning on Nuclear detection from varience image files.

      My trial works to integrate threading webcam stream and the pre-trained for object detection and segmentation tasks.

	  My demo Jupyter Notebook:[https://github.com/Cheng-Lin-Li/MachineLearning/blob/master/Competition/ObjectDetectionSegmentation/Video-Demo-Mask_RCNN.ipynb](https://github.com/Cheng-Lin-Li/MachineLearning/blob/master/Competition/ObjectDetectionSegmentation/Video-Demo-Mask_RCNN.ipynb)

        Github address:[https://github.com/Cheng-Lin-Li/MachineLearning/tree/master/Competition/ObjectDetectionSegmentation](https://github.com/Cheng-Lin-Li/MachineLearning/tree/master/Competition/ObjectDetectionSegmentation)

        Reference Paper: [https://arxiv.org/abs/1703.06870](https://arxiv.org/abs/1703.06870)

        The original model clone from: [https://github.com/matterport/Mask_RCNN](https://github.com/matterport/Mask_RCNN)




