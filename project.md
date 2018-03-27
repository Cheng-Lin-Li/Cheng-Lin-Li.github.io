---
layout: page
title: Projects and Competition notebooks
permalink: /project/
---

How to start a project in machine learning? How can I leverage all those techniques into real world challenge? There are projects and Jupyter notebook collections for you as a starting points.

  1. Stock Price Forecasting by Stock Selections: Python/Tensorflow
    
      This is a project which implemented Neural Network and Long Short Term Memory (LSTM) for stock price predictions. These models beat DJIA performance based on 1 quarter of weekly price, return rate of the DJIA components plus assistant indices to predict the highest increasing rate stock for the next quarter.

      Github address: [https://github.com/Cheng-Lin-Li/MachineLearning/tree/master/TensorFlow](https://github.com/Cheng-Lin-Li/MachineLearning/tree/master/TensorFlow)

  2. Market Trend Prediction with Social Media Listening: Python/Keras/Facebook Graph API/Twitter API/ACHE Crawler/Beautifulsoup
      
      This project leveraged 1.5 years of 30 historical stock prices, Dow Jones Industrial Average(DJIA) index, with semantic information from social media (Facebook and Twitter) on T day to provide better one to many DJIA trend classifications for T+1/T+30 days than the model without social media info by LSTM in python a Keras.

      Github address: [https://github.com/Cheng-Lin-Li/Market-Trend-Prediction](https://github.com/Cheng-Lin-Li/Market-Trend-Prediction)
  3. Information Visualization Project - Business Cycle Introduction
      
      Build up a web application to introduce what business cycle is and how it will impact to us.

      Github address: [https://github.com/Cheng-Lin-Li/InformationVisualization/tree/master/BusinessCycle](https://github.com/Cheng-Lin-Li/InformationVisualization/tree/master/BusinessCycle)
  4. Number of vehicles Prediction: scikit-learn/Keras
      This task is to perform prediction for number of vehicles by given data. This is a demo program to leverage four models (SVR, NN, LSTM, GRU) from existing libraries in one challenge. The final result can be improved by some emsemble techniques like Bootstrap aggregating (bagging), boosting, and stacking to get better performance.
      Github address: [https://github.com/Cheng-Lin-Li/MachineLearning/tree/master/Competition/McKinseyAnalyticsPrediction](https://github.com/Cheng-Lin-Li/MachineLearning/tree/master/Competition/McKinseyAnalyticsPrediction)
  5. Recommendation System: scikit-learn/Surprise (under constructing)
      This task leverages Content Based Filtering and Singular Value decomposition (SVD) to perform recommendation system build up.

      Github address: [https://github.com/Cheng-Lin-Li/MachineLearning/blob/master/Competition/McKinseyAnalyticsRecommendation/Recommendation.ipynb](https://github.com/Cheng-Lin-Li/MachineLearning/blob/master/Competition/McKinseyAnalyticsRecommendation/Recommendation.ipynb)


  6. Objects detection and segmentation: Keras/Tensorflow/OpenCV(under constructing)
      This task is based on Mask RCNN (extends Faster R-CNN by adding a branch for predicting an object mask in parallel with the existing branch for bounding box recognition) to perform transfer learning on Nuclear detection from varience image files.

        Reference Paper: [https://arxiv.org/abs/1703.06870](https://arxiv.org/abs/1703.06870)

        The original model clone from: [https://github.com/matterport/Mask_RCNN](https://github.com/matterport/Mask_RCNN)

        Github address:[https://github.com/Cheng-Lin-Li/MachineLearning/tree/master/Competition/ObjectDetectionSegmentation](https://github.com/Cheng-Lin-Li/MachineLearning/tree/master/Competition/ObjectDetectionSegmentation)

        My trial work to integrate threading webcam stream and the pre-trained for object detection and segmentation tasks.
	Github address:[https://github.com/Cheng-Lin-Li/MachineLearning/blob/master/Competition/ObjectDetectionSegmentation/Video-Demo-Mask_RCNN.ipynb](https://github.com/Cheng-Lin-Li/MachineLearning/blob/master/Competition/ObjectDetectionSegmentation/Video-Demo-Mask_RCNN.ipynb)
