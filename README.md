# MyAnimeList-Hybrid-Recommender-System
End to End implementation of Hybrid Anime Recommender System using Flask 

## 1 Business Problem


### 1.1 Problem Description

MyAnimeList is the largest and most active anime and manga community on the internet, it’s like the IMDB of Anime and Manga where you can rate any Anime or Manga from 1 to 10. As MyAnimeList does not have any recommendation system on their website, so I decided to work on creating one. It will help the users to find new anime that suites their taste and interest, and it could also help MyAnimeList website to grow even bigger.

There are numerous types of Recommendation System but the most commonly used in big companies is Hybrid Recommender System, which is the most robust than other types of Recommendation System. So we will be building a Hybrid Anime Recommender System which will be recommending different and better sets of anime to each user based on users rating and past interaction with different anime.

### 1.2 Problem Statement

MyAnimeList dataset on kaggle provides lot of information about user, anime and users rating. We will build the Hybrid Recommendation System with least RMSE value to recommend various anime to users.

### 1.3 Business Objective

Recommend the anime to user based on users watching history that the user has not watched yet.
Minimize the RMSE value and maximize the Precision@10 value of Content Based Filtering and Collaborative Filtering Model that will be combined to build Hybrid Recommendation System Model.

### 1.4 Business Constraints

The model should have some form of Interpretability when predicting recommendations to users because it helps us to understand model behaviour.

The model should have Strict Latency Constraint as it will be an interactive model.

## 2 Machine Learning Problem

### 2.1 Data Overview

Dataset : https://www.kaggle.com/azathoth42/myanimelist

The MyAnimeList dataset on kaggle has three different files named :

anime_cleaned.csv
animelists_cleaned.csv
users_cleaned.csv
These three files contain all the necessary information about animes, users and users rating.

There are total 108k Users, 6k Anime and about 32M total rating by the users.

And the user can rate an anime between range of 1-10.

### 2.2 Real World Machine Learning Problem

In this case study, we are going to use a Hybrid Recommendation System to build a personalized hybrid anime recommender system with the best machine learning algorithms to help recommend new anime to users. Hybrid Recommendation System which is a combination of Content Based Filtering Model and Collaborative Filtering Model helps the model to overcome Cold Start problem and gives a diverse range of recommendations to users. In Hybrid Recommendation System we will be using different KNN algorithms, Vectorization, Matrix Factorization techniques.

### 2.3 Performance Metric

Mean Absolute Percentage Error (MAPE)

Root Mean Square Error (RMSE)

Precision@10 : We will be using Precision@10 metric to evaluate our model because it gives precision on the top 10 items that we are recommending to users.

### 2.4 Machine Learning Objective and Constraints

The model should have some form of Interpretability when predicting recommendations to users because it helps us to understand model behaviour.

Minimize the RMSE and Precision@10 value of each Content Based Filtering and Collaborative Filtering Model that will be combined to build Hybrid Recommendation System Model.
