# Emotion-Driven Movie Recommendation System

This repository contains a hybrid movie recommendation system that integrates
Large Language Models (LLMs) such as BERT and RoBERTa for sentiment and emotion
analysis with traditional recommendation techniques to improve personalization
and user engagement.

The system combines content-based filtering and collaborative filtering to
generate diverse and emotionally relevant movie recommendations.

---

## Project Overview

The objective of this project is to enhance movie recommendations by
incorporating emotional signals extracted from user reviews and textual data.

The system uses transformer-based language models to capture sentiment and
emotional context, and integrates these signals into a hybrid recommendation
pipeline.

---

## Key Features

- Emotion-aware and sentiment-aware recommendation framework
- Integration of BERT and RoBERTa for text understanding
- Hybrid recommendation approach:
  - content-based filtering
  - collaborative filtering
- Improved personalization using emotion-driven scoring
- End-to-end pipeline from raw data processing to recommendation generation

---

## System Architecture

The recommendation pipeline consists of:

1. Data loading and integration
2. Data cleaning and preprocessing
3. Feature engineering and exploratory analysis
4. Text representation using TF-IDF
5. Sentiment and emotion extraction using LLM-based models
6. Content similarity computation
7. Collaborative filtering using matrix factorization
8. Hybrid recommendation and ranking

---

## Notebook Description

### Data Loading and Integration.ipynb
Loads and integrates movie metadata and user interaction data.

### Data Integration with Reviews.ipynb
Merges user reviews with movie and interaction datasets for downstream NLP and
recommendation tasks.

### Data Cleaning.ipynb
Handles missing values, inconsistent records, and prepares data for modeling.

### Feature Engineering and Eda.ipynb
Performs feature engineering and exploratory data analysis to understand user
behavior and content characteristics.

### TF-IDF Matrix Computation.ipynb
Generates TF-IDF representations of movie descriptions and review text.

### Cosine Similarity Matrix Computation.ipynb
Computes similarity scores between movies using cosine similarity for
content-based recommendation.

### SVD Matrix And Model.ipynb
Implements collaborative filtering using Singular Value Decomposition (SVD).

### LLM System With Sentiment Analysis.ipynb
Applies BERT and RoBERTa models to extract sentiment information from reviews.

### LLM System With Sentiment&Emotion Analysis.ipynb
Extends sentiment modeling by incorporating emotion classification to improve
personalization.

### Hybrid Recommendation System.ipynb
Combines content-based similarity, collaborative filtering, and emotion-aware
signals to generate final recommendations.

---

## Tech Stack

- Python
- Pandas, NumPy
- Scikit-learn
- Transformers (BERT, RoBERTa)
- TF-IDF
- Matrix factorization (SVD)
- Jupyter Notebook

---

## Use Case

This system supports emotion-aware personalization for movie platforms by:

- capturing user sentiment and emotional preferences from reviews
- improving recommendation relevance beyond standard rating-based methods
- delivering more engaging and context-aware movie suggestions


