# H1N1 Vaccine Prediction

## Index

1. [Overview](#overview)
2. [Features](#features)
3. [Data](#data)
4. [Usage](#usage)
5. [Dependencies](#dependencies)
6. [Installation](#installation)
7. [Acknowledgments](#acknowledgments)

## Overview

The H1N1 Vaccine Prediction project aims to predict the likelihood of individuals getting vaccinated against the H1N1 flu based on various demographic and behavioral factors. It utilizes machine learning techniques such as logistic regression, decision trees, ensemble methods, and support vector machines to build predictive models.

## Features

The dataset contains several features that are used to predict the likelihood of H1N1 vaccine uptake. Some of the features include:

Demographic information such as age bracket, education level, sex, income level, marital status, and employment status.
Behavioral attributes including concerns about H1N1 flu, awareness, preventive measures taken, doctor recommendations for vaccination, and perception of vaccine effectiveness.
Other factors like having chronic medical conditions, having children under 6 months old, being a healthcare worker, and health insurance status.

## Data

The dataset used for this project is stored in a CSV file named 'h1n1_vaccine_prediction.csv'. It contains information on individuals' responses to surveys regarding their demographics, behaviors, and attitudes towards the H1N1 flu and vaccination. The dataset is preprocessed to handle missing values and encode categorical variables.

## Usage

**Data Preprocessing:** The code begins by loading the dataset and performing data preprocessing steps such as handling missing values and encoding categorical variables.

**Model Training:** Various machine learning models are trained on the preprocessed data, including logistic regression, decision trees, ensemble methods (bagging, boosting, and random forest), and support vector machines.

**Model Evaluation:** The trained models are evaluated using performance metrics such as accuracy score, confusion matrix, and classification report.

**Model Selection:** Based on the evaluation results, the best-performing model can be selected for making predictions on new data.

## Dependencies

**NumPy**

**Pandas**

**Matplotlib**

**Seaborn**

**Scikit-learn**

## Installation

**Clone the repository:**

```
git clone https://github.com/yourusername/H1N1_Vaccine_Prediction.git
```

**Install dependencies:**

```
pip install -r requirements.txt
```

## Acknowledgments

This project is based on publicly available data from surveys conducted during the H1N1 flu pandemic.
Thanks to the open-source community for providing valuable libraries and resources for machine learning and data analysis.
