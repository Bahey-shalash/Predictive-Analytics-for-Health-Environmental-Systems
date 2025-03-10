# Predictive Analytics Projects

This repository contains two predictive modeling projects along with a comprehensive final report detailing the methodology, analysis, and results.

---

## Overview

- **Project 1: Controlling Disease Propagation**  
  Develops and compares machine learning models to predict whether an individual is infected (y=1) using five input features (x1 to x5). The focus is on minimizing false negatives to prevent outbreak escalation, while reducing data collection costs by selecting the most relevant features.

- **Project 2: Environmental Phenomenon Analysis**  
  Uses sensor data recorded every 15 minutes (from January 1, 2024 to February 29, 2024) to reconstruct missing values in an environmental system. This project employs temporal feature engineering—such as cyclic transformations and lag features—to capture complex patterns and predict the target variable accurately.

- **Final Report**  
  A detailed final report (`rapport.ipynb`) documents the methodology, analysis, and performance of the models used in both projects.

---

## File Structure

- `final_projet_1.ipynb`  
  Notebook for Project 1 (Disease Propagation Modeling).

- `final_projet_2.ipynb`  
  Notebook for Project 2 (Environmental Sensor Data Analysis).

- `rapport.ipynb`  
  Final report detailing the project methodologies, analyses, and results.

---

## Requirements

To run the notebooks, you need Python 3 and the following libraries:

- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn

You can install these packages with:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
