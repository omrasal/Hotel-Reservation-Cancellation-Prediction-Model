# Hotel Reservation Cancellation Prediction

## Project Overview
This project aims to predict hotel reservation cancellations using machine learning techniques. It was developed as part of a data analytics and machine learning class, demonstrating skills in data analysis, feature engineering, and predictive modeling.

## Business Understanding
The growth of online hotel booking channels has led to an increase in cancellations and no-shows, significantly impacting hotel revenue. This project provides an end-to-end machine learning solution to help hotels:
- Maintain high occupancy rates
- Improve revenue management strategies
- Provide a satisfactory customer experience

## Data Understanding
The dataset contains various attributes of customers' reservation details, including:
- Booking information (ID, number of adults/children, nights booked)
- Meal plans
- Car parking requirements
- Room type
- Lead time
- Arrival details
- Market segment
- Guest history
- Pricing
- Special requests
- Booking status (target variable)

## Methodology
1. **Exploratory Data Analysis (EDA)**: Analyzed feature distributions, correlations, and patterns in the data.
2. **Data Preparation**: Handled duplicates, encoded categorical variables, and split the data into training and testing sets.
3. **Model Development**: Implemented and compared multiple models:
   - Logistic Regression
   - Decision Tree
   - XGBoost
4. **Model Evaluation**: Used ROC-AUC score as the primary metric for model performance.

## Key Findings
- XGBoost performed the best with an ROC-AUC score of 0.91.
- Most important factors for predicting cancellations:
  1. Market segment type 1
  2. Number of special requests (negative correlation)
  3. Car parking space requirement (negative correlation)
  4. Lead time (positive correlation)

## Technologies Used
- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- XGBoost

## Future Work
- Feature engineering to create more predictive variables
- Hyperparameter tuning for the XGBoost model
- Deployment of the model as a web service for real-time predictions

## Contact
[https://www.linkedin.com/in/omrasal/](url)
