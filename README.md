# Churn Prediction for Mobile and Online Casual Games Using Play Log Data
---

## Motivation
---
This research aims to predict player churn in mobile and online casual games using player log data. Player churn is important as it
covers how users either continue to play a specific game or become disinterested, which suppliers focus on reducing the churn rate.
By leveraging the use of machine and deep learning algorithms, a standard churn analysis process is developed that further targets these 
free-to-play games, improving upon traditional churn models used in subscriptions.
With the rise of casual games, churn analysis begins to climb in importance for these gaming companies to further improve on both their business model
and catering to users, showing a need for a sophisticated and accurate determination.
Using play log data allows for early identification of at-risk players, crucial for ad/microtransaction-based monetization.

## Novelty
---
- Traditional churn models regarding subscriptions are not effective for free-to-play games
- Focus on new players instead of existing players to ensure up-to-date player base
- Definition of churn using observation period (OP) and chum prediction period (CP)
- Usage of traditional machine learning algorithms instead of more computationally exhaustive ones
- Feature Engineering in determining factors that contribute to the likelihood of churn or not + game specific features

## Methods
---
### Model Evaluation:
- Compares traditional ML (Logistic Regression, Gradient Boosting) and deep learning (CNN, LSTM)
- Achieves 85–93% accuracy across three different games
- Uses brute force with different feature combinations to find the best AUC

## Evaluation
---
### Advantages
- Feature ranking provides targetted approach to game improvement
- Adaptable framework across multiple games with feature engineering
- Computationally efficient yet with high accuracy: can provide insight in how to improve gameplay

### Limitations
- OP/CP durations and features are different across games
- Format of data may change across casual/non-casual games
- Limits of data publicity (IP addresses, etc) since many studies do not provide public game data

## Implementation
---
Implemented a logistic regression model with game 1 and game 2 since game 3's data was unable to be processed normally.
Focused on feature engineering with 10 common features: active duration, best score, best score index, best sub mean count, best sub mean ratio,
mean score, consecutive play ratio, standard deviation score, and worst score. Game 2 had specific features purchase count and highest price.
Used balancing methods against an imbalanced dataset with a high churn rate.

