Model Architecture & Frameworks
The project evaluates three primary models to handle the fraud detection task. Because fraud cases are extremely rare, accuracy is not a reliable metric; instead, these models are optimized for Precision-Recall AUC (PR-AUC) and F1-Score.

1. Random Forest (model_randomforest.pkl)
Type: Ensemble Bagging Model (Decision Trees)

Purpose: Serves as a robust baseline. Random Forest handles non-linear relationships well and is inherently resistant to overfitting when tuned correctly.

Handling Imbalance: Integrates with class weighting mechanisms or pre-sampled data from imbalance_handler.py to ensure the trees learn distinct patterns of the minority (fraud) class.

2. LightGBM (model_lightgbm.pkl)
Type: Gradient Boosting Machine (Leaf-wise tree growth)

Purpose: Built for speed and high performance on large datasets. LightGBM optimizes the gradient boosting process by focusing on the errors (residuals) of previous trees.

Key Benefits: Highly efficient memory usage, native support for handling imbalanced datasets via the scale_pos_weight hyperparameter, and typically yields a superior Precision-Recall curve compared to traditional tree models.

3. Advanced Cyclical Model (model_advancedcyclical.pkl)
Type: Feature-engineered / Specialized Ensemble Model

Purpose: Designed to capture time-dependent, periodic, and cyclical patterns inherent in transaction data (e.g., time of day, day of the week, or spending velocities over rolling windows).

Key Benefits: Transforms raw timestamps into cyclical sine and cosine representations, helping the underlying estimator identify fraud rings that operate during specific hours or structural intervals.

Core Pipeline Modules
data_pipeline.py: Handles missing values, scaling of transaction amounts, and structural preprocessing to output clean training and testing partitions (X_train, y_train, etc.).

imbalance_handler.py: Controls the strategy used to tackle class imbalance, ensuring models aren't biased toward predicting only legitimate transactions.

model_training.py: Contains the execution scripts to train the models, perform hyperparameter tuning, and output the leaderboard performance tracking.

app.py: Exposes the trained serializable models via an interface or API to make real-time predictions on incoming transaction streams.

Performance Tracking
Model metrics are saved directly to phase3_leaderboard.csv. Visual performance validation, specifically showcasing how well the models balance precision (avoiding false accusations) vs. recall (catching actual fraud), can be found in phase3_pr_curve.png.
