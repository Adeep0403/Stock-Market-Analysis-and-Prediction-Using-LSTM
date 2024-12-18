Stock Market Analysis and Prediction Using LSTM: A Case Study on Apple Stock

Overview
This project explores the use of Long Short-Term Memory (LSTM) networks to predict the adjusted closing prices of Apple Inc. (AAPL) stocks. The study examines two distinct approaches for stock market prediction:
1) Comprehensive Feature Integration (Approach 1): Incorporates stock prices of peer companies, macroeconomic indicators, and sentiment data.
2) Minimalistic Design (Approach 2): Relies solely on Apple’s lagged historical stock prices.
The results reveal the strengths and trade-offs of both approaches, emphasizing the balance between simplicity and capturing complex market dynamics.

Features
1) Predicts Apple Inc. stock prices using historical and contextual data.
2) Implements two approaches:
 a) Approach 1: Incorporates external factors like macroeconomic indicators, sentiment analysis, and peer stock prices.
 b) Approach 2: Focuses solely on historical Apple stock prices for predictions.
3) Compares the accuracy and computational efficiency of both approaches.
4) Evaluation using metrics like MAE, MSE, R-squared, and MAPE.

Dataset
The dataset includes:
1) Stock Prices: Adjusted closing prices for Apple, Meta, Amazon, Google, and Microsoft, spanning January 1, 2016, to August 30, 2019 (sourced from Yahoo Finance).
2) Macroeconomic Indicators: Data such as GDP, CPI, unemployment rates, and interest rates (sourced from FRED API).
3) Sentiment Data: Twitter polarity and volume metrics for sentiment analysis (sourced from Kaggle).

Technologies Used
1) Python
2) TensorFlow/Keras: For building and training the LSTM model.
3) Pandas and NumPy: For data manipulation and analysis.
4) Scikit-learn: For preprocessing and evaluation metrics.
5) Matplotlib/Seaborn: For data visualization.

Methodology
1) Data Preprocessing
 a) Missing Data Handling:
  i) Forward fill for time-series gaps (e.g., weekends, holidays).
 ii) Linear interpolation for stock data gaps.
 b) Feature Scaling:
  i) Applied Min-Max scaling to normalize features.
 c) Feature Engineering:
  i) Created lag features for a 10-day window of historical stock prices.
2) LSTM Model Architecture
 a) Input Layer: Accepts lagged stock prices, sentiment data, and macroeconomic indicators.
 b) LSTM Layers:
  i) Two layers with 100 units each, leveraging temporal dependencies.
 ii) Dropout layers added for regularization.
 c) Dense Layer: Outputs the predicted stock price.
 d) Optimization: Trained with Adam optimizer and MSE as the loss function.
3) Training and Evaluation
 a) Data split: 80% training, 10% validation, 10% testing.
 b) Models trained over 50 epochs with a batch size of 32.
 c) Evaluation metrics:
  i) Mean Absolute Error (MAE): Measures average prediction error.
 ii) Mean Squared Error (MSE): Penalizes larger errors.
iii) R-squared (R²): Measures model fit to the data.
 iv) Mean Absolute Percentage Error (MAPE): Expresses error as a percentage.

Results
Approach 1: Comprehensive Feature Integration
 a) R²: 0.9690
 b) MAE: 0.6203
c) Broader insights into stock price dynamics, capturing macroeconomic and sentiment-driven influences.

Approach 2: Minimalistic Design
 a) R²: 0.9812
 b) MAE: 0.4351
 c) Simpler and computationally efficient, focusing on historical stock price patterns.

Setup Instructions
1) Clone the repository:
git clone https://github.com/your_username/stock-market-prediction-lstm.git
cd stock-market-prediction-lstm
2) Install required dependencies:
pip install -r requirements.txt
3) Add the datasets (historical stock prices, macroeconomic data, sentiment data) to the project directory.
4) Run the main script:
python main.py

Usage
1) Preprocess and normalize the dataset.
2) Train the LSTM model using either Approach 1 or Approach 2.
3) Evaluate the model on test data using metrics such as MAE, MSE, and R².
4) Visualize predictions using provided plots.

Future Work
1) Enhance feature engineering by incorporating real-time data and advanced selection techniques.
2) Optimize hyperparameters using grid search or random search.
3) Explore hybrid models combining the strengths of both approaches.
4) Extend the dataset to include more companies or broader market indices for generalized predictions.

License
This project is licensed under the MIT License.

Acknowledgments
1) Yahoo Finance and FRED API for data sourcing.
2) Kaggle for sentiment analysis datasets.
3) Documentation from TensorFlow, Keras, and Scikit-learn.
