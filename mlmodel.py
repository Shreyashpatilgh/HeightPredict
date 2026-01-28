import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib

# Sample dataset
data = {
    'Age': [24, 30, 22, 35, 28, 40],
    'Weight': [55, 85, 68, 90, 72, 95],
    'Height': [165, 175, 180, 178, 170, 182]
}

df = pd.DataFrame(data)

# Features and target
X = df[['Age', 'Weight']]
y = df['Height']

# Train model
model = LinearRegression()
model.fit(X, y)


# Save model
joblib.dump(model, "height_predictor_model.pkl")
