import pandas as pd

df=pd.DataFrame({
    'Name': ['A', 'B', 'C', 'D'],
    'Age': [23, 45, 12, 36],    
    'Score': [85, 90, 78, 92],
    'Grade': ['A', 'B', 'C', 'A']
})
print(df)
print(df.describe())
# Load model
from sklearn.linear_model import LinearRegression
import joblib
model=joblib.load("height_predictor_model.pkl")
# Make predictions
new_data = pd.DataFrame({
    'Age': [25, 32, 29],
    'Weight': [70, 80, 75]
})
predictions = model.predict(new_data)
print("Predicted Heights:", predictions)