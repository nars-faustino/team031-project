# %%
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle

# Load dataset
df = pd.read_csv(r'../data/processed-data.csv')

# Identify categorical columns
categorical_cols = ["season", "city1", "airport_iata_1", 
                    "airport_name_1", "airport_name_concat_1", "state_1", 
                    "city2", "airport_iata_2", "airport_name_2", "airport_name_concat_2", 
                    "state_2", "carrier_lg", "carrier_lg_name", "carrier_lg_name_concat", 
                    "carrier_low", "carrier_low_name", "carrier_low_name_concat"]

# Apply Label Encoding to categorical columns
label_encoders = {}  # Store encoders to use for predictions
for col in categorical_cols:
    if col in df.columns:  # Avoid errors if some columns don't exist
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))  # Convert to string then encode
        label_encoders[col] = le

# Define Features (X) & Target (y)
X = df.drop(columns=["fare"])  # Change "fare" to match your target column
y = df["fare"]

# Split dataset into training & testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the XGBoost model with extracted hyperparameters
model = xgb.XGBRegressor(
    objective="reg:squarederror",
    eval_metric=["rmse", "mae"],  # Fix: Remove "mse" and use "rmse"
    alpha=float("0.087752231735666"),
    colsample_bytree=float("0.7862115041563089"),
    learning_rate=float("0.23141413048164433"),  # "eta" in XGBoost
    gamma=float("30.689156603791872"),
    reg_lambda=float("0.005991930971856792"),  # "lambda" in XGBoost
    max_depth=int("4"),
    min_child_weight=float("8.289439875037123"),
    n_estimators=int("181"),  # "num_round" in SageMaker
    subsample=float("0.7986090224937361"),
    verbosity=1
)

# Train the model
print("Training XGBoost Model...")
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
r2 = r2_score(y_test, y_pred)
mape = (abs(y_test - y_pred) / y_test).mean() * 100  # MAPE Calculation

print("\nModel Training Completed")
print(f"Mean Absolute Error (MAE): {mae}")
print(f"Mean Squared Error (MSE): {mse}")
print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"R² Score: {r2}")
print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")  # Display MAPE


# Save the model & encoders
model.save_model("xgboost_airfare_model.json")
print("Model saved as 'xgboost_airfare_model.json'")

with open("label_encoders.pkl", "wb") as f:
    pickle.dump(label_encoders, f)
print("Label encoders saved as 'label_encoders.pkl'")


