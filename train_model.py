"""Train the Relative Humidity (RH) regression model using the UCI Air Quality data."""
from pathlib import Path
from io import BytesIO
from zipfile import ZipFile
from urllib.request import urlopen
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DATA_FILE = DATA_DIR / "AirQualityUCI.csv"
MODEL_FILE = BASE_DIR / "air_quality_model.pkl"
DATA_URL = "https://archive.ics.uci.edu/static/public/360/air+quality.zip"

def download_dataset():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if DATA_FILE.exists():
        return
    print("Downloading official UCI Air Quality dataset...")
    with urlopen(DATA_URL, timeout=60) as response:
        payload = response.read()
    with ZipFile(BytesIO(payload)) as archive:
        member = next((n for n in archive.namelist() if n.lower().endswith("airqualityuci.csv")), None)
        if member is None:
            raise FileNotFoundError("AirQualityUCI.csv was not found in the UCI dataset archive.")
        DATA_FILE.write_bytes(archive.read(member))

def train():
    download_dataset()
    df = pd.read_csv(DATA_FILE, sep=";", decimal=",", na_values=-200)
    df = df.drop_duplicates()
    df = df.dropna(axis=1, how="all")
    df = df.drop(["Date", "Time"], axis=1, errors="ignore")
    df = df.apply(pd.to_numeric, errors="coerce")
    if "RH" not in df.columns:
        raise ValueError("Target column RH is missing from the dataset.")
    df = df.dropna(subset=["RH"])
    X = df.drop("RH", axis=1)
    y = df["RH"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    # Fit preprocessing only on training data to avoid data leakage.
    imputer = SimpleImputer(strategy="median")
    X_train = imputer.fit_transform(X_train)
    X_test = imputer.transform(X_test)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    model = LinearRegression()
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, pred)
    mse = mean_squared_error(y_test, pred)
    rmse = float(np.sqrt(mse))
    r2 = r2_score(y_test, pred)
    bundle = {
        "model": model,
        "imputer": imputer,
        "scaler": scaler,
        "features": X.columns.tolist(),
    }
    with MODEL_FILE.open("wb") as f:
        pickle.dump(bundle, f)
    print(f"Rows used: {len(df)}")
    print(f"MAE: {mae:.4f}")
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R2: {r2:.4f}")
    print(f"Saved model: {MODEL_FILE}")

if __name__ == "__main__":
    train()
