# Air Quality Relative Humidity (RH) Prediction

This project follows the submitted Colab notebook's workflow: UCI Air Quality data, data cleaning, median imputation, StandardScaler, Linear Regression, regression metrics, a saved Pickle model, and a FastAPI prediction server.

## Dataset
The official UCI Air Quality dataset is downloaded automatically by `train_model.py`:
https://archive.ics.uci.edu/dataset/360/air+quality

The dataset's missing-value marker is `-200`. The target is `RH` (Relative Humidity, %). Date and Time are excluded from the model features.

## Run locally (Windows PowerShell)

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python train_model.py
uvicorn app:app --host 127.0.0.1 --port 8000
```

Open:
- API home: http://127.0.0.1:8000/
- Interactive API documentation: http://127.0.0.1:8000/docs
- Health check: http://127.0.0.1:8000/health

## Test prediction
Open `/docs`, expand `POST /predict`, click **Try it out**. Supply every feature listed by `GET /` in the `values` object. Feature names and order are returned by the API home endpoint. Example request shape:

```json
{
  "values": {
    "CO(GT)": 2.6,
    "PT08.S1(CO)": 1360,
    "NMHC(GT)": 150,
    "C6H6(GT)": 11.9,
    "PT08.S2(NMHC)": 1046,
    "NOx(GT)": 166,
    "PT08.S3(NOx)": 1056,
    "NO2(GT)": 113,
    "PT08.S4(NO2)": 1692,
    "PT08.S5(O3)": 1268,
    "T": 13.6,
    "AH": 0.7578
  }
}
```

The sample values are illustrative; use real feature measurements for meaningful predictions. The exact required feature list is shown by the running API.

## Deploy on Render
1. Push this folder to a GitHub repository.
2. In Render, choose **New + → Blueprint** and connect that repository.
3. Render reads `render.yaml`, installs dependencies, trains the model, and starts Uvicorn.
4. After deployment succeeds, copy the public `onrender.com` service URL. The interactive docs are at `<your-service-url>/docs`.

A public server URL cannot be created by running the notebook locally; it is assigned by the hosting provider after deployment.

## Methodology note
The original notebook fills missing values on the full dataset before splitting, then fits another imputer after the split. This project avoids that leakage: the imputer and scaler are fitted on the training split only, then applied to the test split. The model type, target, split ratio, random seed, and evaluation metrics remain aligned with the notebook.

## Dataset citation
Vito, S. (2008). Air Quality. UCI Machine Learning Repository. https://doi.org/10.24432/C59K5F
