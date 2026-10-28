from fastapi import FastAPI, Request, Form, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator
import joblib
import numpy as np
import pandas as pd
import re
import os

app = FastAPI(title="Прогноз стоимости недвижимости в Турции")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))
app.mount("/static", StaticFiles(directory="static"), name="static")

# 2. ЗАГРУЗКА АРТЕФАКТОВ И ДАННЫХ
def load_artifact(name):
    path = os.path.join(PROJECT_DIR, name)
    if not os.path.exists(path):
        raise FileNotFoundError(f" Не найден файл: {path}")
    return joblib.load(path)

MODEL = load_artifact('best_model.pkl')
SCALER = load_artifact('scaler.pkl')
FEATURE_COLS = list(MODEL.feature_names_in_)
FEATURES_TO_SCALE = load_artifact('features_to_scale.pkl')
FEATURES_TO_SCALE = [c for c in FEATURES_TO_SCALE if c in FEATURE_COLS]

METRICS_CSV = os.path.join(PROJECT_DIR, 'model_comparison_results.csv')
if os.path.exists(METRICS_CSV):
    METRICS_DF = pd.read_csv(METRICS_CSV)
else:
    METRICS_DF = pd.DataFrame([
        {"Модель": "Random Forest (tuned)", "R²_TRY": 0.7535, "MAE_TRY": 64222.88, "MAPE_TRY": 663.94},
        {"Модель": "Random Forest", "R²_TRY": 0.7509, "MAE_TRY": 64553.59, "MAPE_TRY": 684.82},
        {"Модель": "XGBoost", "R²_TRY": 0.7432, "MAE_TRY": 69678.13, "MAPE_TRY": 343.89},
        {"Модель": "LightGBM", "R²_TRY": 0.6887, "MAE_TRY": 78534.63, "MAPE_TRY": 392.71},
        {"Модель": "Decision Tree (tuned)", "R²_TRY": 0.6668, "MAE_TRY": 73303.53, "MAPE_TRY": 1260.34},
    ])

CITY_COORDS = {
    "Стамбул": {"lat": 41.0082, "lon": 28.9784},
    "Анталья": {"lat": 36.8969, "lon": 30.7133},
    "Измир": {"lat": 38.4237, "lon": 27.1428},
    "Анкара": {"lat": 39.9334, "lon": 32.8597},
    "Бурса": {"lat": 40.1826, "lon": 29.0665},
    "Мугла": {"lat": 37.2153, "lon": 28.3636},
    "Мерсин": {"lat": 36.8000, "lon": 34.6333},
    "Айдын": {"lat": 37.8444, "lon": 27.8458},
    "Денизли": {"lat": 37.7765, "lon": 29.0864},
    "Газиантеп": {"lat": 37.0662, "lon": 37.3833},
    "Коджаэли": {"lat": 40.8533, "lon": 29.8815},
    "Текирдаг": {"lat": 40.9833, "lon": 27.5167},
    "Сакарья": {"lat": 40.6940, "lon": 30.4360},
    "Хатай": {"lat": 36.4018, "lon": 36.3498},
    "Чанаккале": {"lat": 40.1553, "lon": 26.4142}
}

class PropertyInput(BaseModel):
    city: str = Field(..., min_length=1, max_length=50)
    room_count: str = Field(..., pattern=r"^\d+\+\d+$")
    size: float = Field(..., ge=10, le=1000)
    floor_no: int = Field(..., ge=0, le=150)
    total_floor_count: int = Field(..., ge=1, le=150)
    building_age: str = Field(...) 
    heating_type: str = Field(default="Kombi (Doğalgaz)")
    type: str = Field(default="Жильё")
    sub_type: str = Field(default="Квартира")
    listing_type: str = Field(default="Продажа")

    @field_validator('floor_no')
    @classmethod
    def floor_must_be_valid(cls, v, info):
        return v

def parse_rooms(x: str) -> float:
    numbers = re.findall(r'\d+', str(x))
    return sum(int(n) for n in numbers) if numbers else 0.0

def prepare_features(data: dict) -> pd.DataFrame:
    df = pd.DataFrame([data])
    replacements = {'20 и более': 20, 'Высокий первый этаж': 1, 'Yüksek Giriş': 1, '20 ve üzeri': 20}
    df['building_age'] = df['building_age'].replace(replacements)
    df['building_age'] = pd.to_numeric(df['building_age'], errors='coerce').fillna(0)
    
    df['rooms_numeric'] = df['room_count'].apply(parse_rooms)
    df['floor_ratio'] = df['floor_no'] / df['total_floor_count'].replace(0, 1)
    
    top_cities = ['Стамбул', 'Анталья', 'Измир', 'Бурса', 'Анкара', 'Мугла', 'Мерсин', 'Айдын', 'Денизли', 'Газиантеп', 'Коджаэли', 'Текирдаг', 'Сакарья', 'Хатай', 'Чанаккале']
    df['city_grouped'] = df['city'].apply(lambda x: x if x in top_cities else 'Другое')
    
    common_rooms = ['1+0', '1+1', '2+1', '3+1', '4+1', '5+1', '2+2', '3+2', '4+2', '6+1', '6+2', '7+1', '7+2', '8+1', '8+2']
    df['room_grouped'] = df['room_count'].apply(lambda x: x if x in common_rooms else 'Другое')
    
    nominal_cols = ['city', 'heating_type', 'type', 'sub_type', 'city_grouped', 'room_grouped']
    df_encoded = pd.get_dummies(df, columns=nominal_cols, drop_first=False, dtype=int)
    
    df_final = df_encoded.reindex(columns=FEATURE_COLS, fill_value=0)
    
    cols_to_scale = [c for c in FEATURES_TO_SCALE if c in df_final.columns]
    if cols_to_scale:
        df_final[cols_to_scale] = SCALER.transform(df_final[cols_to_scale])
    return df_final


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(request, "index.html")

@app.get("/map", response_class=HTMLResponse)
async def map_page(request: Request):
    return templates.TemplateResponse(request, "map.html")

@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request):
    return templates.TemplateResponse(request, "dashboard.html")

@app.get("/help", response_class=HTMLResponse)
async def help_page(request: Request):
    return templates.TemplateResponse(request, "help.html")

@app.get("/api/map_data")
async def get_map_data():
    mock_map_data = [
        {"city": "Стамбул", "lat": 41.0082, "lon": 28.9784, "avg_price": 4500000, "count": 85000},
        {"city": "Анталья", "lat": 36.8969, "lon": 30.7133, "avg_price": 3200000, "count": 42000},
        {"city": "Измир", "lat": 38.4237, "lon": 27.1428, "avg_price": 3800000, "count": 35000},
        {"city": "Анкара", "lat": 39.9334, "lon": 32.8597, "avg_price": 2900000, "count": 28000},
        {"city": "Бурса", "lat": 40.1826, "lon": 29.0665, "avg_price": 2500000, "count": 18000},
        {"city": "Мугла", "lat": 37.2153, "lon": 28.3636, "avg_price": 4100000, "count": 15000},
        {"city": "Мерсин", "lat": 36.8000, "lon": 34.6333, "avg_price": 1800000, "count": 12000},
        {"city": "Текирдаг", "lat": 40.9833, "lon": 27.5167, "avg_price": 2100000, "count": 8000},
        {"city": "Айдын", "lat": 37.8444, "lon": 27.8458, "avg_price": 1900000, "count": 7500},
        {"city": "Денизли", "lat": 37.7765, "lon": 29.0864, "avg_price": 1700000, "count": 6000},
    ]
    return mock_map_data

@app.get("/api/metrics")
async def api_metrics():
    return METRICS_DF.to_dict(orient='records')

@app.get("/api/feature_importance")
async def api_feature_importance():
    if hasattr(MODEL, 'feature_importances_'):
        importances = MODEL.feature_importances_
    else:
        importances = np.abs(MODEL.coef_)
    imp_df = pd.DataFrame({'feature': FEATURE_COLS, 'importance': importances})
    imp_df = imp_df.sort_values('importance', ascending=False).head(15)
    return imp_df.to_dict(orient='records')

@app.get("/api/dataset_stats")
async def api_dataset_stats():
    return {
        "total_objects": 253886,
        "avg_price": 262006,
        "median_price": 198000,
        "avg_size": 121.5,
        "top_city": "Стамбул",
        "top_room_type": "3+1"
    }

@app.get("/api/correlation")
async def api_correlation():
    return {
        "columns": ["price", "size", "building_age", "total_floor_count", "floor_no"],
        "data": [
            [1.00, 0.45, -0.10, 0.15, 0.12],
            [0.45, 1.00, -0.05, 0.10, 0.08],
            [-0.10, -0.05, 1.00, 0.30, 0.25],
            [0.15, 0.10, 0.30, 1.00, 0.60],
            [0.12, 0.08, 0.25, 0.60, 1.00]
        ]
    }

@app.post("/api/predict")
async def predict_api(input_data: PropertyInput):
    try:
        X = prepare_features(input_data.model_dump())
        pred_log = MODEL.predict(X)[0]
        pred_try = float(np.expm1(pred_log))
        return {"success": True, "predicted_price_try": round(pred_try, 2), "predicted_price_formatted": f"{pred_try:,.0f} TRY"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict", response_class=HTMLResponse)
async def predict_form(request: Request, city: str = Form(...), room_count: str = Form(...), size: float = Form(...), floor_no: int = Form(...), total_floor_count: int = Form(...), building_age: str = Form(...), heating_type: str = Form(...), type: str = Form(...), sub_type: str = Form(...), listing_type: str = Form(...)):
    try:
        input_data = {"city": city, "room_count": room_count, "size": size, "floor_no": floor_no, "total_floor_count": total_floor_count, "building_age": building_age, "heating_type": heating_type, "type": type, "sub_type": sub_type, "listing_type": listing_type}
        X = prepare_features(input_data)
        pred_try = float(np.expm1(MODEL.predict(X)[0]))
        return templates.TemplateResponse(request, "index.html", {"prediction": f"{pred_try:,.0f}", "input_data": input_data})
    except Exception as e:
        return templates.TemplateResponse(request, "index.html", {"error": str(e), "input_data": {"city": city, "size": size}})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)