import sys
import os

# Fix encoding issue on Windows
if sys.platform == 'win32':
    os.environ['PYTHONIOENCODING'] = 'utf-8'
    # Redirect stdout to use utf-8
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    if hasattr(sys.stderr, 'reconfigure'):
        sys.stderr.reconfigure(encoding='utf-8')

from fastapi import FastAPI, HTTPException
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from vnstock import Listing, Quote
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from typing import Dict, Any, List
import os
import joblib
from service.model_service_wrapper import run_model_on_top100
def _find_repo_root(start_path: str) -> str:
    cur = os.path.abspath(start_path)
    for _ in range(6):
        candidate = cur
        if os.path.isdir(os.path.join(candidate, 'data')):
            return candidate
        parent = os.path.dirname(candidate)
        if parent == candidate:
            break
        cur = parent
    return os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

app = FastAPI(title="VNStock API - Top 100 Stocks")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:5173",
    "http://127.0.0.1:5173",
],
    allow_credentials=True,
    allow_methods=["*"],
)

# --- Helper Functions ---

# Simple in-memory caches to reduce provider calls and avoid rate limits
CACHE = {
    "top100_list": {"ts": None, "data": []},
    "top100_history": {},  # keyed by days: { days: {"ts": datetime, "data": {...}, "metadata": {...}} }
    "symbol_history": {},  # keyed by f"{symbol}|{days}": {"ts": datetime, "df": pd.DataFrame}
    "model_input": {},     # keyed by f"{symbol}|{source}|{days}": {"ts": datetime, "df": pd.DataFrame}
}

CACHE_TTL_SYMBOLS_SECONDS = 600  # 10 minutes
CACHE_TTL_HISTORY_SECONDS = 300   # 5 minutes
CACHE_TTL_SYMBOL_HISTORY_SECONDS = 600  # 10 minutes for per-symbol history
CACHE_TTL_MODEL_INPUT_SECONDS = 600     # 10 minutes for model input per symbol

def _now():
    return datetime.now()

def _is_fresh(ts, ttl_seconds):
    return ts is not None and (_now() - ts).total_seconds() < ttl_seconds

def _cache_dir() -> str:
    d = os.path.join(os.path.dirname(__file__), 'cache')
    try:
        os.makedirs(d, exist_ok=True)
    except Exception:
        pass
    return d

def _save_csv_safe(path: str, df: pd.DataFrame):
    try:
        df.to_csv(path, index=False)
    except Exception as e:
        # Avoid breaking flow if cannot save
        print(f"Could not save CSV {path}: {e}")

def get_top100_symbols():
    """Lấy danh sách Top 100 mã chứng khoán từ CSV"""
    # Return cached list if still fresh
    cached = CACHE.get("top100_list", {})
    if _is_fresh(cached.get("ts"), CACHE_TTL_SYMBOLS_SECONDS):
        return cached.get("data", [])

    try:
        # Read from local top_100_stocks.csv instead of API
        repo_root = _find_repo_root(os.path.dirname(__file__))
        csv_path = os.path.join(repo_root, 'data', 'raw', 'top_100_stocks.csv')
        
        if not os.path.exists(csv_path):
            print(f"Top 100 CSV not found at {csv_path}")
            if cached.get("data"):
                return cached.get("data")
            return []
        
        df = pd.read_csv(csv_path)
        
        # Try to find symbol column
        if 'symbol' in df.columns:
            symbols = df['symbol'].dropna().astype(str).str.upper().tolist()
        elif 'ticker' in df.columns:
            symbols = df['ticker'].dropna().astype(str).str.upper().tolist()
        elif len(df.columns) > 0:
            symbols = df.iloc[:, 0].dropna().astype(str).str.upper().tolist()
        else:
            symbols = []

        # Update cache
        CACHE["top100_list"] = {"ts": _now(), "data": symbols}
        return symbols
    except Exception as e:
        print(f"Lỗi khi lấy danh sách Top 100: {e}")
        import traceback
        traceback.print_exc()
        # Fallback to stale cache if present
        if cached.get("data"):
            return cached.get("data")
        return []

def get_stock_history(symbol: str, start_date: str, end_date: str):
    """Lấy dữ liệu lịch sử của một mã cụ thể"""
    try:
        # Khởi tạo đối tượng Quote như trong tài liệu
        quote = Quote(symbol=symbol, source='VCI') 
        # Hàm history thường dùng định dạng YYYY-MM-DD
        try:
            df = quote.history(start=start_date, end=end_date, interval='1D')
        except SystemExit as se:
            # Provider rate limit; surface as empty to trigger fallback
            print(f"Rate limit while fetching history for {symbol}: {se}")
            return pd.DataFrame()
        return df
    except Exception as e:
        print(f"Lỗi khi lấy dữ liệu {symbol}: {e}")
        return pd.DataFrame()

def _normalize_history_df(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """
    Chuẩn hóa schema dữ liệu lịch sử về các cột bắt buộc:
    time, open, high, low, close, volume, symbol
    """
    if df is None or df.empty:
        return pd.DataFrame()

    # Rename time-like column to 'time'
    time_candidates = ['time', 'date', 'datetime', 'tradingDate']
    if 'time' not in df.columns:
        for c in time_candidates:
            if c in df.columns:
                df = df.rename(columns={c: 'time'})
                break

    # Normalize price/volume columns if alternative names exist
    rename_map: Dict[str, str] = {}
    if 'open' not in df.columns:
        for c in ['Open', 'o', 'open_price']:
            if c in df.columns:
                rename_map[c] = 'open'
                break
    if 'high' not in df.columns:
        for c in ['High', 'h', 'high_price']:
            if c in df.columns:
                rename_map[c] = 'high'
                break
    if 'low' not in df.columns:
        for c in ['Low', 'l', 'low_price']:
            if c in df.columns:
                rename_map[c] = 'low'
                break
    if 'close' not in df.columns:
        for c in ['Close', 'c', 'close_price', 'adj_close', 'price']:
            if c in df.columns:
                rename_map[c] = 'close'
                break
    if 'volume' not in df.columns:
        for c in ['Volume', 'vol', 'volume_match', 'total_volume']:
            if c in df.columns:
                rename_map[c] = 'volume'
                break

    if rename_map:
        df = df.rename(columns=rename_map)

    required = ['time', 'open', 'high', 'low', 'close', 'volume']
    # If any required missing, try to fill minimally
    for col in required:
        if col not in df.columns:
            df[col] = None

    # Keep only required columns
    df = df[required].copy()

    # Ensure time is string ISO-like
    try:
        df['time'] = pd.to_datetime(df['time']).dt.strftime('%Y-%m-%d')
    except Exception:
        # leave as-is if cannot parse
        pass

    # Add symbol column
    df['symbol'] = str(symbol).upper()

    # Sort ascending by time
    try:
        df = df.sort_values('time')
    except Exception:
        pass

    return df

def _slice_last_n(df: pd.DataFrame, n: int) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    if len(df) <= n:
        return df.tail(n)
    return df.iloc[-n:]

# --- Endpoints ---

@app.get("/")
def read_root():
    return {"message": "Welcome to VNStock API. Use /top100-history to get data."}

@app.get("/top100-list")
def get_top100_list():
    """Trả về danh sách Top 100 mã chứng khoán"""
    symbols = get_top100_symbols()
    return {"count": len(symbols), "symbols": symbols}

@app.get("/top100-history")
def get_top100_history_data(days: int = 30):
    """
    Lấy dữ liệu lịch sử của Top 100 mã.
    - days: Số ngày quá khứ muốn lấy (mặc định 30 ngày).
    """
    # Tính toán ngày bắt đầu và kết thúc
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
    
    # Serve cached history if fresh
    cached_days = CACHE["top100_history"].get(days)
    if cached_days and _is_fresh(cached_days.get("ts"), CACHE_TTL_HISTORY_SECONDS):
        return {
            "metadata": cached_days.get("metadata"),
            "data": cached_days.get("data"),
        }

    symbols = get_top100_symbols()
    if not symbols:
        # Try serving stale cached history if available
        if cached_days:
            return {
                "metadata": cached_days.get("metadata"),
                "data": cached_days.get("data"),
            }
        raise HTTPException(status_code=500, detail="Không thể lấy danh sách Top 100")

    result_data = {}
    for sym in symbols:
        df = get_stock_history(sym, start_date, end_date)
        if not df.empty:
            result_data[sym] = df.to_dict(orient='records')
        else:
            # On rate limit or no data, keep previous cached symbol data if present
            if cached_days and cached_days.get("data", {}).get(sym):
                result_data[sym] = cached_days["data"][sym]
            else:
                result_data[sym] = "No data found"

    payload = {
        "metadata": {
            "source": "Local CSV + VNStock",
            "start_date": start_date,
            "end_date": end_date,
            "group": "Top100",
        },
        "data": result_data,
    }

    # Update cache
    CACHE["top100_history"][days] = {"ts": _now(), "data": result_data, "metadata": payload["metadata"]}
    return payload

@app.get("/stock/{symbol}")
def get_single_stock(symbol: str, days: int = 30):
    """Lấy lịch sử của 1 mã bất kỳ"""
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
    symbol_u = symbol.upper()

    # Try cache first
    cache_key = f"{symbol_u}|{int(days)}"
    cached = CACHE["symbol_history"].get(cache_key)
    if cached and _is_fresh(cached.get("ts"), CACHE_TTL_SYMBOL_HISTORY_SECONDS):
        df_cached: pd.DataFrame = cached.get("df")
        if df_cached is not None and not df_cached.empty:
            return {
                "symbol": symbol_u,
                "data": df_cached.to_dict(orient='records')
            }
    
    df = get_stock_history(symbol_u, start_date, end_date)
    if df.empty:
        raise HTTPException(status_code=404, detail="Symbol not found or no data")

    # Normalize and cache + persist
    df_norm = _normalize_history_df(df, symbol_u)
    CACHE["symbol_history"][cache_key] = {"ts": _now(), "df": df_norm}
    try:
        out_path = os.path.join(_cache_dir(), f"history_{symbol_u}.csv")
        _save_csv_safe(out_path, df_norm)
    except Exception:
        pass

    return {
        "symbol": symbol_u,
        "data": df_norm.to_dict(orient='records')
    }

@app.get("/model-input/{symbol}")
def get_model_input(symbol: str, days: int = 50):
    """
    Trả về input gồm đúng 50 dòng cho một mã cổ phiếu,
    với các cột: time, open, high, low, close, volume, symbol.

    - days: số ngày quá khứ cần lấy tối thiểu để đảm bảo 50 dòng.
    """
    symbol = symbol.upper()

    # Tính khoảng thời gian lấy dữ liệu (lấy dư để đảm bảo đủ 50 bản ghi)
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=max(days, 60))).strftime('%Y-%m-%d')

    df: pd.DataFrame = pd.DataFrame()

    # Try cache first (model_input cache stores last-50 normalized rows)
    cache_key_inp = f"{symbol}|{int(days)}"
    cached_inp = CACHE["model_input"].get(cache_key_inp)
    if cached_inp and _is_fresh(cached_inp.get("ts"), CACHE_TTL_MODEL_INPUT_SECONDS):
        df_cached: pd.DataFrame = cached_inp.get("df")
        if df_cached is not None and len(df_cached) >= 50:
            df_50 = _slice_last_n(df_cached, 50)
            return {
                "symbol": symbol,
                "count": len(df_50),
                "data": df_50.to_dict(orient='records')
            }

    # Always try provider (vnstock)
    df = get_stock_history(symbol, start_date, end_date)
    df = _normalize_history_df(df, symbol)

    # Fallback to local CSV only if provider thiếu dữ liệu
    if df.empty or len(df) < 50:
        try:
            # Build path to local CSV within repo
            repo_root = _find_repo_root(os.path.dirname(__file__))
            local_csv = os.path.join(repo_root, 'data', 'raw', 'ta', 'vietnam_stock_price_history_2022-10-31_2025-10-31.csv')
            if os.path.exists(local_csv):
                df_local = pd.read_csv(local_csv)
                # Try common column names
                # Ensure symbol filter
                sym_col = 'symbol' if 'symbol' in df_local.columns else ('ticker' if 'ticker' in df_local.columns else None)
                if sym_col is None:
                    raise Exception('symbol column not found in local CSV')
                df_local = df_local[df_local[sym_col].astype(str).str.upper() == symbol]

                # Map columns to required
                # time
                time_col = None
                for c in ['time', 'date', 'datetime', 'tradingDate']:
                    if c in df_local.columns:
                        time_col = c
                        break
                if time_col is None:
                    # If no time column, cannot proceed
                    raise Exception('time-like column not found in local CSV')

                rename_map = {time_col: 'time'}
                # price columns
                for src, dst, alts in [
                    ('open', 'open', ['Open','o','open_price']),
                    ('high', 'high', ['High','h','high_price']),
                    ('low', 'low', ['Low','l','low_price']),
                    ('close', 'close', ['Close','c','close_price','adj_close','price']),
                    ('volume', 'volume', ['Volume','vol','volume_match','total_volume']),
                ]:
                    if src in df_local.columns:
                        rename_map[src] = dst
                    else:
                        for alt in alts:
                            if alt in df_local.columns:
                                rename_map[alt] = dst
                                break

                df_local = df_local.rename(columns=rename_map)
                # Keep required columns; create missing if needed
                required = ['time', 'open', 'high', 'low', 'close', 'volume']
                for col in required:
                    if col not in df_local.columns:
                        df_local[col] = None
                df_local = df_local[required].copy()
                df_local['symbol'] = symbol
                try:
                    df_local['time'] = pd.to_datetime(df_local['time']).dt.strftime('%Y-%m-%d')
                except Exception:
                    pass
                df_local = df_local.sort_values('time')
                df = df_local
        except Exception as e:
            print(f"Local CSV fallback failed: {e}")

    # Slice last 50
    df_50 = _slice_last_n(df, 50)
    if df_50.empty or len(df_50) < 50:
        raise HTTPException(status_code=404, detail="Không đủ dữ liệu 50 dòng cho mã này")

    # Cache and persist model input
    CACHE["model_input"][cache_key_inp] = {"ts": _now(), "df": df_50}
    try:
        out_path = os.path.join(_cache_dir(), f"model_input_{symbol}.csv")
        _save_csv_safe(out_path, df_50)
    except Exception:
        pass

    return {
        "symbol": symbol,
        "count": len(df_50),
        "data": df_50.to_dict(orient='records')
    }

@app.get("/predict/{symbol}")
def predict_symbol(symbol: str, days: int = 70, source: str = 'VNStock'):
    """
    Dự báo cho ngày mới nhất sử dụng mô hình lưu tại server/model/best_model.pkl.
    Trả về nhãn dự báo và xác suất mua (nếu có).
    """
    # Build input using same logic as /model-input
    symbol_u = symbol.upper()
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=max(days, 70))).strftime('%Y-%m-%d')

    # Chuẩn hóa source
    src_in = (source or '').strip().lower()
    if src_in in {'api', 'vnstock', 'provider', 'remote', 'live'}:
        source_norm = 'VNStock'
    elif src_in in {'local', 'csv', 'offline'}:
        source_norm = 'local'
    else:
        source_norm = source

    df: pd.DataFrame = pd.DataFrame()
    if source_norm.lower() == 'vnstock':
        df = get_stock_history(symbol_u, start_date, end_date)
        df = _normalize_history_df(df, symbol_u)

    if df.empty or len(df) < 50 or source_norm.lower() == 'local':
        try:
            repo_root = _find_repo_root(os.path.dirname(__file__))
            local_csv = os.path.join(repo_root, 'data', 'raw', 'ta', 'vietnam_stock_price_history_2022-10-31_2025-10-31.csv')
            if os.path.exists(local_csv):
                df_local = pd.read_csv(local_csv)
                sym_col = 'symbol' if 'symbol' in df_local.columns else ('ticker' if 'ticker' in df_local.columns else None)
                if sym_col is None:
                    raise Exception('symbol column not found in local CSV')
                df_local = df_local[df_local[sym_col].astype(str).str.upper() == symbol_u]
                time_col = None
                for c in ['time', 'date', 'datetime', 'tradingDate']:
                    if c in df_local.columns:
                        time_col = c
                        break
                if time_col is None:
                    raise Exception('time-like column not found in local CSV')
                rename_map = {time_col: 'time'}
                for src, dst, alts in [
                    ('open', 'open', ['Open','o','open_price']),
                    ('high', 'high', ['High','h','high_price']),
                    ('low', 'low', ['Low','l','low_price']),
                    ('close', 'close', ['Close','c','close_price','adj_close','price']),
                    ('volume', 'volume', ['Volume','vol','volume_match','total_volume']),
                ]:
                    if src in df_local.columns:
                        rename_map[src] = dst
                    else:
                        for alt in alts:
                            if alt in df_local.columns:
                                rename_map[alt] = dst
                                break
                df_local = df_local.rename(columns=rename_map)
                required = ['time', 'open', 'high', 'low', 'close', 'volume']
                for col in required:
                    if col not in df_local.columns:
                        df_local[col] = None
                df_local = df_local[required].copy()
                df_local['symbol'] = symbol_u
                try:
                    df_local['time'] = pd.to_datetime(df_local['time']).dt.strftime('%Y-%m-%d')
                except Exception:
                    pass
                df_local = df_local.sort_values('time')
                df = df_local
        except Exception as e:
            print(f"Local CSV fallback failed (predict): {e}")

    df_50 = _slice_last_n(df, 50)
    if df_50.empty or len(df_50) < 50:
        raise HTTPException(status_code=404, detail="Không đủ dữ liệu 50 dòng cho mã này")

    # Load model and predict
    model_path = os.path.join(os.path.dirname(__file__), 'model', 'best_model.pkl')
    if not os.path.exists(model_path):
        raise HTTPException(status_code=500, detail="Model file not found")
    pipeline = joblib.load(model_path)

    y_pred = pipeline.predict(df_50)
    try:
        y_prob = pipeline.predict_proba(df_50)
        prob_buy = float(y_prob[0,1])
    except Exception:
        prob_buy = None

    return {
        "symbol": symbol_u,
        "date": df_50.iloc[-1]['time'],
        "prediction": y_pred[0] if len(y_pred) > 0 else None,
        "prob_buy": prob_buy,
        "rows": len(df_50),
    }


@app.get("/predict-top100")
def predict_top100(days: int = 60, limit: int = None, save_csv: bool = False):
    """
    Chạy dự báo cho Top 100 mã và trả về list được sắp xếp theo xác suất mua giảm dần.

    - source: 'local' dùng dữ liệu đã tính sẵn (khuyến nghị nhanh/stable), 'VNStock' gọi provider.
    - limit: giới hạn số mã đầu vào (ví dụ 20 để debug nhanh).
    - save_csv: nếu True, ghi thêm file top100_predictions.csv tại thư mục server.
    """
    try:
        # Use the running API port (5000) to fetch model inputs
        df_res = run_model_on_top100(server_url='http://127.0.0.1:5000', days=days, source='VNStock', limit=limit)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

    if df_res is None or df_res.empty:
        raise HTTPException(status_code=500, detail="Không có kết quả dự báo")

    # Clean NaN/inf to make JSON-safe
    try:
        df_res = df_res.replace([np.inf, -np.inf], np.nan)
        # Cast to object to keep None, then replace all NaN/NaT with None for JSON safety
        df_res = df_res.astype(object).where(pd.notna(df_res), None)
    except Exception:
        pass

    records = df_res.to_dict(orient='records')

    if save_csv:
        out_path = os.path.join(os.path.dirname(__file__), 'top100_predictions.csv')
        try:
            df_res.to_csv(out_path, index=False)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Không ghi được CSV: {e}")

    payload = {
        "count": len(records),
        "source": "VNStock",
        "data": records,
    }

    # Ensure JSON-safe (convert NaN/Inf to null) using FastAPI encoder
    encoded = jsonable_encoder(payload)
    return JSONResponse(content=encoded)


@app.get("/predict-top100-csv")
def predict_top100_csv(limit: int = None, sort: bool = True):
    """
    Trả về kết quả dự đoán từ file CSV đã lưu sẵn (không tính toán lại).
    - Đọc file `web/server/top100_predictions.csv`
    - Mặc định sắp xếp theo `prob_buy` giảm dần (NA xuống cuối). Có thể tắt sort.
    - Có thể giới hạn số dòng trả về bằng `limit`.
    """
    csv_path = os.path.join(os.path.dirname(__file__), 'top100_predictions.csv')
    if not os.path.exists(csv_path):
        raise HTTPException(status_code=404, detail="CSV kết quả chưa tồn tại. Hãy chạy /predict-top100 với save_csv=true hoặc tạo file trước.")

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Không đọc được CSV: {e}")

    # Chuẩn hóa kiểu dữ liệu
    try:
        if 'prob_buy' in df.columns:
            df['prob_buy'] = pd.to_numeric(df['prob_buy'], errors='coerce')
    except Exception:
        pass

    # Sắp xếp theo prob_buy giảm dần nếu yêu cầu
    if sort and 'prob_buy' in df.columns:
        try:
            df = df.sort_values(['prob_buy', 'status'], ascending=[False, True], na_position='last')
        except Exception:
            pass

    # Giới hạn số dòng nếu có
    if limit is not None:
        try:
            df = df.head(int(limit))
        except Exception:
            pass

    # Làm sạch NaN/Inf để trả JSON an toàn
    try:
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.astype(object).where(pd.notna(df), None)
    except Exception:
        pass

    records = df.to_dict(orient='records')
    payload = {
        "count": len(records),
        "source": "csv",
        "data": records,
    }
    encoded = jsonable_encoder(payload)
    return JSONResponse(content=encoded)