import os
from typing import List, Optional
import pandas as pd
import numpy as np
import requests
import joblib
from datetime import datetime

def _find_repo_root(start_path: Optional[str] = None) -> str:
    """Ascend directories to locate repo root containing 'data' folder."""
    if start_path is None:
        start_path = os.path.dirname(__file__)
    cur = os.path.abspath(start_path)
    for _ in range(6):
        candidate = cur
        if os.path.isdir(os.path.join(candidate, 'data')):
            return candidate
        parent = os.path.dirname(candidate)
        if parent == candidate:
            break
        cur = parent
    return os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))

def get_top100_symbols(csv_path: Optional[str] = None) -> List[str]:
    """Đọc danh sách Top 100 mã cổ phiếu từ CSV và trả về list symbol."""
    if csv_path is None:
        repo_root = _find_repo_root()
        csv_path = os.path.join(repo_root, 'data', 'raw', 'top_100_stocks.csv')

    df = pd.read_csv(csv_path)
    if 'symbol' in df.columns:
        return df['symbol'].dropna().astype(str).str.upper().tolist()
    # fallback if column name differs
    for alt in ['ticker', 'Symbol']:
        if alt in df.columns:
            return df[alt].dropna().astype(str).str.upper().tolist()
    return []

def build_model_input(symbol: str, server_url: str = 'http://127.0.0.1:5000', days: int = 50, source: str = 'VNStock') -> pd.DataFrame:
    """
    Tạo input chuẩn cho model từ API hoặc CSV local.
    
    Yêu cầu output:
    - DataFrame có đúng 50 dòng (50 ngày gần nhất)
    - 7 cột bắt buộc: time, open, high, low, close, volume, symbol
    - Sắp xếp tăng dần theo thời gian (dòng cuối là ngày mới nhất)
    - Chỉ 1 mã cổ phiếu (tất cả dòng có cùng giá trị symbol)
    - Dữ liệu đã được làm sạch (NaN -> 0, inf -> 0)
    
    Args:
        symbol: Mã cổ phiếu cần dự báo
        server_url: URL của API server
        days: Số ngày dữ liệu cần lấy (mặc định 50)
        source: Nguồn dữ liệu ('VNStock' hoặc 'local')
    
    Returns:
        DataFrame 50 dòng x 7 cột, sẵn sàng cho model.predict()
    
    Raises:
        RuntimeError: Nếu không đủ dữ liệu hoặc không tìm thấy mã trong CSV
    """
    symbol = symbol.upper()
    
    # Cố gắng lấy từ API trước
    try:
        resp = requests.get(
            f"{server_url}/model-input/{symbol}", 
            params={'days': days}, 
            timeout=30
        )
        resp.raise_for_status()
        payload = resp.json()
        data = payload.get('data', [])
        df = pd.DataFrame(data)
        
        # Đảm bảo có đủ các cột bắt buộc
        required = ['time', 'open', 'high', 'low', 'close', 'volume', 'symbol']
        for col in required:
            if col not in df.columns:
                df[col] = None
        
        df = df[required].copy()
        
        # Chuyển đổi kiểu dữ liệu
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Sắp xếp theo thời gian
        try:
            df['time'] = pd.to_datetime(df['time'])
            df = df.sort_values('time')
            df['time'] = df['time'].dt.strftime('%Y-%m-%d')
        except Exception:
            df = df.sort_values('time')
        
        # Lấy đúng 50 dòng cuối
        df = df.iloc[-50:].copy()
        
        # Làm sạch dữ liệu
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.ffill().bfill().fillna(0)
        
        # Đảm bảo symbol đồng nhất
        df['symbol'] = symbol
        
        if len(df) < 50:
            raise ValueError(f"Không đủ 50 dòng (chỉ có {len(df)} dòng)")
        
        return df[required]
        
    except Exception as api_error:
        print(f"API lỗi cho {symbol}: {api_error}, chuyển sang CSV local")
        
        # Fallback sang CSV local
        repo_root = _find_repo_root()
        local_csv = os.path.join(repo_root, 'data', 'raw', 'ta', 'vietnam_stock_price_history_2022-10-31_2025-10-31.csv')
        
        if not os.path.exists(local_csv):
            raise RuntimeError(f'Không tìm thấy CSV local: {local_csv}')
        
        df_local = pd.read_csv(local_csv)
        
        # Tìm cột symbol
        sym_col = None
        for col_name in ['symbol', 'ticker', 'Symbol', 'Ticker']:
            if col_name in df_local.columns:
                sym_col = col_name
                break
        if sym_col is None:
            raise RuntimeError('Không tìm thấy cột symbol/ticker trong CSV local')
        
        # Lọc theo mã
        df_local = df_local[df_local[sym_col].astype(str).str.upper() == symbol].copy()
        
        if len(df_local) == 0:
            raise RuntimeError(f'Không tìm thấy mã {symbol} trong CSV local')
        
        # Tìm cột thời gian
        time_col = None
        for col_name in ['time', 'date', 'datetime', 'tradingDate', 'Date']:
            if col_name in df_local.columns:
                time_col = col_name
                break
        if time_col is None:
            raise RuntimeError('Không tìm thấy cột thời gian trong CSV local')
        
        # Chuẩn hóa tên cột
        rename_map = {time_col: 'time'}
        for src, dst, alts in [
            ('open', 'open', ['Open', 'o', 'open_price']),
            ('high', 'high', ['High', 'h', 'high_price']),
            ('low', 'low', ['Low', 'l', 'low_price']),
            ('close', 'close', ['Close', 'c', 'close_price', 'adj_close', 'price']),
            ('volume', 'volume', ['Volume', 'vol', 'volume_match', 'total_volume']),
        ]:
            if src in df_local.columns:
                rename_map[src] = dst
            else:
                for alt in alts:
                    if alt in df_local.columns:
                        rename_map[alt] = dst
                        break
        
        df_local = df_local.rename(columns=rename_map)
        
        # Đảm bảo có đủ cột bắt buộc
        required = ['time', 'open', 'high', 'low', 'close', 'volume']
        for col in required:
            if col not in df_local.columns:
                df_local[col] = 0
        
        df_local = df_local[required].copy()
        df_local['symbol'] = symbol
        
        # Chuyển đổi kiểu dữ liệu
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df_local[col] = pd.to_numeric(df_local[col], errors='coerce')
        
        # Sắp xếp và chuẩn hóa thời gian
        try:
            df_local['time'] = pd.to_datetime(df_local['time'])
            df_local = df_local.sort_values('time')
            df_local['time'] = df_local['time'].dt.strftime('%Y-%m-%d')
        except Exception:
            df_local = df_local.sort_values('time')
        
        # Lấy đúng 50 dòng cuối
        df_local = df_local.iloc[-50:].copy()
        
        # Làm sạch dữ liệu
        df_local = df_local.replace([np.inf, -np.inf], np.nan)
        df_local = df_local.ffill().bfill().fillna(0)
        
        if len(df_local) < 50:
            # Thử gọi provider để bổ sung nếu local không đủ
            try:
                resp2 = requests.get(
                    f"{server_url}/model-input/{symbol}",
                    params={'days': max(days, 70), 'source': 'VNStock'},
                    timeout=30,
                )
                resp2.raise_for_status()
                data2 = resp2.json().get('data', [])
                df2 = pd.DataFrame(data2)
                req2 = ['time','open','high','low','close','volume','symbol']
                for col in req2:
                    if col not in df2.columns:
                        df2[col] = None
                for col in ['open','high','low','close','volume']:
                    df2[col] = pd.to_numeric(df2[col], errors='coerce')
                try:
                    df2['time'] = pd.to_datetime(df2['time'])
                    df2 = df2.sort_values('time')
                    df2['time'] = df2['time'].dt.strftime('%Y-%m-%d')
                except Exception:
                    df2 = df2.sort_values('time')
                df2['symbol'] = symbol
                df2 = df2.iloc[-50:].copy()
                df2 = df2.replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0)
                if len(df2) >= 50:
                    return df2[['time','open','high','low','close','volume','symbol']]
            except Exception:
                pass
            raise RuntimeError(f'Không đủ 50 dòng cho {symbol} (chỉ có {len(df_local)} dòng)')
        
        return df_local[['time', 'open', 'high', 'low', 'close', 'volume', 'symbol']]

def build_model_features_input(symbol: str, server_url: str = 'http://127.0.0.1:5000', days: int = 60, source: str = 'VNStock') -> pd.DataFrame:
    """
    Build input feature DataFrame by computing technical indicators from raw data.
    Fetches raw OHLCV data via API or local fallback, then calculates indicators.
    Returns last 50 rows with all features needed for model prediction.
    """
    # Get raw OHLCV data
    df = build_model_input(symbol, server_url=server_url, days=days, source=source)
    
    if df is None or len(df) < 20:  # Need at least 20 rows for indicators
        return None
    
    # Ensure numeric types
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Fill any missing values
    df = df.ffill().bfill().fillna(0)
    
    # Calculate technical indicators
    # 1. Price-based features
    df['price_range'] = df['high'] - df['low']
    df['price_range_pct'] = (df['price_range'] / df['low']) * 100
    df['body_size_pct'] = abs((df['close'] - df['open']) / df['open']) * 100
    
    # 2. Moving Averages
    df['ma_5'] = df['close'].rolling(window=5, min_periods=1).mean()
    df['ma_20'] = df['close'].rolling(window=20, min_periods=1).mean()
    df['ma_50'] = df['close'].rolling(window=50, min_periods=1).mean()
    df['ma_5_divergence'] = ((df['close'] - df['ma_5']) / df['ma_5']) * 100
    df['ma_20_divergence'] = ((df['close'] - df['ma_20']) / df['ma_20']) * 100
    df['ma_50_divergence'] = ((df['close'] - df['ma_50']) / df['ma_50']) * 100
    
    # 3. RSI (Relative Strength Index)
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=1).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=1).mean()
    rs = gain / loss.replace(0, 1e-10)
    df['rsi_14'] = 100 - (100 / (1 + rs))
    
    # 4. MACD
    ema_12 = df['close'].ewm(span=12, adjust=False, min_periods=1).mean()
    ema_26 = df['close'].ewm(span=26, adjust=False, min_periods=1).mean()
    df['macd'] = ema_12 - ema_26
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False, min_periods=1).mean()
    df['macd_histogram'] = df['macd'] - df['macd_signal']
    
    # 5. Stochastic Oscillator
    low_14 = df['low'].rolling(window=14, min_periods=1).min()
    high_14 = df['high'].rolling(window=14, min_periods=1).max()
    df['stochastic_k'] = ((df['close'] - low_14) / (high_14 - low_14).replace(0, 1e-10)) * 100
    
    # 6. Volatility
    df['volatility_20'] = df['close'].rolling(window=20, min_periods=1).std()
    
    # 7. ATR (Average True Range)
    high_low = df['high'] - df['low']
    high_close = abs(df['high'] - df['close'].shift())
    low_close = abs(df['low'] - df['close'].shift())
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['atr_14'] = true_range.rolling(window=14, min_periods=1).mean()
    
    # 8. Bollinger Bands
    bb_ma = df['close'].rolling(window=20, min_periods=1).mean()
    bb_std = df['close'].rolling(window=20, min_periods=1).std()
    df['bb_upper'] = bb_ma + (2 * bb_std)
    df['bb_lower'] = bb_ma - (2 * bb_std)
    df['bb_width'] = ((df['bb_upper'] - df['bb_lower']) / bb_ma) * 100
    df['bb_position'] = ((df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower']).replace(0, 1e-10)) * 100
    
    # 9. Volume indicators
    df['volume_ma_20'] = df['volume'].rolling(window=20, min_periods=1).mean()
    df['volume_ratio'] = df['volume'] / df['volume_ma_20'].replace(0, 1e-10)
    
    # 10. OBV (On Balance Volume)
    obv = [0]
    for i in range(1, len(df)):
        if df['close'].iloc[i] > df['close'].iloc[i-1]:
            obv.append(obv[-1] + df['volume'].iloc[i])
        elif df['close'].iloc[i] < df['close'].iloc[i-1]:
            obv.append(obv[-1] - df['volume'].iloc[i])
        else:
            obv.append(obv[-1])
    df['obv'] = obv
    
    # 11. ADX (Average Directional Index)
    plus_dm = df['high'].diff()
    minus_dm = -df['low'].diff()
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm < 0] = 0
    atr = true_range.rolling(window=14, min_periods=1).mean()
    df['plus_di'] = 100 * (plus_dm.rolling(window=14, min_periods=1).mean() / atr.replace(0, 1e-10))
    df['minus_di'] = 100 * (minus_dm.rolling(window=14, min_periods=1).mean() / atr.replace(0, 1e-10))
    dx = 100 * abs(df['plus_di'] - df['minus_di']) / (df['plus_di'] + df['minus_di']).replace(0, 1e-10)
    df['adx'] = dx.rolling(window=14, min_periods=1).mean()
    
    # 12. Binary indicators
    df['Volume_Spike'] = (df['volume_ratio'] > 2).astype(int)
    df['RSI_Oversold'] = (df['rsi_14'] < 30).astype(int)
    df['RSI_Overbought'] = (df['rsi_14'] > 70).astype(int)
    df['Price_Above_MA20'] = (df['close'] > df['ma_20']).astype(int)
    df['Price_Above_MA50'] = (df['close'] > df['ma_50']).astype(int)
    
    # Replace inf/nan values
    df = df.replace([np.inf, -np.inf], 0).fillna(0)
    
    # Return last 50 rows with feature columns
    df = df.iloc[-50:]
    
    # Select feature columns (exclude time, symbol, intermediate calculations)
    exclude = {'time', 'symbol', 'ma_5', 'ma_20', 'ma_50', 'macd_signal', 'bb_upper', 'bb_lower'}
    feature_cols = [c for c in df.columns if c not in exclude]
    
    return df[feature_cols]
    
def _model_path_default() -> str:
    here = os.path.dirname(__file__)
    server_dir = os.path.dirname(here)
    return os.path.join(server_dir, 'model', 'best_model.pkl')

def predict_for_symbol(symbol: str, model_path: Optional[str] = None, server_url: str = 'http://127.0.0.1:5000', days: int = 60, source: str = 'local') -> dict:
    """
    Tạo input chuẩn cho 1 mã cổ phiếu và chạy dự báo bằng pipeline đã lưu.
    
    Pipeline yêu cầu:
    - Input: DataFrame 50 dòng x 7 cột (time, open, high, low, close, volume, symbol)
    - Dữ liệu sắp xếp theo thời gian tăng dần
    - Không có NaN/Inf (đã được làm sạch)
    
    Args:
        symbol: Mã cổ phiếu cần dự báo
        model_path: Đường dẫn tới file model (mặc định: server/model/best_model.pkl)
        server_url: URL API server để lấy dữ liệu
        days: Số ngày dữ liệu cần lấy
        source: Nguồn dữ liệu ('local' hoặc 'VNStock')
    
    Returns:
        Dict chứa: symbol, date, prediction, prob_buy, status
        - prediction: 0 (không mua) hoặc 1 (nên mua)
        - prob_buy: xác suất khuyến nghị mua (0-1)
        - status: 'ok', 'insufficient_input', hoặc 'error: ...'
    """
    if model_path is None:
        model_path = _model_path_default()
    
    try:
        # Bước 1: Load model pipeline trước để nhận biết kỳ vọng feature
        pipeline = joblib.load(model_path)

        expected = None
        if hasattr(pipeline, 'feature_names_in_'):
            expected = list(pipeline.feature_names_in_)
        else:
            # Thử lấy từ estimator cuối cùng trong sklearn Pipeline
            try:
                final_est = getattr(pipeline, 'steps', [])[ -1 ][1]
                if hasattr(final_est, 'feature_names_in_'):
                    expected = list(final_est.feature_names_in_)
            except Exception:
                expected = None

        needs_features = False
        if expected is not None:
            expected_set = set([str(x) for x in expected])
            markers = {'Price_Above_MA20','Price_Above_MA50','RSI_Overbought','RSI_Oversold','Volume_Spike','macd','rsi_14'}
            needs_features = len(expected_set & markers) > 0

        # Bước 2: Chuẩn bị input phù hợp với mô hình
        if needs_features:
            df_input = build_model_features_input(symbol, server_url=server_url, days=max(days,60), source=source)
            if df_input is None or len(df_input) < 50:
                return {"symbol": symbol.upper(), "status": "insufficient_input", "date": None, "prediction": None, "prob_buy": None}
            # Căn chỉnh cột theo expected
            if expected is not None:
                for col in expected:
                    if col not in df_input.columns:
                        df_input[col] = 0
                df_input = df_input[[c for c in expected if c in df_input.columns]]
        else:
            df_input = build_model_input(symbol, server_url=server_url, days=days, source=source)
            if df_input is None or len(df_input) < 50:
                return {"symbol": symbol.upper(), "status": "insufficient_input", "date": None, "prediction": None, "prob_buy": None}

        # Bước 3: Chạy dự báo
        y_pred = pipeline.predict(df_input)
        
        # Bước 4: Lấy xác suất (probability) nếu có
        try:
            y_prob = pipeline.predict_proba(df_input)
            prob_buy = float(y_prob[0, 1])  # Xác suất class 1 (mua)
        except Exception:
            prob_buy = None
        
        # Bước 5: Lấy ngày dự báo (ngày cuối cùng trong input)
        try:
            last_time = df_input.iloc[-1]['time'] if 'time' in df_input.columns else None
        except Exception:
            last_time = None
        
        return {
            "symbol": symbol.upper(),
            "date": last_time,
            "prediction": int(y_pred[0]) if len(y_pred) > 0 else None,
            "prob_buy": prob_buy,
            "status": "ok",
        }
        
    except Exception as e:
        return {
            "symbol": symbol.upper(),
            "date": None,
            "prediction": None,
            "prob_buy": None,
            "status": f"error: {str(e)}"
        }

def run_model_on_top100(model_path: Optional[str] = None, server_url: str = 'http://127.0.0.1:5000', days: int = 60, source: str = 'local', limit: Optional[int] = None) -> pd.DataFrame:
    """
    Chạy dự báo cho Top 100 mã cổ phiếu và trả về DataFrame kết quả.
    
    Quy trình:
    1. Đọc danh sách Top 100 từ CSV (data/raw/top_100_stocks.csv)
    2. Với mỗi mã:
       - Gọi API/CSV để lấy 50 dòng OHLCV chuẩn hóa
       - Load pipeline model
       - Chạy dự báo và lấy xác suất
    3. Tổng hợp kết quả và sắp xếp theo xác suất giảm dần
    
    Args:
        model_path: Đường dẫn file model (mặc định: server/model/best_model.pkl)
        server_url: URL API server
        days: Số ngày dữ liệu cần lấy
        source: 'local' (dùng CSV) hoặc 'VNStock' (gọi provider)
        limit: Giới hạn số mã dự báo (để test nhanh)
    
    Returns:
        DataFrame với các cột: symbol, date, prediction, prob_buy, status
        Sắp xếp theo prob_buy giảm dần (NA xuống cuối)
    """
    symbols = get_top100_symbols()
    if limit is not None:
        symbols = symbols[:limit]
    
    rows = []
    total = len(symbols)
    
    for idx, sym in enumerate(symbols, 1):
        print(f"Đang dự báo {idx}/{total}: {sym}")
        res = predict_for_symbol(sym, model_path=model_path, server_url=server_url, days=days, source=source)
        rows.append(res)
    
    df_res = pd.DataFrame(rows)
    
    # Đảm bảo có đủ các cột
    cols = ['symbol', 'date', 'prediction', 'prob_buy', 'status']
    for c in cols:
        if c not in df_res.columns:
            df_res[c] = None
    df_res = df_res[cols]
    
    # Sắp xếp theo xác suất giảm dần (NA xuống cuối)
    try:
        df_res['prob_buy'] = pd.to_numeric(df_res['prob_buy'], errors='coerce')
        df_res = df_res.sort_values(['prob_buy', 'status'], ascending=[False, True], na_position='last')
    except Exception:
        pass
    
    return df_res

if __name__ == '__main__':
    """
    Ví dụ sử dụng:
    
    # Dự báo cho 1 mã:
    python model.service.py --symbol VNM
    
    # Dự báo cho Top 100 (lưu CSV):
    python model.service.py --limit 100 --source local --out predictions.csv
    
    # Dự báo 10 mã đầu (test nhanh):
    python model.service.py --limit 10
    """
    import argparse
    parser = argparse.ArgumentParser(description='Chạy dự báo cho Top 100 mã cổ phiếu')
    parser.add_argument('--server_url', type=str, default='http://127.0.0.1:5000', help='URL API server')
    parser.add_argument('--days', type=int, default=60, help='Số ngày dữ liệu cần lấy')
    parser.add_argument('--source', type=str, default='local', choices=['local','VNStock'], help='Nguồn dữ liệu')
    parser.add_argument('--limit', type=int, default=None, help='Giới hạn số mã (để test)')
    parser.add_argument('--out', type=str, default='top100_predictions.csv', help='File output CSV')
    parser.add_argument('--symbol', type=str, default=None, help='Dự báo cho 1 mã cụ thể')
    args = parser.parse_args()

    # Nếu chỉ dự báo 1 mã
    if args.symbol:
        print(f'\n=== Dự báo cho {args.symbol} ===')
        result = predict_for_symbol(args.symbol, server_url=args.server_url, days=args.days, source=args.source)
        print(f"Symbol: {result['symbol']}")
        print(f"Date: {result['date']}")
        print(f"Prediction: {result['prediction']} ({'Mua' if result['prediction'] == 1 else 'Không mua'})")
        print(f"Probability Buy: {result['prob_buy']:.4f}" if result['prob_buy'] else "Probability: N/A")
        print(f"Status: {result['status']}")
    else:
        # Dự báo cho Top 100
        df_out = run_model_on_top100(
            server_url=args.server_url, 
            days=args.days, 
            source=args.source, 
            limit=args.limit
        )
        print('\n=== Top Predictions (Top 10) ===')
        try:
            print(df_out.head(10).to_string(index=False))
        except Exception:
            print(df_out.head(10))
        
        try:
            df_out.to_csv(args.out, index=False)
            print(f'\n✓ Đã lưu {len(df_out)} kết quả vào {args.out}')
        except Exception as e:
            print(f'\n✗ Không thể lưu CSV: {e}')
    
