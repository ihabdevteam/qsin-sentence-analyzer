import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression

def _process_raw_data(data: list):
    if not data:
        return pd.DataFrame()

    records = []
    for item in data:
        report_info = item.get('test_reports_qsin')
        if report_info and report_info.get('snr_level') is not None:
            records.append({
                'session_id': report_info.get('session_id'),
                'sentence_id': item['index'],
                'sentences': item['sentences'],
                'total_score': item['total_score'],
                'full_sentence': item['full_sentence'],
                'score': item['score'],
                'snr_level': report_info['snr_level']
            })
    
    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    df['num_keywords'] = df['sentences'].apply(lambda x: len(x) if isinstance(x, list) else 0)
    df = df[df['num_keywords'] > 0]
    df['correct_rate'] = df['total_score'] / df['num_keywords']
    
    return df

@st.cache_data(ttl=600)
def get_all_sentence_data(_supabase_client, use_dummy_prefix: bool):    
    try:
        all_records = []
        page = 0
        page_size = 1000  # Supabase default

        while True:
            start_index = page * page_size
            end_index = start_index + page_size - 1

            query = _supabase_client.table('score_qsin').select(
                'index, sentences, total_score, full_sentence, score, test_reports_qsin!inner(snr_level, session_id)'
            )

            if use_dummy_prefix:
                query = query.like('test_reports_qsin.session_id', 'dummy_%')
            else:
                query = query.not_.like('test_reports_qsin.session_id', 'dummy_%')
            
            # 페이지네이션을 위한 range 적용
            response = query.range(start_index, end_index).execute()
            
            current_page_data = response.data
            
            if not current_page_data:
                break
            
            all_records.extend(current_page_data)            
            if len(current_page_data) < page_size:
                break
                
            page += 1

        df = _process_raw_data(all_records)
        if df.empty:
            return pd.DataFrame()

        return df[['session_id', 'sentence_id', 'full_sentence', 'snr_level', 'score', 'total_score', 'correct_rate']]

    except Exception as e:
        st.error(f"전체 데이터를 불러오는 중 오류가 발생했습니다: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=600)
def get_data_for_sentence(_supabase_client, sentence_id: int, use_dummy_prefix: bool):    
    try:
        all_records = []
        page = 0
        page_size = 1000  # Supabase 기본 제한

        while True:
            start_index = page * page_size
            end_index = start_index + page_size - 1

            query = _supabase_client.table('score_qsin').select(
                'index, sentences, total_score, full_sentence, score, test_reports_qsin!inner(snr_level, session_id)'
            ).eq('index', sentence_id)
            
            if use_dummy_prefix:
                query = query.like('test_reports_qsin.session_id', 'dummy_%')
            else:
                query = query.not_.like('test_reports_qsin.session_id', 'dummy_%')

            # 페이지네이션을 위한 range 적용
            response = query.range(start_index, end_index).execute()
            
            current_page_data = response.data
            
            if not current_page_data:
                break # 더 이상 데이터가 없으면 루프 종료
            
            all_records.extend(current_page_data)
            
            # 현재 페이지에서 가져온 데이터가 페이지 사이즈보다 작으면 마지막 페이지임
            if len(current_page_data) < page_size:
                break
                
            page += 1

        df = _process_raw_data(all_records)
        if df.empty:
            return pd.DataFrame()
        
        return df[['session_id', 'sentence_id', 'full_sentence', 'snr_level', 'score', 'total_score', 'correct_rate']]

    except Exception as e:
        st.error(f"데이터를 불러오는 중 오류가 발생했습니다: {e}")
        return pd.DataFrame()

def estimate_snr50_for_sentence(data: pd.DataFrame):
    """
    단일 문장에 대한 데이터(snr_level, correct_rate)를 받아
    로지스틱 회귀 분석으로 SNR-50과 기울기를 추정합니다.
    """
    agg_data = data.groupby('snr_level')['correct_rate'].mean().reset_index()

    if len(agg_data) < 3:
        return {'status': 'Error: Not Enough Data Points', 'snr_50': None, 'slope': None, 'model': None, 'plot_data': agg_data}
    
    if agg_data['correct_rate'].min() == agg_data['correct_rate'].max():
        return {'status': 'Error: All Same Results', 'snr_50': None, 'slope': None, 'model': None, 'plot_data': agg_data}

    X = agg_data[['snr_level']]
    y = agg_data['correct_rate']
    
    try:
        n_trials = 100 
        X_resampled = np.repeat(agg_data['snr_level'].values, n_trials).reshape(-1, 1)
        y_resampled = [val for rate in agg_data['correct_rate'] for val in np.random.binomial(1, rate, n_trials)]
        
        model = LogisticRegression(solver='liblinear')
        model.fit(X_resampled, y_resampled)

        intercept = model.intercept_[0]
        coef = model.coef_[0][0]

        if abs(coef) < 1e-6:
            return {'status': 'Error: Zero Slope', 'snr_50': None, 'slope': None, 'model': model, 'plot_data': agg_data}

        snr_50 = -intercept / coef
        slope = coef / 4 * 100

        return {
            'status': 'Success', 
            'snr_50': float(snr_50), 
            'slope': float(slope), 
            'model': model, 
            'plot_data': agg_data
        }

    except Exception as e:
        return {'status': f'Error: {e}', 'snr_50': None, 'slope': None, 'model': None, 'plot_data': agg_data}
