import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression

@st.cache_data(ttl=600)
def get_data_for_sentence(_supabase_client, sentence_id: int, use_dummy_prefix: bool):
    """
    특정 문장 번호에 해당하는 모든 점수 데이터를 DB에서 조회하고,
    분석에 필요한 형태로 전처리하여 반환합니다.
    use_dummy_prefix 플래그에 따라 session_id를 필터링합니다.
    """
    try:
        # Step 1: 필터링 조건에 맞는 test_report_qsin의 ID 목록을 가져옵니다.
        query = _supabase_client.table('test_reports_qsin').select('id')
        if use_dummy_prefix:
            query = query.like('session_id', 'dummy_%')
        else:
            query = query.not_.like('session_id', 'dummy_%')
        
        reports_res = query.execute()
        report_ids = [r['id'] for r in reports_res.data]

        if not report_ids:
            filter_text = "'dummy_' 접두사가 있는" if use_dummy_prefix else "'dummy_' 접두사가 없는"
            st.warning(f"{filter_text} 테스트 리포트를 찾을 수 없습니다.")
            return pd.DataFrame()

        # Step 2: 필터링된 리포트 ID와 문장 번호로 score_qsin 데이터를 조회합니다.
        # 수정된 부분: full_sentence를 select 목록에 추가
        response = _supabase_client.table('score_qsin').select(
            'index, sentences, total_score, full_sentence, test_reports_qsin(snr_level)'
        ).eq('index', sentence_id).in_('test_result_id', report_ids).execute()
        
        data = response.data
        if not data:
            return pd.DataFrame()

        # Step 3: 데이터를 평탄화하고 전처리합니다.
        records = []
        for item in data:
            report_info = item.get('test_reports_qsin')
            if report_info and 'snr_level' in report_info:
                records.append({
                    'sentence_id': item['index'],
                    'sentences': item['sentences'],
                    'total_score': item['total_score'],
                    'full_sentence': item['full_sentence'], # 수정된 부분: 레코드에 full_sentence 추가
                    'snr_level': report_info['snr_level']
                })
        
        if not records:
            return pd.DataFrame()

        df = pd.DataFrame(records)
        df['num_keywords'] = df['sentences'].apply(lambda x: len(x) if isinstance(x, list) else 0)
        df = df[df['num_keywords'] > 0]
        df['correct_rate'] = df['total_score'] / df['num_keywords']
        
        # 수정된 부분: full_sentence 컬럼을 반환 데이터에 포함
        return df[['sentence_id', 'full_sentence', 'snr_level', 'total_score', 'correct_rate']]

    except Exception as e:
        st.error(f"데이터를 불러오는 중 오류가 발생했습니다: {e}")
        return pd.DataFrame()


def estimate_snr50_for_sentence(data: pd.DataFrame):
    """
    단일 문장에 대한 데이터(snr_level, correct_rate)를 받아
    로지스틱 회귀분석으로 SNR-50과 기울기를 추정합니다.
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
