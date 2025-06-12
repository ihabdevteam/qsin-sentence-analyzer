import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from modules.db_utils import init_supabase_client
from modules.analysis_utils import get_data_for_sentence, estimate_snr50_for_sentence

st.set_page_config(page_title="점수 분석", layout="wide")
st.title("Quick-SIN 개별 문장 분석 페이지 (SNR-50 추정) 📊")
st.write("분석하고 싶은 문장 번호를 입력하고 분석을 실행하세요.")

# --- 클라이언트 초기화 ---
supabase = init_supabase_client()
if not supabase:
    st.stop()

# --- 1. 분석할 문장 및 데이터 소스 선택 ---
st.header("1. 분석 대상 선택")
sentence_id_to_analyze = st.number_input(
    "분석할 문장 번호(index)",
    min_value=1,
    max_value=360,
    value=1,
    help="1부터 360 사이의 숫자를 입력하세요."
)
use_dummy_data = st.checkbox(
    "dummy_ 접두사가 붙은 테스트 데이터만 사용",
    value=True,
    help="체크 시 session_id가 'dummy_'로 시작하는 데이터만 분석합니다. 체크 해제 시 그 외의 데이터를 분석합니다."
)

if st.button(f"🔍 문장 {sentence_id_to_analyze}번 데이터 분석 실행"):

    # --- 2. 데이터 로드 및 중간 결과 표시 (디버깅) ---
    st.header("2. 데이터 처리 과정 확인")
    with st.spinner(f"DB에서 문장 {sentence_id_to_analyze}번의 데이터를 로드하고 전처리 중입니다..."):
        processed_data = get_data_for_sentence(supabase, sentence_id_to_analyze, use_dummy_data)

    if processed_data.empty:
        st.error(f"문장 {sentence_id_to_analyze}번에 대한 분석을 진행할 수 없습니다. 데이터가 없거나 유효한 SNR 레벨이 연결되지 않았습니다.")
    else:        
        st.info("선택한 문장에 대해 데이터베이스에서 조회하고 정답률을 계산한 결과입니다. 이 데이터를 기반으로 분석을 시작합니다.")

        full_sentence_text = processed_data['full_sentence'].iloc[0]
        st.markdown(f"**분석 대상 문장: \"{full_sentence_text}\"**")

        display_df = processed_data.drop(columns=['full_sentence', 'sentence_id'])
        st.dataframe(display_df)

        # --- 3. SNR-50 추정 및 결과 표시 ---
        st.header("3. 분석 결과")
        with st.spinner("로지스틱 회귀 모델을 학습하고 SNR-50을 추정합니다..."):
            result = estimate_snr50_for_sentence(processed_data)

        status = result.get('status')
        if status == 'Success':
            snr50_val = result.get('snr_50')
            slope_val = result.get('slope')            
            st.success(f"분석 성공!  |  추정 SNR-50: **{snr50_val:.2f} dB** |  기울기: **{slope_val:.2f} %/dB**")
            st.info("""
                    - **추정 SNR-50 (dB)**: 피험자가 이 문장의 단어를 50% 확률로 맞추는 데 필요한 신호 대 잡음비(Signal-to-Noise Ratio)입니다. **값이 낮을수록 더 시끄러운 환경에서도 잘 들리는 쉬운 문장**임을 의미합니다.

                    - **기울기 (%/dB)**: SNR-50 지점 부근에서 SNR이 1dB 변할 때마다 정답률이 몇 %씩 변하는지를 나타내는 **민감도 지표**입니다. **값이 높을수록 소음 변화에 따라 난이도가 급격하게 변하는 문장**임을 의미합니다.
                    """)            
        else:
            st.error(f"분석 실패: **{status}**")
            st.warning("데이터 분포를 확인해주세요. 신뢰도 있는 분석을 위해서는 최소 3개 이상의 다양한 SNR 레벨에 대한 데이터가 필요합니다.")

        # --- 4. 시각화 ---
        st.header("4. Psychometric Function Curve")
        plot_data = result.get('plot_data')
        model = result.get('model')
        snr50_val = result.get('snr_50')

        if not plot_data.empty:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=plot_data['snr_level'],
                y=plot_data['correct_rate'],
                mode='markers',
                name='SNR 레벨별 평균 정답률',
                marker=dict(size=12, color='blue', symbol='diamond')
            ))

            if model and snr50_val is not None:
                x_range = np.linspace(plot_data['snr_level'].min() - 5, plot_data['snr_level'].max() + 5, 100)
                y_curve = model.predict_proba(x_range.reshape(-1, 1))[:, 1]
                
                fig.add_trace(go.Scatter(
                    x=x_range, y=y_curve, mode='lines', name='로지스틱 회귀 곡선', line=dict(color='red', width=2)
                ))
                fig.add_vline(x=snr50_val, line_width=2, line_dash="dash", line_color="green",
                              annotation_text=f"SNR-50: {snr50_val:.2f} dB", annotation_position="top right")
                fig.add_hline(y=0.5, line_width=2, line_dash="dash", line_color="green")
            
            fig.update_layout(
                title=f"문장 {sentence_id_to_analyze}번 : \"{full_sentence_text}\"",
                xaxis_title="SNR Level (dB)",
                yaxis_title="정답률 (Correct Rate)",
                yaxis_range=[-0.05, 1.05],
                legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("시각화할 데이터가 없습니다.")
