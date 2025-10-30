import streamlit as st
import pandas as pd
from modules.db_utils import init_supabase_client
from modules.analysis_utils import (    
    get_all_sentence_data, 
    estimate_snr50_for_sentence,
    analyze_all_sentences,
    display_analysis_metrics,
    create_psychometric_plot,
    create_combined_psychometric_plot
)

st.set_page_config(page_title="점수 분석", layout="wide")
st.title("Quick-SIN 개별 문장 분석 페이지 (SNR-50 추정) 📊")

# --- 클라이언트 초기화 ---
supabase = init_supabase_client()
if not supabase:
    st.stop()

# --- 세션 상태 초기화: 분석 결과 및 메타데이터만 유지 (메모리 최적화) ---
if 'analysis_results_df' not in st.session_state:
    st.session_state.analysis_results_df = None
if 'data_snr_range' not in st.session_state:
    st.session_state.data_snr_range = None
if 'temp_download_data' not in st.session_state:
    st.session_state.temp_download_data = None

st.header("0. 전체 원본 데이터 다운로드")
download_use_dummy = st.checkbox(
    "테스트 데이터(dummy_ 접두사) 다운로드",
    value=False,
    key='download_dummy_check',
    help="체크 시 session_id가 'dummy_'로 시작하는 데이터만 다운로드합니다. 체크 해제 시 그 외의 데이터를 다운로드합니다."
)

# 다운로드 버튼을 누르면 데이터를 조회하고, st.download_button을 즉시 생성합니다.
if st.button("📥 전체 데이터 조회 및 다운로드 준비"):
    with st.spinner("전체 데이터를 DB에서 조회 중입니다..."):
        all_data_df = get_all_sentence_data(supabase, use_dummy_prefix=download_use_dummy)
        if not all_data_df.empty:
            # SNR 범위 저장 (그래프용)
            st.session_state.data_snr_range = (
                all_data_df['snr_level'].min(),
                all_data_df['snr_level'].max()
            )
            
            with st.spinner("전체 문장에 대한 SNR-50 분석을 실행 중입니다..."):
                st.session_state.analysis_results_df = analyze_all_sentences(all_data_df)
            
            # 다운로드용 CSV 데이터를 임시로 저장 (다운로드 후 자동 삭제)
            st.session_state.temp_download_data = all_data_df.to_csv(index=False).encode('utf-8-sig')
            
            st.success("데이터를 조회하고 분석하여 세션에 저장했습니다. 아래에서 계속 확인할 수 있습니다.")
        else:
            st.session_state.analysis_results_df = None
            st.session_state.data_snr_range = None
            st.session_state.temp_download_data = None
            st.warning("다운로드할 데이터가 없습니다.")

# 데이터 초기화 버튼
cols_reset = st.columns([1, 1, 6])
with cols_reset[0]:
    if st.button("🧹 데이터 초기화"):
        st.session_state.analysis_results_df = None
        st.session_state.data_snr_range = None
        st.session_state.temp_download_data = None
        st.rerun()

# --- 조회/분석 결과 표시 (세션 유지) ---
if st.session_state.temp_download_data is not None:
    analysis_results_df = st.session_state.analysis_results_df
    
    # 다운로드 버튼
    st.success("데이터 조회 완료! 아래 버튼을 눌러 CSV 파일을 저장하세요.")
    st.download_button(
        label="다운로드 준비 완료. 클릭하여 저장 (CSV)",
        data=st.session_state.temp_download_data,
        file_name="all_qsin_scores.csv",
        mime="text/csv",
    )

    # 전체 데이터 분석 결과 표시
    st.header("전체 데이터 분석 결과")
    if analysis_results_df is not None and not analysis_results_df.empty:
        st.success(f"총 {len(analysis_results_df)}개 문장에 대한 분석이 완료되었습니다.")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("분석된 문장 수", len(analysis_results_df))
        with col2:
            st.metric("평균 SNR-50", f"{analysis_results_df['snr_50'].mean():.2f} dB")
        with col3:
            st.metric("평균 기울기", f"{analysis_results_df['slope'].mean():.2f} %/dB")
        with col4:
            # Validity 등급별 카운트
            ideal_count = len(analysis_results_df[analysis_results_df['validity'] == 'Ideal'])
            st.metric("Ideal 등급 문장 수", ideal_count)

        st.subheader("문장별 분석 결과")
        display_df = analysis_results_df.copy()
        display_df['snr_50'] = display_df['snr_50'].round(2)
        display_df['slope'] = display_df['slope'].round(2)
        if 'total_score_sum' in display_df.columns:
            display_df['total_score_sum'] = display_df['total_score_sum'].round(0)
        if 'avg_score' in display_df.columns:
            display_df['avg_score'] = display_df['avg_score'].round(2)
        
        st.dataframe(
            display_df,
            column_config={
                "sentence_id": "문장 ID",
                "full_sentence": st.column_config.TextColumn("문장", width="large"),
                "snr_50": "SNR-50 (dB)",
                "slope": "기울기 (%/dB)",
                "validity": "등급",
                "total_score_sum": "총 점수",
                "avg_score": "평균 점수",
                "data_points": "데이터 수",
                "snr_levels": "SNR 레벨 수"
            },
            use_container_width=True
        )

        analysis_csv = analysis_results_df.to_csv(index=False).encode('utf-8-sig')
        st.download_button(
            label="📊 분석 결과 다운로드 (CSV)",
            data=analysis_csv,
            file_name="qsin_analysis_results.csv",
            mime="text/csv",
        )

        # 문장별 데이터 시각화 (모든 문장 겹쳐서)
        col_header, col_help = st.columns([0.85, 0.15])
        with col_header:
            st.subheader("문장별 데이터 시각화 (전체 겹쳐보기)")
        with col_help:
            with st.popover("💡 도움말"):
                st.markdown("""
                #### 그래프 해석 가이드
                이 그래프는 모든 문장의 Psychometric Function을 겹쳐서 보여주어, 각 문장의 난이도와 변별력을 한눈에 비교할 수 있도록 돕습니다.

                - **기울기 (곡선의 가파름)**: **문장의 변별력** 또는 **측정 품질**을 나타냅니다.
                    - 곡선이 가파를수록(기울기 값이 높을수록) SNR 변화에 민감하게 반응하는 좋은 측정 문항입니다.

                - **SNR-50 (마커의 x축 위치 및 색상)**: **문장의 난이도**와 **목표값(2dB) 근접성**을 나타냅니다.
                    - `x` 값이 낮을수록 (왼쪽에 있을수록) 더 쉬운 문장입니다.
                    - **마커 색상 의미:**
                        - 🟢 **녹색**: 이상적 난이도 (0.5 ~ 3.5 dB)
                        - 🟠 **주황색**: 수용 가능 난이도 (-1.0 ~ 5.0 dB)
                        - 🔴 **빨간색**: 목표 난이도에서 많이 벗어남
                        - ⚫ **회색**: 신뢰도 낮은 추정치 (`Extrapolated`)

                **좋은 문장을 선별하려면, `녹색` 또는 `주황색` 마커를 가지면서 곡선이 가파른(기울기가 높은) 문장을 우선적으로 고려해야 합니다.**
                """)

        st.caption("총 360개 문장을 하나의 그래프에 겹쳐서 표시합니다. 로지스틱 곡선의 색상은 각 문장의 등급을 나타냅니다.")

        available_sentence_ids = analysis_results_df['sentence_id'].tolist()
        selected_sentence_ids = available_sentence_ids

        if not selected_sentence_ids:
            st.info("시각화할 문장이 없습니다.")
        else:
            fig = create_combined_psychometric_plot(
                sentence_ids=selected_sentence_ids,
                include_logistic=True,
                include_mean=False,
                show_legend=False,
                precomputed_results=analysis_results_df,
                snr_range=st.session_state.data_snr_range,
            )
            if fig:
                st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("분석 가능한 문장이 없습니다. 데이터를 확인해주세요.")

st.divider()

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
    "테스트 데이터(dummy_ 접두사)만 사용",
    value=False,
    key='analyze_dummy_check',
    help="체크 시 session_id가 'dummy_'로 시작하는 데이터만 분석합니다. 체크 해제 시 그 외의 데이터를 분석합니다."
)

if st.button(f"🔍 문장 {sentence_id_to_analyze}번 데이터 분석 실행"):
    try:
        # --- 2. 데이터 로드 및 중간 결과 표시 (디버깅) ---
        st.header("2. 데이터 처리 과정 확인")
        with st.spinner(f"DB에서 문장 {sentence_id_to_analyze}번의 데이터를 로드하고 전처리 중입니다..."):
            processed_data = get_all_sentence_data(supabase, use_dummy_data, sentence_id_to_analyze)

        if processed_data.empty:
            st.error(f"문장 {sentence_id_to_analyze}번에 대한 분석을 진행할 수 없습니다. 데이터가 없거나 유효한 SNR 레벨이 연결되지 않았습니다.")
        else:        
            st.info("선택한 문장에 대해 데이터베이스에서 조회하고 정답률을 계산한 결과입니다. 이 데이터를 기반으로 분석을 시작합니다.")

            full_sentence_text = processed_data['full_sentence'].iloc[0]
            st.markdown(f"**분석 대상 문장: \"{full_sentence_text}\"**")

            display_df = processed_data.drop(columns=['session_id', 'full_sentence', 'sentence_id'])
            st.dataframe(display_df)

            csv_data = processed_data.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 이 문장 데이터 다운로드 (CSV)",
                data=csv_data,
                file_name=f"sentence_{sentence_id_to_analyze}_data.csv",
                mime="text/csv",
            )
            st.header("3. 분석 결과")
            with st.spinner("로지스틱 회귀 모델을 학습하고 SNR-50을 추정합니다..."):
                result = estimate_snr50_for_sentence(processed_data)

            status = result.get('status')
            if status == 'Success':
                snr50_val = result.get('snr_50')
                slope_val = result.get('slope')
                display_analysis_metrics(snr50_val, slope_val)
            else:
                st.error(f"분석 실패: **{status}**")
                st.warning("데이터 분포를 확인해주세요. 신뢰도 있는 분석을 위해서는 최소 3개 이상의 다양한 SNR 레벨에 대한 데이터가 필요합니다.")

            # --- 4. 시각화 ---
            st.header("4. Psychometric Function Curve")
            fig = create_psychometric_plot(processed_data, result, sentence_id_to_analyze)
            if fig:
                st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"데이터를 처리하는 중 오류가 발생했습니다: {e}")
        st.info("간헐적인 네트워크 오류일 수 있습니다. 아래 버튼을 눌러 캐시를 초기화하고 다시 시도해 보세요.")
        if st.button("🔄 캐시 지우고 재시도"):
            st.cache_data.clear()
            st.rerun()