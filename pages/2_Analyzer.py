import streamlit as st
import pandas as pd
import numpy as np
import io
from modules.db_utils import init_supabase_client
from modules.analysis_utils import (
    get_all_sentence_data,
    estimate_snr50_for_sentence,
    analyze_all_sentences,
    display_analysis_metrics,
    create_psychometric_plot,
    create_combined_psychometric_plot,
    calculate_dynamic_ranges,
    reclassify_with_absolute_criterion,
    analyze_subjects,
    TARGET_SNR50,
    ABSOLUTE_TOLERANCE,
    SLOPE_THRESHOLD,
    SENTENCE_SD_THRESHOLD,
    SNR_LOSS_GRADES,
)

st.set_page_config(page_title="점수 분석", layout="wide")
st.title("Quick-SIN 개별 문장 분석 페이지 (SNR-50 추정)")

def _is_hearing_loss(series: pd.Series) -> pd.Series:
    """hearing_loss 컬럼을 bool/str/None 모두 안전하게 처리하여 boolean mask 반환"""
    return series.map(lambda v: str(v).strip().lower() == 'true' if pd.notna(v) else False)

# --- 클라이언트 초기화 ---
supabase = init_supabase_client()
if not supabase:
    st.stop()

# --- 세션 상태 초기화 ---
for key in ['normal_results_df', 'hl_results_df', 'hl_subject_results_df',
            'data_snr_range', 'temp_download_data', 'total_data_rows',
            'normal_data_rows', 'hl_data_rows']:
    if key not in st.session_state:
        st.session_state[key] = None

# ============================================================
# 섹션 0: 전체 원본 데이터 다운로드
# ============================================================
st.header("0. 전체 원본 데이터 다운로드")
download_use_dummy = st.checkbox(
    "테스트 데이터(dummy_ 접두사) 다운로드",
    value=False,
    key='download_dummy_check',
    help="체크 시 session_id가 'dummy_'로 시작하는 데이터만 다운로드합니다. 체크 해제 시 그 외의 데이터를 다운로드합니다."
)

if st.button("전체 데이터 조회 및 다운로드 준비"):
    with st.spinner("전체 데이터를 DB에서 조회 중입니다..."):
        all_data_df = get_all_sentence_data(supabase, use_dummy_prefix=download_use_dummy)
        if not all_data_df.empty:
            st.session_state.total_data_rows = len(all_data_df)
            st.session_state.data_snr_range = (
                all_data_df['snr_level'].min(),
                all_data_df['snr_level'].max()
            )

            # hearing_loss 기준으로 분리 (bool/str/None 모두 안전하게 처리)
            hl_mask = _is_hearing_loss(all_data_df['hearing_loss'])
            normal_data = all_data_df[~hl_mask]
            hl_data = all_data_df[hl_mask]
            st.session_state.normal_data_rows = len(normal_data)
            st.session_state.hl_data_rows = len(hl_data)

            # 정상군 분석
            with st.spinner("정상군 문장 분석 중..."):
                st.session_state.normal_results_df = (
                    analyze_all_sentences(normal_data, label="[정상군] ")
                    if not normal_data.empty else pd.DataFrame()
                )

            # 난청군 분석
            if not hl_data.empty:
                with st.spinner("난청군 문장 분석 중..."):
                    st.session_state.hl_results_df = analyze_all_sentences(hl_data, label="[난청군] ")
                with st.spinner("난청군 피험자별 분석 중..."):
                    st.session_state.hl_subject_results_df = analyze_subjects(hl_data)
            else:
                st.session_state.hl_results_df = None
                st.session_state.hl_subject_results_df = None

            st.session_state.temp_download_data = all_data_df.to_csv(index=False).encode('utf-8-sig')
            st.success("데이터 조회 및 분석 완료.")
        else:
            for key in ['normal_results_df', 'hl_results_df', 'hl_subject_results_df',
                        'data_snr_range', 'temp_download_data', 'total_data_rows',
                        'normal_data_rows', 'hl_data_rows']:
                st.session_state[key] = None
            st.warning("다운로드할 데이터가 없습니다.")

# 데이터 초기화 버튼
cols_reset = st.columns([1, 1, 6])
with cols_reset[0]:
    if st.button("데이터 초기화"):
        for key in ['normal_results_df', 'hl_results_df', 'hl_subject_results_df',
                    'data_snr_range', 'temp_download_data', 'total_data_rows',
                    'normal_data_rows', 'hl_data_rows']:
            st.session_state[key] = None
        st.rerun()

# ============================================================
# 분석 결과 표시
# ============================================================
if st.session_state.temp_download_data is not None:
    # 다운로드 버튼
    st.download_button(
        label="다운로드 준비 완료. 클릭하여 저장 (CSV)",
        data=st.session_state.temp_download_data,
        file_name="all_qsin_scores.csv",
        mime="text/csv",
    )

    st.header("전체 데이터 분석 결과")

    # --- 절대 기준 설정 (탭 공통) ---
    st.subheader("절대 기준 설정 (Absolute Criterion)")
    col_target, col_tol = st.columns(2)
    with col_target:
        target = st.number_input(
            "목표 SNR-50 (dB)", value=TARGET_SNR50, step=0.01, format="%.2f",
            help="K-Quick-SIN 정상 규준값입니다."
        )
    with col_tol:
        tolerance = st.number_input(
            "허용 편차 (+-dB)", value=ABSOLUTE_TOLERANCE, min_value=0.0, step=0.1, format="%.1f",
            help="목표값으로부터의 허용 편차입니다."
        )
    st.session_state['target'] = target
    st.session_state['tolerance'] = tolerance
    st.info(f"**임상 유효 범위**: {target - tolerance:.2f} ~ {target + tolerance:.2f} dB")

    # --- 탭 구성 ---
    tab_normal, tab_hl = st.tabs(["정상군 (문장 검증)", "난청군 (SNR Loss 분석)"])

    # ============================================================
    # 정상군 탭
    # ============================================================
    with tab_normal:
        normal_results_df = st.session_state.normal_results_df

        if normal_results_df is not None and not normal_results_df.empty:
            st.success(f"총 {len(normal_results_df)}개 문장 분석 완료 (정상군 데이터 {st.session_state.normal_data_rows:,}행)")

            # 제외된 문장 정보
            full_df = pd.read_csv(io.StringIO(st.session_state.temp_download_data.decode('utf-8-sig')))
            normal_full = full_df[~_is_hearing_loss(full_df['hearing_loss'])]
            all_ids = set(normal_full['sentence_id'].unique()) if not normal_full.empty else set()
            analyzed_ids = set(normal_results_df['sentence_id'].unique())
            excluded_ids = sorted(list(all_ids - analyzed_ids))
            if excluded_ids:
                st.warning(f"**{len(excluded_ids)}개**의 문장이 분석에서 제외되었습니다. (데이터 부족 또는 정답률 불변)")
                with st.expander("제외된 문장 ID 목록 확인"):
                    st.write(", ".join(map(str, excluded_ids)))

            # --- 1차 분류: 절대 기준 ---
            st.subheader("1차 분류: 절대 기준")

            # 절대 기준만 적용 (2차 없이)
            display_df = reclassify_with_absolute_criterion(normal_results_df, target, tolerance)

            cv_counts = display_df['clinical_validity'].value_counts()
            cols_1st = st.columns(3)
            with cols_1st[0]:
                st.metric("Pass", f"{cv_counts.get('Pass', 0)} 개")
            with cols_1st[1]:
                st.metric("Fail", f"{cv_counts.get('Fail', 0)} 개")
            with cols_1st[2]:
                st.metric("Extrapolated", f"{cv_counts.get('Extrapolated', 0)} 개")

            # --- 2차 분류: 상대 기준 (Pass 내) ---
            st.subheader("2차 분류: 상대 기준 (Pass 문장 내 세분화)")
            pass_df = display_df[display_df['clinical_validity'] == 'Pass']

            ideal_range = None
            acceptable_range = None

            if not pass_df.empty:
                classification_method = st.radio(
                    "등급 분류 기준 선택",
                    ("사분위수(IQR) 기반 (권장)", "평균 +- 표준편차 기반"),
                    index=0, horizontal=True,
                    help="Pass 문장들의 SNR-50 분포를 바탕으로 세부 등급을 나눕니다."
                )

                ranges = calculate_dynamic_ranges(pass_df)

                if classification_method == "사분위수(IQR) 기반 (권장)":
                    ideal_range = ranges['iqr']['ideal']
                    acceptable_range = ranges['iqr']['acceptable']
                    st.caption(
                        f"Ideal: {ideal_range[0]:.2f} ~ {ideal_range[1]:.2f} dB (Q1~Q3) | "
                        f"Acceptable: {acceptable_range[0]:.2f} ~ {acceptable_range[1]:.2f} dB (Q1-0.5*IQR ~ Q3+0.5*IQR)"
                    )
                else:
                    ideal_range = ranges['mean_std']['ideal']
                    acceptable_range = ranges['mean_std']['acceptable']
                    st.caption(
                        f"Ideal: {ideal_range[0]:.2f} ~ {ideal_range[1]:.2f} dB (mean+-0.5s) | "
                        f"Acceptable: {acceptable_range[0]:.2f} ~ {acceptable_range[1]:.2f} dB (mean+-1.0s)"
                    )

                # 절대 + 상대 기준 모두 적용
                display_df = reclassify_with_absolute_criterion(
                    normal_results_df, target, tolerance, ideal_range, acceptable_range
                )

                sg_counts = display_df[display_df['clinical_validity'] == 'Pass']['sub_grade'].value_counts()
                cols_2nd = st.columns(3)
                with cols_2nd[0]:
                    st.metric("Ideal", f"{sg_counts.get('Ideal', 0)} 개")
                with cols_2nd[1]:
                    st.metric("Acceptable", f"{sg_counts.get('Acceptable', 0)} 개")
                with cols_2nd[2]:
                    st.metric("Marginal", f"{sg_counts.get('Marginal', 0)} 개")
            else:
                st.info("Pass 문장이 없어 2차 분류를 수행할 수 없습니다.")

            # --- 전체 통계 ---
            st.subheader("전체 통계")
            total_analyzed = len(display_df)
            pass_count = cv_counts.get('Pass', 0)
            pass_rate = (pass_count / total_analyzed * 100) if total_analyzed > 0 else 0

            cols_stat = st.columns(6)
            with cols_stat[0]:
                st.metric("총 데이터 행 수", f"{st.session_state.normal_data_rows:,}")
            with cols_stat[1]:
                st.metric("분석된 문장 수", total_analyzed)
            with cols_stat[2]:
                st.metric("임상 유효 통과율", f"{pass_rate:.1f}%")
            with cols_stat[3]:
                st.metric("중앙값 SNR-50", f"{display_df['snr_50'].median():.2f} dB")
            with cols_stat[4]:
                st.metric("평균 SNR-50", f"{display_df['snr_50'].mean():.2f} dB")
            with cols_stat[5]:
                st.metric("표준편차", f"{display_df['snr_50'].std():.2f} dB")

            # --- 문장별 결과 테이블 ---
            st.subheader("문장별 분석 결과")
            table_df = display_df.copy()
            table_df['snr_50'] = table_df['snr_50'].round(2)
            table_df['slope'] = table_df['slope'].round(2)
            if 'total_score_sum' in table_df.columns:
                table_df['total_score_sum'] = table_df['total_score_sum'].round(0)
            if 'avg_score' in table_df.columns:
                table_df['avg_score'] = table_df['avg_score'].round(2)

            st.dataframe(
                table_df,
                column_config={
                    "sentence_id": "문장 ID",
                    "full_sentence": st.column_config.TextColumn("문장", width="large"),
                    "snr_50": "SNR-50 (dB)",
                    "slope": "기울기 (%/dB)",
                    "clinical_validity": "1차 (절대)",
                    "sub_grade": "2차 (상대)",
                    "total_score_sum": "총 점수",
                    "avg_score": "평균 점수",
                    "data_points": "데이터 수",
                    "snr_levels": "SNR 레벨 수"
                },
                use_container_width=True
            )

            analysis_csv = display_df.to_csv(index=False).encode('utf-8-sig')
            st.download_button(
                label="분석 결과 다운로드 (CSV)",
                data=analysis_csv,
                file_name="qsin_normal_analysis_results.csv",
                mime="text/csv",
            )

            # --- Psychometric Function Curve ---
            st.subheader("문장별 데이터 시각화 (전체 겹쳐보기)")

            # 그래프용 validity 매핑 (기존 plot 함수와 호환)
            plot_df = display_df.copy()
            def _map_validity(row):
                cv = row.get('clinical_validity', '')
                sg = row.get('sub_grade', '')
                if cv == 'Extrapolated':
                    return 'Extrapolated'
                if cv == 'Fail':
                    return 'Fail'
                if sg == 'Ideal':
                    return 'Ideal'
                if sg == 'Acceptable':
                    return 'Acceptable'
                return 'Warning'
            plot_df['validity'] = plot_df.apply(_map_validity, axis=1)

            available_ids = plot_df['sentence_id'].tolist()
            if available_ids:
                fig = create_combined_psychometric_plot(
                    sentence_ids=available_ids,
                    include_logistic=True,
                    include_mean=False,
                    show_legend=False,
                    precomputed_results=plot_df,
                    snr_range=st.session_state.data_snr_range,
                    ideal_range=ideal_range if ideal_range else (-8.56, -5.57),
                    acceptable_range=acceptable_range if acceptable_range else (-10.05, -4.07)
                )
                if fig:
                    # 목표선 및 허용 범위 밴드 추가
                    fig.add_vrect(
                        x0=target - tolerance, x1=target + tolerance,
                        fillcolor="blue", opacity=0.06, line_width=0,
                        annotation_text="허용 범위", annotation_position="top left"
                    )
                    fig.add_vline(
                        x=target, line_width=2, line_dash="dash", line_color="blue",
                        annotation_text=f"목표: {target} dB", annotation_position="bottom right"
                    )
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("정상군 분석 가능한 문장이 없습니다. 데이터를 확인해주세요.")

    # ============================================================
    # 난청군 탭
    # ============================================================
    with tab_hl:
        hl_results_df = st.session_state.hl_results_df
        hl_subject_df = st.session_state.hl_subject_results_df

        if hl_results_df is None or (isinstance(hl_results_df, pd.DataFrame) and hl_results_df.empty):
            st.info(
                "난청군 데이터가 없습니다.\n\n"
                "난청군 실험(실험 2-2) 데이터가 수집되면 여기에 자동으로 표시됩니다.\n\n"
                "**참고**: `patient_user` 테이블의 `hearing_loss = true`인 피험자의 데이터가 필요합니다."
            )
        else:
            st.success(f"난청군 데이터 {st.session_state.hl_data_rows:,}행, {len(hl_results_df)}개 문장 분석 완료")

            # --- 피험자별 SNR Loss 분석 ---
            st.subheader("피험자별 SNR Loss 분석")

            if hl_subject_df is not None and not hl_subject_df.empty:
                # Extrapolated 피험자 분리
                valid_subjects = hl_subject_df[hl_subject_df.get('validity', 'Analyzed') != 'Extrapolated'] if 'validity' in hl_subject_df.columns else hl_subject_df
                extrapolated_count = len(hl_subject_df) - len(valid_subjects)

                cols_subj = st.columns(4)
                with cols_subj[0]:
                    st.metric("총 피험자 수", len(hl_subject_df))
                with cols_subj[1]:
                    st.metric("유효 피험자 수", len(valid_subjects))
                with cols_subj[2]:
                    snr_loss_mean = valid_subjects['snr_loss'].mean() if not valid_subjects.empty else 0
                    st.metric("SNR Loss 평균", f"{snr_loss_mean:.2f} dB")
                with cols_subj[3]:
                    snr_loss_median = valid_subjects['snr_loss'].median() if not valid_subjects.empty else 0
                    st.metric("SNR Loss 중앙값", f"{snr_loss_median:.2f} dB")

                if extrapolated_count > 0:
                    st.caption(f"* {extrapolated_count}명의 피험자는 Extrapolated(신뢰도 낮은 추정)로 등급 산정에서 제외되었습니다.")

                # 등급별 인원 분포 (유효 피험자만)
                grade_counts = valid_subjects['grade'].value_counts() if not valid_subjects.empty else pd.Series(dtype=int)
                cols_grade = st.columns(4)
                grade_labels = ['정상', '경도', '중도', '고도']
                for i, grade_name in enumerate(grade_labels):
                    with cols_grade[i]:
                        st.metric(f"{grade_name}", f"{grade_counts.get(grade_name, 0)} 명")

                # 피험자 테이블
                subj_table = hl_subject_df.copy()
                subj_table['snr_50'] = subj_table['snr_50'].round(2)
                subj_table['snr_loss'] = subj_table['snr_loss'].round(2)
                subj_table['slope'] = subj_table['slope'].round(2)

                st.dataframe(
                    subj_table,
                    column_config={
                        "patient_user_id": "피험자 ID",
                        "snr_50": "SNR-50 (dB)",
                        "snr_loss": "SNR Loss (dB)",
                        "grade": "등급",
                        "validity": "신뢰도",
                        "slope": "기울기 (%/dB)",
                        "data_points": "데이터 수",
                        "sentences_tested": "테스트 문장 수"
                    },
                    use_container_width=True
                )

                # SNR Loss 분포 히스토그램
                st.subheader("SNR Loss 분포")
                import plotly.graph_objects as go

                fig_hist = go.Figure()
                fig_hist.add_trace(go.Histogram(
                    x=valid_subjects['snr_loss'],
                    nbinsx=20,
                    marker_color='steelblue',
                    name='SNR Loss'
                ))

                # 등급 구간 밴드 추가
                grade_colors = ['rgba(0,200,0,0.1)', 'rgba(255,200,0,0.1)',
                                'rgba(255,130,0,0.1)', 'rgba(255,0,0,0.1)']
                for i, (grade_name, low, high) in enumerate(SNR_LOSS_GRADES):
                    max_loss = valid_subjects['snr_loss'].max() if not valid_subjects.empty else 20
                    display_high = min(high, max_loss + 5) if high != float('inf') else max_loss + 5
                    fig_hist.add_vrect(
                        x0=low, x1=display_high,
                        fillcolor=grade_colors[i], line_width=0,
                        annotation_text=grade_name, annotation_position="top left"
                    )

                fig_hist.update_layout(
                    title="피험자별 SNR Loss 분포",
                    xaxis_title="SNR Loss (dB)",
                    yaxis_title="피험자 수",
                    bargap=0.1
                )
                st.plotly_chart(fig_hist, use_container_width=True)
            else:
                st.warning("피험자별 분석 결과가 없습니다. 피험자 데이터가 충분한지 확인해주세요.")

            # --- 문장 품질 검증 ---
            st.subheader("문장 품질 검증 (난청군)")

            # 전역 문장 간 SD
            sentence_sd = hl_results_df['snr_50'].std()
            low_slope_count = len(hl_results_df[hl_results_df['slope'] < SLOPE_THRESHOLD])

            # 문장별 편차 계산 (중앙값 기준)
            hl_median_snr50 = hl_results_df['snr_50'].median()
            hl_results_df = hl_results_df.copy()
            hl_results_df['deviation'] = (hl_results_df['snr_50'] - hl_median_snr50).abs()
            sd_flag_fail_count = len(hl_results_df[hl_results_df['deviation'] >= SENTENCE_SD_THRESHOLD])

            cols_quality = st.columns(4)
            with cols_quality[0]:
                sd_delta = f"{SENTENCE_SD_THRESHOLD - sentence_sd:.2f}"
                st.metric(
                    "문장 간 SD",
                    f"{sentence_sd:.2f} dB",
                    delta=sd_delta if sentence_sd < SENTENCE_SD_THRESHOLD else f"-{sentence_sd - SENTENCE_SD_THRESHOLD:.2f}",
                    delta_color="normal" if sentence_sd < SENTENCE_SD_THRESHOLD else "inverse"
                )
            with cols_quality[1]:
                st.metric("편차 >= 3.0 dB 문장 수", f"{sd_flag_fail_count} 개")
            with cols_quality[2]:
                st.metric("Slope < 3%/dB 문장 수", f"{low_slope_count} 개")
            with cols_quality[3]:
                both_pass = len(hl_results_df[
                    (hl_results_df['deviation'] < SENTENCE_SD_THRESHOLD) &
                    (hl_results_df['slope'] >= SLOPE_THRESHOLD)
                ])
                st.metric("품질 검증 통과 문장 수", f"{both_pass} 개")

            if sentence_sd >= SENTENCE_SD_THRESHOLD:
                st.warning(
                    f"문장 간 SD ({sentence_sd:.2f} dB)가 기준값 ({SENTENCE_SD_THRESHOLD} dB) 이상입니다. "
                    f"일부 문장의 난이도가 난청군에서 불균등하게 작동할 수 있습니다."
                )

            # 난청군 문장별 결과 테이블
            hl_display = hl_results_df.copy()
            hl_display['snr_loss'] = (hl_display['snr_50'] - target).round(2)
            hl_display['snr_50'] = hl_display['snr_50'].round(2)
            hl_display['slope'] = hl_display['slope'].round(2)
            hl_display['deviation'] = hl_display['deviation'].round(2)
            hl_display['slope_flag'] = hl_display['slope'].apply(
                lambda s: 'Pass' if s >= SLOPE_THRESHOLD else 'Fail'
            )
            hl_display['sd_flag'] = hl_display['deviation'].apply(
                lambda d: 'Pass' if d < SENTENCE_SD_THRESHOLD else 'Fail'
            )

            # 정상군 Pass 문장 목록과 교차
            normal_display = st.session_state.normal_results_df
            normal_pass_ids = set()
            if normal_display is not None and not normal_display.empty:
                normal_classified = reclassify_with_absolute_criterion(normal_display, target, tolerance)
                normal_pass_ids = set(
                    normal_classified[normal_classified['clinical_validity'] == 'Pass']['sentence_id']
                )

            hl_display['normal_pass'] = hl_display['sentence_id'].apply(
                lambda sid: 'Pass' if sid in normal_pass_ids else 'Fail'
            )
            hl_display['final_select'] = hl_display.apply(
                lambda row: 'Pass' if (
                    row['normal_pass'] == 'Pass' and
                    row['slope_flag'] == 'Pass' and
                    row['sd_flag'] == 'Pass'
                ) else 'Fail',
                axis=1
            )

            st.dataframe(
                hl_display,
                column_config={
                    "sentence_id": "문장 ID",
                    "full_sentence": st.column_config.TextColumn("문장", width="large"),
                    "snr_50": "SNR-50 (dB)",
                    "snr_loss": "SNR Loss (dB)",
                    "slope": "기울기 (%/dB)",
                    "deviation": "편차 (dB)",
                    "sd_flag": "편차 검증",
                    "slope_flag": "기울기 검증",
                    "normal_pass": "정상군 Pass",
                    "final_select": "최종 선별",
                    "data_points": "데이터 수",
                    "snr_levels": "SNR 레벨 수"
                },
                use_container_width=True
            )

            # 최종 선별 요약
            final_pass_count = len(hl_display[hl_display['final_select'] == 'Pass'])
            final_fail_count = len(hl_display[hl_display['final_select'] == 'Fail'])

            st.subheader("최종 문장 선별 요약")
            cols_final = st.columns(3)
            with cols_final[0]:
                st.metric("최종 선별 문장", f"{final_pass_count} 개")
            with cols_final[1]:
                st.metric("제외 문장", f"{final_fail_count} 개")
            with cols_final[2]:
                select_rate = (final_pass_count / len(hl_display) * 100) if len(hl_display) > 0 else 0
                st.metric("선별률", f"{select_rate:.1f}%")

            with st.expander("선별 기준 안내"):
                st.markdown(f"""
                **최종 선별 조건** (모두 충족해야 Pass):
                1. 정상군 절대 기준 통과: SNR-50이 목표값({target} dB) +-{tolerance} dB 이내
                2. 난청군 기울기 하한선: slope >= {SLOPE_THRESHOLD} %/dB (변별력 확보)
                3. 난청군 편차 검증: 중앙값 기준 편차 < {SENTENCE_SD_THRESHOLD} dB (난이도 균등성)

                **보조 지표**:
                - 문장 간 SD: 난청군 내 전체 문장 SNR-50의 표준편차 (목표: < {SENTENCE_SD_THRESHOLD} dB)
                """)

            # 난청군 분석 결과 다운로드
            hl_csv = hl_display.to_csv(index=False).encode('utf-8-sig')
            st.download_button(
                label="난청군 분석 결과 다운로드 (CSV)",
                data=hl_csv,
                file_name="qsin_hl_analysis_results.csv",
                mime="text/csv",
            )

st.divider()

# ============================================================
# 섹션 1: 개별 문장 분석
# ============================================================
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

if st.button(f"문장 {sentence_id_to_analyze}번 데이터 분석 실행"):
    try:
        st.header("2. 데이터 처리 과정 확인")
        with st.spinner(f"DB에서 문장 {sentence_id_to_analyze}번의 데이터를 로드하고 전처리 중입니다..."):
            processed_data = get_all_sentence_data(supabase, use_dummy_data, sentence_id_to_analyze)

        if processed_data.empty:
            st.error(f"문장 {sentence_id_to_analyze}번에 대한 분석을 진행할 수 없습니다. 데이터가 없거나 유효한 SNR 레벨이 연결되지 않았습니다.")
        else:
            st.info("선택한 문장에 대해 데이터베이스에서 조회하고 정답률을 계산한 결과입니다.")

            full_sentence_text = processed_data['full_sentence'].iloc[0]
            st.markdown(f"**분석 대상 문장: \"{full_sentence_text}\"**")

            ind_display_df = processed_data.drop(columns=['session_id', 'full_sentence', 'sentence_id', 'user_id'], errors='ignore')
            st.dataframe(ind_display_df)

            csv_data = processed_data.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="이 문장 데이터 다운로드 (CSV)",
                data=csv_data,
                file_name=f"sentence_{sentence_id_to_analyze}_data.csv",
                mime="text/csv",
            )

            st.header("3. 분석 결과")

            # UI에서 설정한 값 사용 (설정 전이면 모듈 기본값)
            ind_target = st.session_state.get('target', TARGET_SNR50)
            ind_tolerance = st.session_state.get('tolerance', ABSOLUTE_TOLERANCE)

            with st.spinner("로지스틱 회귀 모델을 학습하고 SNR-50을 추정합니다..."):
                result = estimate_snr50_for_sentence(processed_data)

                # 절대 기준으로 등급 분류
                if result['status'] == 'Success' and result.get('validity') != 'Extrapolated':
                    snr = result['snr_50']
                    if abs(snr - ind_target) <= ind_tolerance:
                        result['validity'] = 'Pass'
                    else:
                        result['validity'] = 'Fail'

            status = result.get('status')
            if status == 'Success':
                snr50_val = result.get('snr_50')
                slope_val = result.get('slope')
                display_analysis_metrics(snr50_val, slope_val)

                # 절대 기준 판정 표시
                validity = result.get('validity', '')
                if validity == 'Pass':
                    st.success(f"절대 기준 Pass: SNR-50 ({snr50_val:.2f} dB)이 목표값 ({ind_target} dB) +-{ind_tolerance} dB 이내")
                elif validity == 'Fail':
                    st.error(f"절대 기준 Fail: SNR-50 ({snr50_val:.2f} dB)이 목표값 ({ind_target} dB) +-{ind_tolerance} dB 범위 밖")
                elif validity == 'Extrapolated':
                    st.warning("Extrapolated: 추정된 SNR-50이 테스트 범위를 크게 벗어남 (신뢰도 낮음)")
            else:
                st.error(f"분석 실패: **{status}**")
                st.warning("데이터 분포를 확인해주세요. 최소 3개 이상의 다양한 SNR 레벨에 대한 데이터가 필요합니다.")

            # --- 시각화 ---
            st.header("4. Psychometric Function Curve")
            fig = create_psychometric_plot(processed_data, result, sentence_id_to_analyze)
            if fig:
                # 목표선 및 허용 범위 밴드 추가
                fig.add_vrect(
                    x0=ind_target - ind_tolerance,
                    x1=ind_target + ind_tolerance,
                    fillcolor="blue", opacity=0.06, line_width=0
                )
                fig.add_vline(
                    x=ind_target, line_width=2, line_dash="dash", line_color="blue",
                    annotation_text=f"목표: {ind_target} dB", annotation_position="bottom right"
                )
                st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"데이터를 처리하는 중 오류가 발생했습니다: {e}")
        st.info("간헐적인 네트워크 오류일 수 있습니다. 아래 버튼을 눌러 캐시를 초기화하고 다시 시도해 보세요.")
        if st.button("캐시 지우고 재시도"):
            st.cache_data.clear()
            st.rerun()
