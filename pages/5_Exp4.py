"""실험 4 — 리스트 최종 검증(13개 → 최종 12개) + 검사-재검사 페이지.

2차(재검사)가 진행 중이어도 동작하며, 수집이 끝나면 '잠정' 표시가 자동으로 '최종'으로 바뀐다.
계산은 modules/exp4_utils.py (판정 기준 고정).
"""

import io
import zipfile
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from modules import exp4_utils as U
from modules.db_utils import init_supabase_client

st.set_page_config(page_title="실험 4 최종 검증", layout="wide")

C_R1 = "#1a73c8"     # 1차 / 기본
C_R2 = "#c0502a"     # 2차 / 강조(색맹 검증 통과 조합)
C_GRID = "rgba(128,128,128,0.25)"


def p_fmt(p):
    return "< .001" if p < .001 else f"{p:.3f}".lstrip("0")


# ============================================================
# 데이터
# ============================================================
st.title("실험 4 — 리스트 최종 검증")
st.markdown(
    "실험 3에서 고른 **13개 리스트**(3, 4, 5, 6, 7, 10, 11, 12, 13, 14, 15, 16, 18)를 정상 청력 성인 26명에게 검사해 "
    "① 리스트끼리 난이도가 같은지, ② 1~2주 뒤 다시 검사해도 결과가 같은지(검사-재검사), "
    "③ 정상인의 기준값(정상 기준)을 확인합니다. **최종 리스트는 12개(리스트 4 제외)로 확정**됐습니다. "
    "결과는 DB 데이터로 매번 다시 계산합니다."
)

supabase = init_supabase_client()
if not supabase:
    st.stop()

if st.button("🔄 데이터 새로고침", help="DB에서 다시 읽어옵니다(평소에는 10분간 저장된 결과를 씁니다)."):
    U.load_exp4.clear()

try:
    df = U.load_exp4(supabase)
except Exception as e:
    st.error(f"데이터를 불러오지 못했습니다: {e}")
    st.stop()
if df.empty:
    st.warning("실험 4 데이터가 없습니다.")
    st.stop()

cells = U.build_cells(df)
status = U.collection_status(cells)
stg = U.stage(status)
m1 = U.matrix(cells, 1)

if stg["final"]:
    drop_msg = f", 2차 미참여 {stg['n_drop']}명" if stg["n_drop"] else ""
    st.success(f"✅ **최종 결과** — 수집이 끝났습니다. 1차 {stg['n_r1']}명, 2차 {stg['n_r2']}명{drop_msg}.")
else:
    st.warning(
        f"⏳ **잠정 결과** — 2차(재검사) 진행 중입니다. 1차 {stg['n_r1']}명 완료, "
        f"2차 {stg['n_r2']} / {stg['n_r1']}명 완료 · {stg['n_wait']}명 대기. "
        "모든 참가자가 2차를 마치거나 2차 기한이 지나면 자동으로 '최종'으로 바뀝니다."
    )

tabs = st.tabs(["① 수집 현황", "② 데이터 점검", "③ 리스트 동등성", "④ 순서 효과", "⑤ 정상 기준",
                "⑥ 검사-재검사", "⑦ 최종 리스트", "📦 다운로드"])

# ============================================================
# ① 수집 현황
# ============================================================
with tabs[0]:
    c = st.columns(4)
    c[0].metric("1차 완료", f"{stg['n_r1']} / {U.TARGET_PARTICIPANTS}명")
    c[1].metric("2차 완료", f"{stg['n_r2']} / {stg['n_r1']}명")
    c[2].metric("2차 대기", f"{stg['n_wait']}명")
    c[3].metric("2차 미참여(기한 초과)", f"{stg['n_drop']}명")

    soon = status[status["상태"].str.startswith("기한 임박")]
    if len(soon):
        st.error("⚠️ 2차 기한이 3일 이내인 참가자: " +
                 ", ".join(f"{r['참가자']}번({r['2차 기한']:%m-%d})" for _, r in soon.iterrows()))

    st.dataframe(
        status, hide_index=True, width="stretch",
        column_config={
            "1차 날짜": st.column_config.DateColumn(format="YYYY-MM-DD"),
            "2차 기한": st.column_config.DateColumn(format="YYYY-MM-DD", help="1차 후 14일. 이 날이 지나면 2차를 할 수 없습니다."),
            "2차 날짜": st.column_config.DateColumn(format="YYYY-MM-DD"),
            "순서표 행": st.column_config.NumberColumn(help="Williams 26행 순서표의 몇 번째 행인지 = (번호−1) mod 26 + 1"),
        },
    )
    st.caption(
        "· 청력 기준(양쪽 250~8000 Hz 모두 20 dB HL 이하) 미달자는 DB에서 'exp4_excluded'로 옮겨 이 표에 나오지 않습니다.  \n"
        f"· 참가자는 {U.TARGET_PARTICIPANTS}명(Williams 순서표 26행 1주기)입니다.  \n"
        "· 2차는 1차와 같은 참가자 번호·같은 순서로 진행합니다."
    )

# ============================================================
# ② 데이터 점검
# ============================================================
with tabs[1]:
    st.markdown("데이터가 제대로 저장됐는지 자동으로 확인합니다. ⚠️ 항목은 내용을 확인해 주세요.")
    for chk in U.data_checks(cells, status):
        icon = "✅" if chk["ok"] else "⚠️"
        with st.expander(f"{icon} {chk['항목']} — {chk['결과']}", expanded=not chk["ok"]):
            if not chk["ok"]:
                st.write("해당: " + chk["해당"])
            if chk["설명"]:
                st.caption(chk["설명"])
    st.caption("점수 수정·참가자 제외는 페이지에서 할 수 없습니다. 인수인계 문서의 절차를 따라 주세요.")

# ============================================================
# ③ 리스트 동등성
# ============================================================
with tabs[2]:
    st.markdown(
        "리스트들의 난이도가 같은지 **1차 데이터**로 확인합니다. 점수는 **SNR-50**(문장 키워드의 절반을 "
        "알아듣는 신호대잡음비)으로 바꿔 비교하며, **낮을수록 쉬운 리스트**입니다."
    )
    set_name = st.radio("리스트 세트", list(U.LIST_SETS), horizontal=True,
                        help="최종은 12개(리스트 4 제외)입니다. 13개는 계획 원안, 11개는 민감도 분석용입니다.")
    mm = U.matrix(cells, 1, U.LIST_SETS[set_name])
    if len(mm) < 3:
        st.info("1차 데이터가 3명 이상 모이면 표시됩니다.")
    else:
        e = U.equivalence(mm)
        verdict = "✅ 세 기준 모두 통과" if e["passed"] else "❌ 통과하지 못한 기준이 있습니다"
        st.subheader(f"판정 ({e['n']}명): {verdict}")
        cv = e["crit"].copy()
        cv["통과"] = cv["통과"].map({True: "✅", False: "❌"})
        st.dataframe(cv, hide_index=True, width="stretch")

        ls = e["lists"].sort_values("mean")
        fig = go.Figure(go.Scatter(
            x=ls["mean"], y=[f"리스트 {l}" for l in ls["list_no"]], mode="markers",
            marker=dict(color=C_R1, size=11), error_x=dict(type="data", array=ls["ci"], color=C_R1, thickness=2, width=0),
            customdata=np.stack([ls["dev"], ls["ci"], ls["sd"]], axis=-1),
            hovertemplate="<b>%{y}</b><br>평균 %{x:.2f} dB (95% CI ±%{customdata[1]:.2f})"
                          "<br>전체 평균과 차이 %{customdata[0]:+.2f} dB<br>참가자 간 SD %{customdata[2]:.2f}<extra></extra>",
        ))
        fig.add_vline(x=e["grand"], line=dict(color="gray", dash="dot", width=1.5),
                      annotation_text=f"전체 평균 {e['grand']:.2f} dB", annotation_position="top")
        fig.update_layout(
            title=f"리스트별 평균 SNR-50 (1차, {e['n']}명, 막대는 95% 신뢰구간)", height=500, showlegend=False,
            xaxis=dict(title="SNR-50 (dB) — 왼쪽일수록 쉬움", gridcolor=C_GRID, zeroline=False),
            yaxis=dict(categoryorder="array", categoryarray=[f"리스트 {l}" for l in ls["list_no"]], autorange="reversed"),
            margin=dict(l=10, r=10, t=70, b=40),
        )
        st.plotly_chart(fig, use_container_width=True)

        if U.LIST_SETS[set_name] == U.LISTS:
            st.subheader("13개에서 기준 미달 → 단계적 제외")
            steps = U.stepwise_candidates(mm)
            if len(steps) == 1 and steps[0]["passed"]:
                st.write("13개 모두 기준을 통과해 제외 후보가 없습니다.")
            else:
                st.dataframe(pd.DataFrame([{
                    "단계": i + 1, "리스트 수": s["n_lists"],
                    "리스트 간 SD (dB)": round(s["between_sd"], 2), "리스트 효과 p": p_fmt(s["p_gg"]),
                    "판정": "✅ 통과" if s["passed"] else "❌ 미달",
                    "다음에 뺄 리스트 (평균과 차이)": f"리스트 {s['drop']} ({s['drop_dev']:+.2f} dB)" if "drop" in s else "—",
                } for i, s in enumerate(steps)]), hide_index=True, width="stretch")
                st.info(
                    "평균에서 가장 먼 리스트를 하나씩 빼며 기준을 다시 확인한 결과입니다(최소 10개 유지). "
                    "리스트 4를 빼자 세 기준을 모두 통과해 멈췄습니다. 참고로 리스트 4는 다른 리스트와 하나씩 "
                    "비교했을 때는 유의하지 않아(Holm p = .14) 경계선 사례입니다."
                )

        st.subheader("리스트 세트별 비교 (민감도 분석)")
        sens_tbl = U.final_decision(cells)["sensitivity"].copy()
        sens_tbl["리스트 효과 p"] = sens_tbl["리스트 효과 p"].map(p_fmt)
        st.dataframe(sens_tbl, hide_index=True, width="stretch")
        st.caption("12개에서 이미 세 기준을 통과하므로 더 빼지 않습니다(사전 계획). 리스트 11(가장 어려운 리스트)까지 "
                   "빼도 결론과 정상 기준은 거의 같습니다.")

# ============================================================
# ④ 순서 효과
# ============================================================
with tabs[3]:
    pe = U.position_effect(cells)
    fig = go.Figure()
    for r, color, name in [(1, C_R1, "1차"), (2, C_R2, "2차")]:
        d = pe[pe["test_round"] == r] if len(pe) else pe
        if len(d):
            n_r = int(d["count"].max())
            fig.add_trace(go.Scatter(
                x=d["presentation_order"], y=d["mean"], mode="lines+markers", name=f"{name} ({n_r}명)",
                line=dict(color=color, width=2, dash="solid" if r == 1 else "dash"), marker=dict(size=8),
                hovertemplate=f"{name} · %{{x}}번째 리스트<br>평균 SNR-50 %{{y:.2f}} dB<extra></extra>",
            ))
    fig.update_layout(
        title="몇 번째로 들은 리스트인지에 따른 평균 SNR-50", height=400,
        xaxis=dict(title="제시 순서", dtick=1, gridcolor=C_GRID), yaxis=dict(title="SNR-50 (dB)", gridcolor=C_GRID),
        legend=dict(orientation="h", y=1.1, x=0), margin=dict(l=10, r=10, t=70, b=40),
    )
    st.plotly_chart(fig, use_container_width=True)
    if len(pe):
        p1 = pe[pe["test_round"] == 1].set_index("presentation_order")["mean"]
        st.markdown(
            f"1차에서 첫 번째 리스트는 평균 **{p1.iloc[0]:.1f} dB**, 마지막(13번째)은 **{p1.iloc[-1]:.1f} dB**로, "
            "뒤로 갈수록 점수가 좋아졌습니다(학습 효과). 모든 리스트가 각 순서에 같은 횟수로 오도록 설계(Williams)해서 "
            "리스트 비교에는 영향이 없지만, **정상 기준을 정할 때는 이 효과를 고려합니다**(⑤ 정상 기준 탭)."
        )
    if stg["n_r2"] < 5:
        st.caption("2차 곡선은 참가자가 적을 때 들쭉날쭉합니다. 2차 참가자가 늘어나면 안정됩니다.")

# ============================================================
# ⑤ 정상 기준
# ============================================================
with tabs[4]:
    nv = U.norm_value(cells)
    st.markdown(
        "**정상 기준**은 정상 청력인 사람이 이 검사에서 보통 받는 SNR-50입니다. 환자 결과는 "
        "**SNR 손실 = 환자 SNR-50 − 정상 기준**으로 해석합니다(예: 정상 기준 −3.3 dB, 환자 +4.7 dB → SNR 손실 8 dB)."
    )
    if nv is None:
        st.info("1차 데이터가 모이면 표시됩니다.")
    else:
        c = st.columns(3)
        c[0].metric("정상 기준", f"{nv['value_rounded']:.1f} dB", help=f"소수 첫째 자리 반올림 (계산값 {nv['value']:.2f} dB)")
        c[1].metric("95% 신뢰구간", f"{nv['ci'][0]:.1f} ~ {nv['ci'][1]:.1f} dB")
        c[2].metric("참가자 수", f"{nv['n']}명", help="1차에서 리스트를 모두 마친 참가자")

        st.subheader("어떤 값을 썼나 (사전에 정한 규칙)")
        st.dataframe(pd.DataFrame([
            ("A", "1차 제시 순서 1~2번째 리스트 평균 (최종 12개 중)", f"{nv['A']:.2f}", f"{nv['A_sd']:.2f}",
             "임상과 같은 조건(연습 2개 후 처음 듣는 리스트)"),
            ("B", "1차 최종 12개 리스트 전체 평균", f"{nv['B']:.2f}", f"{nv['B_sd']:.2f}", "리스트를 많이 들어 학습 효과가 섞임"),
        ], columns=["", "계산", "평균 (dB)", "참가자 간 SD", "의미"]), hide_index=True, width="stretch")
        st.markdown(
            f"규칙: **|A − B| < {U.NORM_RULE_DIFF} dB이면 B, 아니면 A.** "
            f"현재 |A − B| = {abs(nv['diff']):.2f} dB → **{nv['use']} 사용**. "
            "0.5 dB는 키워드 1개(0.6 dB)보다 작은 차이로, 임상적으로 의미 없는 수준입니다. "
            "2차 데이터는 학습 효과가 섞여 쓰지 않습니다."
        )
        st.info(
            f"**참고 (계획서 외): 정상 상한 {nv['upper']:.1f} dB** = 정상 기준 + 1.645 × SD({nv['sd']:.2f}). "
            "정상인의 약 95%가 이 값보다 좋은 점수를 받는다는 뜻입니다. 정상/경도 등 등급 기준은 이 실험이 아니라 "
            "난청군 후속 연구에서 정하므로, 이 값은 참고로만 보세요."
        )
        vals = nv["A_p"] if nv["use"] == "A" else nv["B_p"]
        fig = go.Figure(go.Histogram(x=vals, xbins=dict(size=0.6), marker=dict(color=C_R1, line=dict(color="white", width=2)),
                                     hovertemplate="%{x} dB: %{y}명<extra></extra>"))
        fig.add_vline(x=nv["value"], line=dict(color="gray", width=2), annotation_text="정상 기준", annotation_position="top")
        fig.add_vline(x=nv["upper"], line=dict(color=C_R2, dash="dash", width=1.5),
                      annotation_text="정상 상한(참고)", annotation_position="top right")
        fig.update_layout(title=f"참가자별 값({nv['use']}) 분포", height=340, showlegend=False, bargap=0.05,
                          xaxis=dict(title="SNR-50 (dB)", gridcolor=C_GRID), yaxis=dict(title="참가자 수", gridcolor=C_GRID),
                          margin=dict(l=10, r=10, t=60, b=40))
        st.plotly_chart(fig, use_container_width=True)

# ============================================================
# ⑥ 검사-재검사
# ============================================================
rt = U.retest(cells)
with tabs[5]:
    st.markdown(
        "같은 사람이 1~2주 뒤 **같은 순서로 다시 검사했을 때 결과가 얼마나 같은지** 봅니다. "
        "두 가지 수준으로 계산합니다.  \n"
        "· **12개 리스트 평균**: 최종 12개를 모두 했을 때의 안정성  \n"
        "· **리스트 1개**: 같은 리스트의 1차 vs 2차. 임상에서는 리스트를 1~2개만 하므로 실제 검사 한 번의 안정성에 가깝습니다."
    )
    if not rt["ready"]:
        st.info(f"현재 1차·2차를 모두 마친 참가자는 **{rt['n']}명**입니다. "
                f"{U.MIN_N_RETEST}명 이상이 되면 자동으로 계산됩니다.")
    else:
        ml, sg = rt["mean_level"], rt["single"]
        if not stg["final"]:
            st.caption(f"⏳ 잠정 — 현재 {rt['n']}명 기준입니다. 2차가 끝나면 값이 바뀔 수 있습니다.")
        st.subheader(f"요약 ({rt['n']}명)")
        st.dataframe(pd.DataFrame([
            ("ICC(2,1) [95% CI]", f"{ml['icc']:.2f} [{ml['lo']:.2f}, {ml['hi']:.2f}] · {U.icc_label(ml['icc'])}",
             f"평균 {sg['icc_mean']:.2f} (범위 {sg['icc_min']:.2f}~{sg['icc_max']:.2f}) · {U.icc_label(sg['icc_mean'])}",
             "1에 가까울수록 1차와 2차가 일치. .5/.75/.9 = 보통/좋음/매우 좋음 (Koo & Li)"),
            ("SEM (측정의 표준오차)", f"{ml['sem']:.2f} dB", f"{sg['sem']:.2f} dB", "검사할 때마다 생기는 흔들림의 크기"),
            ("CD95 (의미 있는 변화)", f"{ml['cd95']:.2f} dB", f"{sg['cd95']:.2f} dB",
             "다시 검사했을 때 이보다 크게 달라져야 실제 변화로 봄 (= 1.96 × √2 × SEM)"),
            ("2차 − 1차 평균 차이", f"{ml['diff_mean']:+.2f} dB (p = {p_fmt(ml['p'])})", f"{sg['diff_mean']:+.2f} dB",
             "음수 = 2차가 더 좋아짐(학습 효과). p는 대응표본 t검정"),
        ], columns=["지표", "12개 리스트 평균", "리스트 1개", "의미"]), hide_index=True, width="stretch")

        pr = rt["pairs"]
        c1, c2 = st.columns(2)
        with c1:
            lo_, hi_ = float(min(pr.r1.min(), pr.r2.min())) - 0.5, float(max(pr.r1.max(), pr.r2.max())) + 0.5
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=[lo_, hi_], y=[lo_, hi_], mode="lines", line=dict(color="gray", dash="dot", width=1),
                                     hoverinfo="skip", name="1차 = 2차"))
            fig.add_trace(go.Scatter(x=pr.r1, y=pr.r2, mode="markers", marker=dict(color=C_R1, size=10), name="참가자",
                                     customdata=pr.participant_no,
                                     hovertemplate="%{customdata}번<br>1차 %{x:.2f} dB<br>2차 %{y:.2f} dB<extra></extra>"))
            fig.update_layout(title="참가자별 12개 평균: 1차 vs 2차", height=400, showlegend=False,
                              xaxis=dict(title="1차 SNR-50 (dB)", range=[lo_, hi_], gridcolor=C_GRID),
                              yaxis=dict(title="2차 SNR-50 (dB)", range=[lo_, hi_], gridcolor=C_GRID, scaleanchor="x"),
                              margin=dict(l=10, r=10, t=60, b=40))
            st.plotly_chart(fig, use_container_width=True)
            st.caption("점선 위에 있으면 1차와 2차가 같습니다. 점선 아래 = 2차가 더 좋아짐.")
        with c2:
            avg, dif = (pr.r1 + pr.r2) / 2, pr.r2 - pr.r1
            fig = go.Figure(go.Scatter(x=avg, y=dif, mode="markers", marker=dict(color=C_R1, size=10),
                                       customdata=pr.participant_no,
                                       hovertemplate="%{customdata}번<br>평균 %{x:.2f} dB<br>차이 %{y:+.2f} dB<extra></extra>"))
            fig.add_hline(y=ml["diff_mean"], line=dict(color="gray", width=1.5),
                          annotation_text=f"평균 차이 {ml['diff_mean']:+.2f}", annotation_position="top left")
            for v_ in ml["loa"]:
                fig.add_hline(y=v_, line=dict(color=C_R2, dash="dash", width=1.2),
                              annotation_text=f"{v_:+.2f}", annotation_position="top right")
            fig.update_layout(title="Bland-Altman (1차·2차 일치 범위)", height=400, showlegend=False,
                              xaxis=dict(title="1차·2차 평균 (dB)", gridcolor=C_GRID),
                              yaxis=dict(title="2차 − 1차 (dB)", gridcolor=C_GRID, zeroline=True, zerolinecolor="gray"),
                              margin=dict(l=10, r=10, t=60, b=40))
            st.plotly_chart(fig, use_container_width=True)
            st.caption("주황 점선 = 95% 일치 한계(평균 차이 ± 1.96 SD). 참가자 대부분이 이 안에 있어야 합니다.")

        st.subheader("리스트별 결과 (리스트 1개 수준)")
        pl = rt["per_list"].copy()
        pl["ICC [95% CI]"] = pl.apply(lambda r: f"{r.icc:.2f} [{r.lo:.2f}, {r.hi:.2f}]", axis=1)
        pl["안정성"] = np.where(pl["unstable"], "⚠️ 들쭉날쭉", "—")
        pl = pl.rename(columns={"list_no": "리스트", "r1_mean": "1차 평균", "r2_mean": "2차 평균",
                                "diff_mean": "평균 차이", "diff_sd": "차이 SD", "sem": "SEM"})
        st.dataframe(pl[["리스트", "1차 평균", "2차 평균", "평균 차이", "차이 SD", "SEM", "ICC [95% CI]", "안정성"]],
                     hide_index=True, width="stretch",
                     column_config={c: st.column_config.NumberColumn(format="%.2f")
                                    for c in ["1차 평균", "2차 평균", "평균 차이", "차이 SD", "SEM"]})
        st.caption(f"'들쭉날쭉' = 차이 SD가 리스트 중앙값({rt['stability_median']:.2f} dB)의 1.5배를 넘는 리스트. "
                   "제외하지 않고 표시만 합니다(리스트 구성에는 2차를 쓰지 않음).")

# ============================================================
# ⑦ 최종 리스트
# ============================================================
fd = U.final_decision(cells)
with tabs[6]:
    st.success(f"✅ **최종 리스트 12개 (확정)**: {', '.join(map(str, fd['final_lists']))}  \n"
               f"제외: 리스트 {', '.join(map(str, U.DROPPED_LISTS))} · 1차 {len(m1)}명 데이터로 판정 (2026-11-27)")
    if len(m1) >= 3 and not fd["matches"]:
        st.error(f"⚠️ DB 데이터로 다시 계산한 결과({fd['computed']})가 확정 결과와 다릅니다. "
                 "1차 데이터가 수정되었을 수 있습니다. 담당자에게 확인을 요청하세요.")
    elif len(m1) >= 3:
        st.caption("✔ DB 데이터로 다시 계산한 결과가 확정 결과와 같습니다.")

    st.subheader("판정 규칙 (사전 분석 계획)")
    for line in U.FINAL_RULE:
        st.markdown(f"- {line}")

    st.subheader("판정 과정")
    st.dataframe(pd.DataFrame([{
        "단계": i + 1, "리스트 수": s_["n_lists"], "리스트 간 SD (dB)": round(s_["between_sd"], 2),
        "리스트 효과 p": p_fmt(s_["p_gg"]), "판정": "✅ 세 기준 통과" if s_["passed"] else "❌ 미달",
        "제외": f"리스트 {s_['drop']} (평균과 {s_['drop_dev']:+.2f} dB)" if "drop" in s_ else "—",
    } for i, s_ in enumerate(fd["steps1"])]), hide_index=True, width="stretch")

    st.subheader("2차(재검사)에서 들쭉날쭉한 리스트 (표시만)")
    if not rt["ready"]:
        st.write(f"2차 {rt['n']}명 — {U.MIN_N_RETEST}명 이상이면 표시됩니다.")
    else:
        st.write(", ".join(f"리스트 {l}" for l in fd["unstable"]) or "없음")

    st.subheader("보고서에 밝힐 점")
    st.markdown(
        "- 계획 원안 13개는 리스트 효과가 유의해(p = .032) 사전 규칙대로 리스트 4를 제외했다.\n"
        "- 리스트 4는 개별 비교(Holm 보정)에서는 유의하지 않은 경계선 사례였다.\n"
        "- 민감도 분석: 13개·11개(4·11 제외)로 계산해도 정상 기준은 −3.3 dB로 같다(③ 리스트 동등성 탭).\n"
        "- 정상 기준과 신뢰도는 최종 12개로 계산했다(계획서의 '13리스트'를 최종 세트로 적용)."
    )

# ============================================================
# 다운로드
# ============================================================
CODEBOOK = pd.DataFrame([
    ("exp4_cells.csv", "participant_no", "참가자 번호"),
    ("exp4_cells.csv", "test_round", "1 = 1차, 2 = 2차(재검사)"),
    ("exp4_cells.csv", "is_practice", "연습 리스트(8, 9) 여부. 분석에는 쓰지 않음(채점하지 않은 경우도 있음)"),
    ("exp4_cells.csv", "list_no", "리스트 번호"),
    ("exp4_cells.csv", "presentation_order", "제시 순서(본 리스트 1~13번째, 연습은 빈칸)"),
    ("exp4_cells.csv", "total_correct", "리스트 정답 키워드 수 (0~30)"),
    ("exp4_cells.csv", "snr50", "SNR-50 (dB) = 5.5 − 0.6 × total_correct. 문장이 빠진 리스트는 빈칸"),
    ("exp4_cells.csv", "complete", "6문장이 모두 한 번씩 저장됐는지"),
    ("exp4_cells.csv", "test_date", "검사 날짜"),
    ("exp4_sentences.csv", "trial_index", "리스트 안 문장 순서"),
    ("exp4_sentences.csv", "snr_level", "문장 SNR (dB): +4, +1, −2, −5, −8, −11"),
    ("exp4_sentences.csv", "sentence_id", "문장 번호"),
    ("exp4_sentences.csv", "keywords_correct", "맞힌 키워드 수 (0~5)"),
    ("exp4_status.csv", "—", "참가자별 1차·2차 날짜, 2차 기한, 상태"),
    ("exp4_list_stats_round1.csv", "mean, sd, ci, dev", "1차 리스트별 평균·참가자 간 SD·95% CI 반폭·전체 평균과 차이(dB)"),
    ("exp4_norm.csv", "—", "정상 기준 계산(A, B, 사용 값, 참고용 정상 상한)"),
    ("exp4_retest_summary.csv", "icc, lo, hi, sem, cd95", "검사-재검사 요약(최종 12개 평균 / 리스트 1개). 2차 5명 이상일 때만"),
    ("exp4_retest_by_list.csv", "diff_sd, unstable", "리스트별 1차·2차 비교, 들쭉날쭉 표시"),
    ("exp4_final_lists.csv", "final, retest_unstable_flag", "최종 12개 포함 여부, 재검사 들쭉날쭉 표시"),
    ("exp4_list_set_sensitivity.csv", "—", "12·13·11개 리스트 세트별 기준·정상 기준 비교"),
], columns=["파일", "열", "설명"])


def build_zip() -> bytes:
    cl = cells[["participant_no", "test_round", "is_practice", "list_no", "presentation_order", "total_correct",
                "snr50", "complete", "test_date"]].copy()
    cl["snr50"] = cl["snr50"].round(2)
    se = df[["participant_no", "test_round", "is_practice", "list_no", "presentation_order", "trial_index",
             "snr_level", "sentence_id", "total_score", "test_date"]].rename(columns={"total_score": "keywords_correct"})
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        def put(name, frame):
            z.writestr(name, frame.to_csv(index=False).encode("utf-8-sig"))
        put("exp4_cells.csv", cl)
        put("exp4_sentences.csv", se)
        put("exp4_status.csv", status)
        if len(m1) >= 3:
            put("exp4_list_stats_round1.csv", U.equivalence(m1)["lists"].round(4))
        nv_ = U.norm_value(cells)
        if nv_:
            put("exp4_norm.csv", pd.DataFrame([{k: nv_[k] for k in ("n", "A", "A_sd", "B", "B_sd", "diff", "use",
                                                                      "value", "value_rounded", "sd", "upper")}]).round(4))
        if rt["ready"]:
            put("exp4_retest_summary.csv", pd.DataFrame([
                {"level": "12개 리스트 평균", **{k: rt["mean_level"][k] for k in
                                            ("n", "icc", "lo", "hi", "sem", "cd95", "diff_mean", "diff_sd", "p")}},
                {"level": "리스트 1개", "n": rt["n"], "icc": rt["single"]["icc_mean"], "sem": rt["single"]["sem"],
                 "cd95": rt["single"]["cd95"], "diff_mean": rt["single"]["diff_mean"]},
            ]).round(4))
            put("exp4_retest_by_list.csv", rt["per_list"].round(4))
        put("exp4_final_lists.csv", pd.DataFrame({
            "list_no": U.LISTS, "final": [l in fd["final_lists"] for l in U.LISTS],
            "retest_unstable_flag": [l in fd["unstable"] for l in U.LISTS]}))
        if len(fd["sensitivity"]):
            put("exp4_list_set_sensitivity.csv", fd["sensitivity"])
        put("codebook.csv", CODEBOOK)
        z.writestr("README.txt", (
            "Quick-SIN 한국어판 실험 4 데이터\n"
            f"내려받은 시각: {datetime.now(U.KST):%Y-%m-%d %H:%M} (KST)\n"
            f"상태: {'최종' if stg['final'] else '잠정(수집 중)'} — 1차 {stg['n_r1']}명, 2차 {stg['n_r2']}명\n\n"
            f"리스트: {', '.join(map(str, U.LISTS))} (연습: 8, 9)\n"
            "SNR-50 = 5.5 - 0.6 x 정답 키워드 수 (사다리 +4 ~ -11 dB, 3 dB 간격)\n"
            "청력 기준 미달자(exp4_excluded)는 포함하지 않음.\n"
            "개인정보(이름 등)와 검사 문장 원문은 포함하지 않음. 각 열의 뜻은 codebook.csv 참고.\n"
        ).encode("utf-8-sig"))
    return buf.getvalue()


with tabs[7]:
    st.markdown(
        "엑셀에서 바로 열 수 있는 CSV 파일 묶음(ZIP)입니다. **이름 등 개인정보는 없고 참가자 번호만 있습니다.** "
        "검사 문장 원문은 검사 자료 보호를 위해 빼고 문장 번호만 넣었습니다. "
        + ("" if stg["final"] else "**아직 수집 중이라 잠정 데이터입니다.**")
    )
    st.download_button("📦 실험 4 데이터 내려받기 (ZIP)", data=build_zip(),
                       file_name=f"qsin_exp4_data_{datetime.now(U.KST):%Y%m%d}.zip", mime="application/zip", type="primary")
    with st.expander("ZIP 안에 들어 있는 파일과 열 설명"):
        st.dataframe(CODEBOOK, hide_index=True, width="stretch")
