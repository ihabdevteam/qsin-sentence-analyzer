"""실험 4(13개 리스트 최종 검증 + 검사-재검사) 데이터 로더와 분석 함수.

판정 기준은 수집 전에 정한 분석 계획(2026-11-18)을 따른다. 기준값은 여기에 고정하고 화면에서 바꾸지 않는다.

데이터
  - test_reports_qsin.experiment = 'exp4' (청력 미달자는 'exp4_excluded'로 옮겨져 자동 제외)
  - test_round: 1 = 1차, 2 = 2차(재검사) / is_practice: 연습 리스트(8, 9) 여부
  - 리스트 번호 = sound_set, 제시 순서 = presentation_order(본 리스트만 1~13)
  - 문장 SNR = score_qsin.snr_level, 정답 키워드 수 = score_qsin.total_score
"""

from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import streamlit as st
from scipy import stats

KST = ZoneInfo("Asia/Seoul")

# ------------------------------------------------------------
# 설계 (고정)
# ------------------------------------------------------------
LISTS = [3, 4, 5, 6, 7, 10, 11, 12, 13, 14, 15, 16, 18]
PRACTICE_LISTS = [8, 9]
N_POS = len(LISTS)
TARGET_PARTICIPANTS = 30        # Williams 26행 + 예비 4명
WILLIAMS_ROWS = 26
RETEST_MAX_DAYS = 14            # 2차는 1차 후 14일 이내
RETEST_SOON_DAYS = 3            # 기한 임박 표시

# 리스트 세트 (민감도 분석용)
LIST_SETS = {
    "13개 전체 (계획)": LISTS,
    "12개 (리스트 4 제외)": [l for l in LISTS if l != 4],
    "11개 (리스트 4·11 제외)": [l for l in LISTS if l not in (4, 11)],
}

# 판정 기준 (분석 계획)
CRIT_BETWEEN_LIST_SD = 1.0      # 리스트 간 SNR-50 SD < 1.0 dB
CRIT_SCORE_SD = 1.5             # 리스트 간 점수 SD < 1.5점
CRIT_ANOVA_P = 0.05             # 리스트 효과 p ≥ .05 (Greenhouse-Geisser 보정)
MIN_LISTS = 10                  # 최종 리스트 최소 개수
NORM_RULE_DIFF = 0.5            # |A − B| < 0.5 dB면 B, 아니면 A
NORM_UPPER_Z = 1.645            # 참고용 정상 상한 = 기준 + 1.645 SD (계획서 외)

# Williams 26행 순서표: 행 = (참가자 번호 − 1) mod 26. 각 행 = 제시 순서 1~13번째 리스트.
WILLIAMS26 = [
    [3, 4, 18, 5, 16, 6, 15, 7, 14, 10, 13, 11, 12], [4, 5, 3, 6, 18, 7, 16, 10, 15, 11, 14, 12, 13],
    [5, 6, 4, 7, 3, 10, 18, 11, 16, 12, 15, 13, 14], [6, 7, 5, 10, 4, 11, 3, 12, 18, 13, 16, 14, 15],
    [7, 10, 6, 11, 5, 12, 4, 13, 3, 14, 18, 15, 16], [10, 11, 7, 12, 6, 13, 5, 14, 4, 15, 3, 16, 18],
    [11, 12, 10, 13, 7, 14, 6, 15, 5, 16, 4, 18, 3], [12, 13, 11, 14, 10, 15, 7, 16, 6, 18, 5, 3, 4],
    [13, 14, 12, 15, 11, 16, 10, 18, 7, 3, 6, 4, 5], [14, 15, 13, 16, 12, 18, 11, 3, 10, 4, 7, 5, 6],
    [15, 16, 14, 18, 13, 3, 12, 4, 11, 5, 10, 6, 7], [16, 18, 15, 3, 14, 4, 13, 5, 12, 6, 11, 7, 10],
    [18, 3, 16, 4, 15, 5, 14, 6, 13, 7, 12, 10, 11], [12, 11, 13, 10, 14, 7, 15, 6, 16, 5, 18, 4, 3],
    [13, 12, 14, 11, 15, 10, 16, 7, 18, 6, 3, 5, 4], [14, 13, 15, 12, 16, 11, 18, 10, 3, 7, 4, 6, 5],
    [15, 14, 16, 13, 18, 12, 3, 11, 4, 10, 5, 7, 6], [16, 15, 18, 14, 3, 13, 4, 12, 5, 11, 6, 10, 7],
    [18, 16, 3, 15, 4, 14, 5, 13, 6, 12, 7, 11, 10], [3, 18, 4, 16, 5, 15, 6, 14, 7, 13, 10, 12, 11],
    [4, 3, 5, 18, 6, 16, 7, 15, 10, 14, 11, 13, 12], [5, 4, 6, 3, 7, 18, 10, 16, 11, 15, 12, 14, 13],
    [6, 5, 7, 4, 10, 3, 11, 18, 12, 16, 13, 15, 14], [7, 6, 10, 5, 11, 4, 12, 3, 13, 18, 14, 16, 15],
    [10, 7, 11, 6, 12, 5, 13, 4, 14, 3, 15, 18, 16], [11, 10, 12, 7, 13, 6, 14, 5, 15, 4, 16, 3, 18],
]


def expected_order(participant_no: int) -> list[int]:
    return WILLIAMS26[(int(participant_no) - 1) % WILLIAMS_ROWS]


def snr50(total_correct):
    """정답 키워드 수(0~30) → SNR-50(dB) = 5.5 − 0.6 × 정답 수."""
    return 5.5 - 0.6 * np.asarray(total_correct, dtype=float)


def today_kst() -> date:
    return datetime.now(KST).date()


# ------------------------------------------------------------
# 데이터 로딩
# ------------------------------------------------------------
@st.cache_data(ttl=600)
def load_exp4(_supabase_client) -> pd.DataFrame:
    """문장 단위 long 데이터. 한 행 = 참가자 1명이 한 회차·한 리스트에서 들은 문장 1개.

    조회 실패 시 예외를 그대로 올린다(캐시에 빈 결과가 남지 않도록).
    """
    sel = ("index, total_score, snr_level, trial_index,"
           " test_reports_qsin!inner(id, session_id, participant_no, sound_set, presentation_order,"
           " test_datetime, experiment, test_round, is_practice)")
    rows, page, size = [], 0, 1000
    while True:
        resp = (_supabase_client.table("score_qsin").select(sel)
                .eq("test_reports_qsin.experiment", "exp4")
                .order("id").range(page * size, page * size + size - 1).execute())
        batch = resp.data or []
        rows.extend(batch)
        if len(batch) < size:
            break
        page += 1
    if not rows:
        return pd.DataFrame()
    rec = []
    for it in rows:
        r = it.get("test_reports_qsin") or {}
        rec.append({
            "participant_no": r.get("participant_no"), "test_round": r.get("test_round"),
            "is_practice": bool(r.get("is_practice")), "report_id": r.get("id"),
            "list_no": r.get("sound_set"), "presentation_order": r.get("presentation_order"),
            "test_datetime": r.get("test_datetime"), "trial_index": it.get("trial_index"),
            "snr_level": it.get("snr_level"), "sentence_id": it.get("index"), "total_score": it.get("total_score"),
        })
    df = pd.DataFrame(rec)
    for c in ["participant_no", "test_round", "list_no", "presentation_order", "trial_index", "sentence_id"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")
    for c in ["snr_level", "total_score"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["test_date"] = pd.to_datetime(df["test_datetime"], utc=True, errors="coerce").dt.tz_convert(KST).dt.date
    return df.drop(columns=["test_datetime"]).sort_values(
        ["participant_no", "test_round", "is_practice", "presentation_order", "trial_index"], ignore_index=True)


def build_cells(df: pd.DataFrame) -> pd.DataFrame:
    """(참가자 × 회차 × 리스트) 셀. 6문장이 모두 있어야 complete."""
    if df.empty:
        return pd.DataFrame()
    g = (df.dropna(subset=["participant_no", "test_round", "list_no"])
         .groupby(["participant_no", "test_round", "is_practice", "list_no"], as_index=False)
         .agg(presentation_order=("presentation_order", "first"), total_correct=("total_score", "sum"),
              n_rows=("sentence_id", "size"), n_sentences=("sentence_id", "nunique"),
              n_reports=("report_id", "nunique"), test_date=("test_date", "min")))
    g["complete"] = (g["n_rows"] == 6) & (g["n_sentences"] == 6) & (g["n_reports"] == 1)
    g["snr50"] = np.where(g["complete"], snr50(g["total_correct"]), np.nan)
    for c in ["participant_no", "test_round", "list_no"]:
        g[c] = g[c].astype(int)
    return g


def main_cells(cells: pd.DataFrame, test_round: int) -> pd.DataFrame:
    c = cells[(cells["test_round"] == test_round) & (~cells["is_practice"]) & cells["complete"]]
    return c.astype({"presentation_order": int})


def matrix(cells: pd.DataFrame, test_round: int, lists=LISTS) -> pd.DataFrame:
    """참가자 × 리스트 SNR-50 행렬. 해당 리스트를 모두 마친 참가자만."""
    c = main_cells(cells, test_round)
    m = c.pivot(index="participant_no", columns="list_no", values="snr50")
    m = m.reindex(columns=lists)
    return m.dropna(how="any").sort_index()


# ------------------------------------------------------------
# 수집 현황
# ------------------------------------------------------------
def collection_status(cells: pd.DataFrame, today: date | None = None) -> pd.DataFrame:
    today = today or today_kst()
    rows = []
    main = cells[~cells["is_practice"]]
    for p in sorted(main["participant_no"].unique()):
        c1 = main[(main.participant_no == p) & (main.test_round == 1)]
        c2 = main[(main.participant_no == p) & (main.test_round == 2)]
        n1, n2 = int(c1["complete"].sum()), int(c2["complete"].sum())
        d1 = c1["test_date"].min() if len(c1) else None
        d2 = c2["test_date"].min() if len(c2) else None
        deadline = d1 + timedelta(days=RETEST_MAX_DAYS) if d1 else None
        if n1 < N_POS:
            state = "1차 미완료"
        elif n2 >= N_POS:
            state = "2차 완료"
        elif n2 > 0:
            state = "2차 진행 중(미완료)"
        elif today > deadline:
            state = "기한 초과(탈락)"
        elif (deadline - today).days <= RETEST_SOON_DAYS:
            state = f"기한 임박 (D-{(deadline - today).days})"
        else:
            state = f"2차 대기 (D-{(deadline - today).days})"
        rows.append({
            "참가자": int(p), "순서표 행": (int(p) - 1) % WILLIAMS_ROWS + 1,
            "1차 날짜": d1, "1차 리스트": n1, "2차 기한": deadline, "2차 날짜": d2, "2차 리스트": n2,
            "간격(일)": (d2 - d1).days if (d1 and d2) else None, "상태": state,
        })
    return pd.DataFrame(rows)


def stage(status: pd.DataFrame) -> dict:
    """잠정/최종 판정. 1차 26명 이상 + 모든 1차 완료자가 2차 완료 또는 기한 초과 → 최종."""
    r1 = status[status["1차 리스트"] >= N_POS]
    done = int((r1["상태"] == "2차 완료").sum())
    dropped = int((r1["상태"] == "기한 초과(탈락)").sum())
    waiting = len(r1) - done - dropped
    final = len(r1) >= WILLIAMS_ROWS and waiting == 0
    return dict(final=final, n_r1=len(r1), n_r2=done, n_drop=dropped, n_wait=waiting)


# ------------------------------------------------------------
# 데이터 점검
# ------------------------------------------------------------
def data_checks(cells: pd.DataFrame, status: pd.DataFrame) -> list[dict]:
    """항목별 점검 결과. 각 항목: 항목, 결과(문장), 해당(참가자 목록), ok(bool)."""
    out = []
    main = cells[~cells["is_practice"]]

    def add(name, bad, ok_msg, bad_msg, help_=""):
        out.append(dict(항목=name, ok=not bad, 결과=ok_msg if not bad else bad_msg.format(n=len(bad)),
                        해당=", ".join(map(str, bad)) if bad else "—", 설명=help_))

    # 1. Williams 순서
    bad = []
    for (p, r), g in main.groupby(["participant_no", "test_round"]):
        got = g.dropna(subset=["presentation_order"]).sort_values("presentation_order")["list_no"].tolist()
        exp = expected_order(p)
        if got != exp[:len(got)]:
            bad.append(f"{p}번({r}차)")
    add("제시 순서가 순서표와 같은가", bad, "모두 일치", "{n}건 불일치",
        "참가자 번호로 정해진 Williams 순서표와 실제 제시 순서를 비교합니다.")

    # 2. 빠진 문장·중복 저장
    bad = [f"{r.participant_no}번({r.test_round}차) 리스트 {r.list_no}" for r in cells[~cells["complete"]].itertuples()]
    add("리스트마다 6문장이 한 번씩 저장됐는가", bad, "이상 없음", "{n}개 리스트 불완전",
        "문장이 빠졌거나 같은 리스트가 두 번 저장된 경우입니다. 해당 리스트는 분석에서 빠집니다.")

    # 3. 연습 리스트 30/30
    pr = cells[cells["is_practice"]]
    full = pr.groupby(["participant_no", "test_round"])["total_correct"].apply(lambda s: (s >= 30).all() and len(s) == 2)
    bad = [f"{p}번({r}차)" for (p, r), v in full.items() if v]
    add("연습 리스트를 채점했는가", bad, "모두 채점됨", "{n}건 연습 2개 모두 30/30",
        "연습 리스트 2개가 모두 만점이면 채점 없이 넘겼을 가능성이 있습니다. 분석에는 영향 없습니다(연습은 분석 제외).")

    # 4. 2차 순서 = 1차 순서
    bad = []
    for p, g in main.groupby("participant_no"):
        o1 = g[g.test_round == 1].sort_values("presentation_order")["list_no"].tolist()
        o2 = g[g.test_round == 2].sort_values("presentation_order")["list_no"].tolist()
        if o2 and o1[:len(o2)] != o2:
            bad.append(f"{p}번")
    add("2차 순서가 1차와 같은가", bad, "모두 같음(또는 2차 없음)", "{n}명 다름")

    # 5. 간격
    s = status.dropna(subset=["간격(일)"])
    bad = [f"{r['참가자']}번({int(r['간격(일)'])}일)" for _, r in s.iterrows() if r["간격(일)"] > RETEST_MAX_DAYS]
    add("2차가 1차 후 14일 이내인가", bad, "모두 14일 이내(또는 2차 없음)", "{n}명 14일 초과")

    # 6. 참가자 이상치 (1차 13리스트 평균)
    m = matrix(cells, 1)
    bad = []
    if len(m) >= 5:
        pm = m.mean(1)
        q1, q3 = pm.quantile(.25), pm.quantile(.75)
        lo, hi = q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1)
        bad = [f"{p}번({v:+.2f} dB)" for p, v in pm.items() if v < lo or v > hi]
    add("평균이 다른 참가자와 크게 다른 사람이 있는가", bad, "없음", "{n}명",
        "1차 13개 리스트 평균이 사분위 범위(IQR)의 1.5배 밖이면 표시합니다. 표시만 하고 분석에서 빼지는 않습니다.")
    return out


# ------------------------------------------------------------
# 통계
# ------------------------------------------------------------
def rm_anova(Y: np.ndarray) -> dict:
    n, k = Y.shape
    g = Y.mean()
    ssl = n * ((Y.mean(0) - g) ** 2).sum()
    E = Y - Y.mean(1, keepdims=True) - Y.mean(0, keepdims=True) + g
    sse = (E ** 2).sum()
    df1, df2 = k - 1, (n - 1) * (k - 1)
    F = (ssl / df1) / (sse / df2)
    S = np.cov(Y, rowvar=False)
    C = np.eye(k) - 1 / k
    Sc = C @ S @ C
    eps = max(np.trace(Sc) ** 2 / ((k - 1) * (Sc ** 2).sum()), 1 / (k - 1))
    return dict(F=F, df1=df1, df2=df2, p=stats.f.sf(F, df1, df2), eps=eps,
                p_gg=stats.f.sf(F, df1 * eps, df2 * eps), pes=ssl / (ssl + sse))


def equivalence(m: pd.DataFrame) -> dict:
    """리스트 동등성 기준 3개 + 리스트별 통계."""
    n = len(m)
    means = m.mean()
    gm = float(m.values.mean())
    t = stats.t.ppf(.975, n - 1)
    ls = pd.DataFrame({"list_no": m.columns, "mean": means.values, "sd": m.std(ddof=1).values})
    ls["ci"] = t * ls["sd"] / np.sqrt(n)
    ls["dev"] = ls["mean"] - gm
    between_sd = float(means.std(ddof=1))
    a = rm_anova(m.values)
    crit = pd.DataFrame([
        ("리스트 간 SNR-50 SD", f"{between_sd:.2f} dB", f"< {CRIT_BETWEEN_LIST_SD} dB", between_sd < CRIT_BETWEEN_LIST_SD),
        ("리스트 간 점수 SD", f"{between_sd / 0.6:.2f}점", f"< {CRIT_SCORE_SD}점", between_sd / 0.6 < CRIT_SCORE_SD),
        ("리스트 효과 (반복측정 ANOVA, GG 보정)",
         f"F({a['df1']},{a['df2']}) = {a['F']:.2f}, p = {a['p_gg']:.3f}", f"p ≥ {CRIT_ANOVA_P}", a["p_gg"] >= CRIT_ANOVA_P),
    ], columns=["기준", "현재 값", "통과선", "통과"])
    return dict(n=n, grand=gm, lists=ls, between_sd=between_sd, anova=a, crit=crit, passed=bool(crit["통과"].all()))


def stepwise_candidates(m: pd.DataFrame) -> list[dict]:
    """기준을 통과할 때까지 평균에서 가장 먼 리스트를 하나씩 뺀다(최소 10개). 제외 '후보'만 계산."""
    steps, cur = [], m.copy()
    while True:
        e = equivalence(cur)
        steps.append(dict(n_lists=cur.shape[1], lists=list(cur.columns), passed=e["passed"],
                          between_sd=e["between_sd"], p_gg=e["anova"]["p_gg"]))
        if e["passed"] or cur.shape[1] - 1 < MIN_LISTS:
            break
        worst = int(e["lists"].loc[e["lists"]["dev"].abs().idxmax(), "list_no"])
        steps[-1]["drop"] = worst
        steps[-1]["drop_dev"] = float(e["lists"].set_index("list_no").loc[worst, "dev"])
        cur = cur.drop(columns=worst)
    return steps


def position_effect(cells: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for r in (1, 2):
        c = main_cells(cells, r)
        if c.empty:
            continue
        g = c.groupby("presentation_order")["snr50"].agg(["mean", "std", "count"]).reset_index()
        g["test_round"] = r
        rows.append(g)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def norm_value(cells: pd.DataFrame) -> dict | None:
    """정상 기준 평균(분석 계획). 1차만, 13개 리스트를 모두 마친 참가자."""
    c = main_cells(cells, 1)
    full = c.groupby("participant_no")["list_no"].nunique()
    c = c[c["participant_no"].isin(full[full == N_POS].index)]
    if c.empty:
        return None
    A_p = c[c["presentation_order"] <= 2].groupby("participant_no")["snr50"].mean()
    B_p = c.groupby("participant_no")["snr50"].mean()
    n = len(B_p)
    A, B = float(A_p.mean()), float(B_p.mean())
    use_A = abs(A - B) >= NORM_RULE_DIFF
    chosen = A_p if use_A else B_p
    t = stats.t.ppf(.975, n - 1) if n > 1 else np.nan
    sd = float(chosen.std(ddof=1)) if n > 1 else np.nan
    val = float(chosen.mean())
    return dict(n=n, A=A, B=B, A_sd=float(A_p.std(ddof=1)), B_sd=float(B_p.std(ddof=1)), diff=A - B,
                use="A" if use_A else "B", value=val, value_rounded=round(val, 1), sd=sd,
                ci=(val - t * sd / np.sqrt(n), val + t * sd / np.sqrt(n)),
                upper=val + NORM_UPPER_Z * sd, A_p=A_p, B_p=B_p)
