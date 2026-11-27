"""실험 3 최종 리스트 선별 판정(기준 A~E, 단계적 제외) 계산 모듈.

2026-11-17 판정에 쓴 로컬 스크립트(judge.py / iter.py / final.py)를 그대로 옮긴 것이다.
판정 규칙과 숫자는 여기에 고정한다. 기준값을 바꾸면 최종 13개 리스트가 달라질 수 있으므로
수정하지 말 것(바꾸려면 연구 책임자와 상의 후 판정 문서도 함께 고친다).

분석 단위: (참가자 x 리스트) 셀 1개 = 리스트 SNR-50 1개.
본분석: 참가자 1~18번(Williams 18x18 1주기, 324셀). 보조: 32~34번(민감도 분석에만 사용).
"""

import numpy as np
import pandas as pd
from scipy import stats

# ------------------------------------------------------------
# 확정 결과 (판정 문서: Quick-SIN 실험3 리스트 판정표, 2026-11-17)
# ------------------------------------------------------------
ALL_LISTS = list(range(1, 19))
FINAL_LISTS = [3, 4, 5, 6, 7, 10, 11, 12, 13, 14, 15, 16, 18]
EXCLUDED_LISTS = [1, 2, 8, 9, 17]
MAIN_PARTICIPANTS = list(range(1, 19))
SUPP_PARTICIPANTS = [32, 33, 34]

# ------------------------------------------------------------
# 판정 기준 (고정)
# ------------------------------------------------------------
DEV_LIMIT = 1.5         # A: |리스트 편차| > 1.5 dB → 제외
HOLM_ALPHA = 0.05       # B: Holm 보정 p < .05 → 제외
EQ_MARGIN = 1.0         # C: TOST 허용 범위 ±1.0 dB (90% CI)
RSD_FACTOR = 1.5        # D: 잔차 SD > 리스트 중앙값 x 1.5
SENT_Z = 2.0            # E: 같은 SNR 문장 평균의 |z| > 2
SENT_MIN_DIFF = 0.5     # E: 그리고 실제 차이 ≥ 0.5 키워드
MIN_LISTS = 12          # 최종 리스트 최소 개수

CRITERIA = pd.DataFrame([
    ('제외', 'A. 평균 편차', '리스트 평균이 전체 평균에서 얼마나 벗어났는가', f'|편차| > {DEV_LIMIT} dB'),
    ('제외', 'B. 통계적 이탈', '각 리스트 vs 전체 평균, 참가자별 편차의 t검정', f'Holm 보정 p < {HOLM_ALPHA}'),
    ('주의', 'C. 동등성', f'TOST(동등성 검정), 허용 범위 ±{EQ_MARGIN} dB', f'90% CI가 ±{EQ_MARGIN} dB 밖'),
    ('주의', 'D. 안정성', '순서 보정 모형의 잔차 SD(참가자마다 들쭉날쭉한 정도)', f'리스트 중앙값의 {RSD_FACTOR}배 초과'),
    ('주의', 'E. 문장 점검', '같은 SNR 문장 18개 안에서 문장 평균의 z',
     f'|z| > {SENT_Z:g} 이고 실제 차이 ≥ {SENT_MIN_DIFF} 키워드'),
], columns=['구분', '기준', '내용', '판정선'])


def snr50(total_correct):
    """정답 키워드 수(0~30) → SNR-50(dB). 5.5 − 0.6 × 정답 수."""
    return 5.5 - 0.6 * np.asarray(total_correct, dtype=float)


# ------------------------------------------------------------
# 통계 도구
# ------------------------------------------------------------
def rm_anova(Y: np.ndarray) -> dict:
    """반복측정 ANOVA(리스트 효과) + Greenhouse-Geisser 보정."""
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
                p_gg=stats.f.sf(F, df1 * eps, df2 * eps))


def holm(p) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    m = len(p)
    adj = np.empty(m)
    run = 0.0
    for r, i in enumerate(np.argsort(p)):
        run = max(run, (m - r) * p[i])
        adj[i] = min(1.0, run)
    return adj


def _fit(cells: pd.DataFrame, lists: list[int]):
    """OLS: SNR-50 ~ 참가자 + 제시위치 + 리스트. 순서 보정 행렬과 잔차를 돌려준다."""
    cl = cells[cells['list_no'].isin(lists)]
    subs = sorted(cl['participant_no'].unique())
    poss = sorted(cl['presentation_order'].unique())
    y = cl['snr50'].to_numpy(float)
    X = [np.ones(len(y))]
    X += [(cl['participant_no'] == s).to_numpy(float) for s in subs[1:]]
    X += [(cl['presentation_order'] == q).to_numpy(float) for q in poss[1:]]
    X += [(cl['list_no'] == l).to_numpy(float) for l in lists[1:]]
    X = np.array(X).T
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    res = y - X @ b
    ns, npos = len(subs), len(poss)
    peff = np.concatenate([[0.0], b[ns:ns + npos - 1]])
    peff -= peff.mean()
    pmap = dict(zip(poss, peff))
    yadj = y - cl['presentation_order'].map(pmap).to_numpy(float)
    k = len(lists)
    Yr, Ya, R = np.zeros((ns, k)), np.zeros((ns, k)), np.zeros((ns, k))
    si = {s: i for i, s in enumerate(subs)}
    li = {l: j for j, l in enumerate(lists)}
    for (p, l), a, r_, yy in zip(cl[['participant_no', 'list_no']].itertuples(index=False), yadj, res, y):
        Yr[si[p], li[l]] = yy
        Ya[si[p], li[l]] = a
        R[si[p], li[l]] = r_
    return Yr, Ya, R, pd.Series(peff, index=poss)


def judge(cells: pd.DataFrame, lists: list[int], sent_flags: dict, margin: float = EQ_MARGIN):
    """현재 리스트 세트에 기준 A~E를 1회 적용한다."""
    lists = sorted(lists)
    Yr, Ya, R, peff = _fit(cells, lists)
    n = Ya.shape[0]
    d = Ya - Ya.mean(1, keepdims=True)
    dev = d.mean(0)
    se = d.std(0, ddof=1) / np.sqrt(n)
    p = 2 * stats.t.sf(np.abs(dev / se), n - 1)
    ph = holm(p)
    tc = stats.t.ppf(0.95, n - 1)
    lo, hi = dev - tc * se, dev + tc * se
    rsd = R.std(0, ddof=1)
    med = float(np.median(rsd))
    rows = pd.DataFrame({
        'list_no': lists, 'dev': dev, 'ci_lo': lo, 'ci_hi': hi, 'p': p, 'p_holm': ph, 'resid_sd': rsd,
    })
    rows['A'] = rows['dev'].abs() > DEV_LIMIT
    rows['B'] = rows['p_holm'] < HOLM_ALPHA
    rows['C'] = ~((rows['ci_lo'] > -margin) & (rows['ci_hi'] < margin))
    rows['D'] = rows['resid_sd'] > RSD_FACTOR * med
    rows['E'] = rows['list_no'].map(lambda l: bool(sent_flags.get(l, False)))
    info = dict(n=n, resid_median=med, raw=rm_anova(Yr), adj=rm_anova(Ya), peff=peff,
                mean_snr50=float(Yr.mean()))
    return rows, info


def _reason(r: pd.Series, final_check: bool = False) -> str:
    if final_check:
        return f"최종 확인 TOST 미달 (90% CI [{r.ci_lo:+.2f}, {r.ci_hi:+.2f}] dB)"
    parts = []
    if r.A:
        parts.append(f"A(편차 {r.dev:+.2f} dB)")
    if r.B:
        parts.append(f"B(Holm p = {r.p_holm:.3f})" if r.p_holm >= .001 else "B(Holm p < .001)")
    if r.C:
        parts.append("C")
    if r.D:
        parts.append("D")
    if r.E:
        parts.append("E")
    return ", ".join(parts)


def iterate(cells: pd.DataFrame, sent_flags: dict, margin: float = EQ_MARGIN):
    """단계적 제외: 제외 리스트가 없을 때까지 남은 리스트로 기준을 다시 적용한다.

    규칙
      - A 또는 B → 제외, C·D·E 중 2개 이상 → 제외 (1개면 주의 표시 후 유지)
      - 제외할 리스트가 없으면 최종 확인: 남은 리스트 전부 TOST 통과해야 함.
        TOST 미달 리스트가 있으면 그 리스트를 빼고 다시 확인(12개 이상 남을 때만).
    """
    keep = list(ALL_LISTS)
    stages = []
    while True:
        rows, info = judge(cells, keep, sent_flags, margin)
        rule = rows[rows.A | rows.B | ((rows.C.astype(int) + rows.D.astype(int) + rows.E.astype(int)) >= 2)]
        final_check = False
        if rule.empty:
            tost_fail = rows[rows.C]
            if not tost_fail.empty and len(keep) - len(tost_fail) >= MIN_LISTS:
                rule, final_check = tost_fail, True
        stages.append(dict(
            stage=len(stages) + 1, n_lists=len(keep), lists=list(keep),
            excluded=[int(l) for l in rule['list_no']],
            reason="; ".join(f"리스트 {int(r.list_no)}: {_reason(r, final_check)}" for r in rule.itertuples()),
            gg_raw=info['raw']['p_gg'], gg_adj=info['adj']['p_gg'], rows=rows, info=info,
        ))
        if rule.empty:
            break
        keep = [l for l in keep if l not in set(rule['list_no'])]
    return stages


# ------------------------------------------------------------
# 데이터 준비
# ------------------------------------------------------------
def cells_from_sentences(df: pd.DataFrame) -> pd.DataFrame:
    """문장 단위 데이터 → (참가자 x 리스트) 셀. 사다리 6문장이 모두 있는 셀만 쓴다."""
    g = (df.dropna(subset=['participant_no', 'list_no'])
         .groupby(['participant_no', 'list_no'], as_index=False)
         .agg(presentation_order=('presentation_order', 'first'),
              total_correct=('total_score', 'sum'),
              n_sentences=('sentence_id', 'nunique'),
              n_rows=('sentence_id', 'size')))
    g = g[(g.n_sentences == 6) & (g.n_rows == 6)].copy()
    for c in ['participant_no', 'list_no', 'presentation_order']:
        g[c] = g[c].astype(int)
    g['snr50'] = snr50(g['total_correct'])
    return g.drop(columns=['n_sentences', 'n_rows'])


def sentence_check(df: pd.DataFrame) -> pd.DataFrame:
    """기준 E: 리스트 x SNR(= 문장 1개)별 평균 정답 키워드, 같은 SNR 18문장 안에서 z."""
    main = df[df['participant_no'].isin(MAIN_PARTICIPANTS)]
    s = (main.groupby(['list_no', 'snr_level', 'sentence_id'], as_index=False)
         .agg(mean_correct=('total_score', 'mean'), n=('total_score', 'size')))
    s['list_no'] = s['list_no'].astype(int)
    grp = s.groupby('snr_level')['mean_correct']
    s['snr_mean'] = grp.transform('mean')
    s['snr_sd'] = grp.transform(lambda v: v.std(ddof=1))
    s['diff'] = s['mean_correct'] - s['snr_mean']
    s['z'] = np.where(s['snr_sd'] > 0, s['diff'] / s['snr_sd'], 0.0)
    s['z_flag'] = s['z'].abs() > SENT_Z
    s['E'] = s['z_flag'] & (s['diff'].abs() >= SENT_MIN_DIFF)
    return s.sort_values(['snr_level', 'list_no'], ascending=[False, True], ignore_index=True)


def run_all(df: pd.DataFrame) -> dict:
    """페이지·다운로드에 필요한 결과를 한 번에 계산한다."""
    cells_all = cells_from_sentences(df)
    main = cells_all[cells_all['participant_no'].isin(MAIN_PARTICIPANTS)]
    sent = sentence_check(df)
    flags = sent.groupby('list_no')['E'].any().to_dict()
    flags_strict = sent.groupby('list_no')['z_flag'].any().to_dict()
    stages = iterate(main, flags)
    kept = stages[-1]['lists']

    # 리스트별 판정표: 18개 기준(1단계) 편차 + 최종 13개 기준 편차
    s1 = stages[0]['rows'].set_index('list_no')
    final_rows, final_info = judge(main, FINAL_LISTS, flags)
    fr = final_rows.set_index('list_no')
    excl_stage = {l: st_['stage'] for st_ in stages for l in st_['excluded']}
    excl_reason = {}
    for st_ in stages:
        for part in st_['reason'].split('; '):
            if part.startswith('리스트 '):
                l, why = part[4:].split(': ', 1)
                excl_reason[int(l)] = f"{st_['stage']}단계 {why}"
    verdict = pd.DataFrame({'list_no': ALL_LISTS})
    verdict['판정'] = verdict['list_no'].map(lambda l: '사용' if l in kept else '제외')
    verdict['사유'] = verdict['list_no'].map(lambda l: excl_reason.get(l, '—'))
    verdict['dev18'] = verdict['list_no'].map(s1['dev'])
    verdict['holm18'] = verdict['list_no'].map(s1['p_holm'])
    verdict['dev13'] = verdict['list_no'].map(fr['dev'])
    verdict['lo13'] = verdict['list_no'].map(fr['ci_lo'])
    verdict['hi13'] = verdict['list_no'].map(fr['ci_hi'])
    verdict['raw_mean'] = verdict['list_no'].map(main.groupby('list_no')['snr50'].mean())

    # 순서 효과(본분석 18리스트, 원자료)
    pos = main.groupby('presentation_order')['snr50'].agg(['mean', 'std', 'count']).reset_index()
    slope = stats.linregress(main['presentation_order'], main['snr50'])

    # 민감도 분석
    sens = []
    conds = [
        ('본분석 (참가자 1~18번, ±1.0 dB)', main, flags, EQ_MARGIN),
        ('① 보조 참가자 32~34번 추가 (n=21)', cells_all, flags, EQ_MARGIN),
        ('② 참가자 11번 제외 (n=17)', main[main['participant_no'] != 11], flags, EQ_MARGIN),
        ('③ 허용 범위 ±1.5 dB', main, flags, 1.5),
        ('④ 기준 E를 사전 기준(|z| > 2만)으로', main, flags_strict, EQ_MARGIN),
    ]
    for name, c, fl, mg in conds:
        stg = iterate(c, fl, mg)
        r13, i13 = judge(c, FINAL_LISTS, fl, mg)
        sens.append({
            '조건': name,
            '참가자 수': int(c['participant_no'].nunique()),
            '단계별 제외': ' → '.join(str(s_['excluded']) for s_ in stg if s_['excluded']) or '없음',
            '남은 리스트 수': len(stg[-1]['lists']),
            '최종 13개 TOST 미달': ', '.join(str(int(l)) for l in r13.loc[r13.C, 'list_no']) or '없음',
            '13개 리스트 효과 GG p (원자료)': i13['raw']['p_gg'],
            '13개 리스트 효과 GG p (순서 보정)': i13['adj']['p_gg'],
        })

    return dict(
        cells=cells_all, main=main, sent=sent, stages=stages, kept=kept,
        matches_final=sorted(kept) == FINAL_LISTS,
        verdict=verdict, final_rows=final_rows, final_info=final_info,
        position=pos, slope=slope, peff=stages[0]['info']['peff'],
        sensitivity=pd.DataFrame(sens),
        n_main=int(main['participant_no'].nunique()), n_cells_main=len(main),
        mean18=float(main['snr50'].mean()),
        mean13=float(main[main['list_no'].isin(FINAL_LISTS)]['snr50'].mean()),
    )
