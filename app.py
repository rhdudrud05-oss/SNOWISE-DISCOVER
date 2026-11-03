# -*- coding: utf-8 -*-
"""SNOWISE DISCOVER - 비교과 맞춤형 추천·안내 대시보드

실행:
    pip install -r requirements_snowise.txt
    streamlit run snowise_discover_app.py

사이드바에서 03_비교과_프로그램.csv와 04_비교과_이수자.csv를 업로드하세요.
주의: 본 앱은 2022~2025 과거 데이터를 활용한 분석·추천 시뮬레이션입니다.
실시간 모집상태, 정원, 실제 SNOWISE 추천 노출/클릭 로그는 포함하지 않습니다.
"""

from __future__ import annotations

import io
import math
import re
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


# -----------------------------------------------------------------------------
# 페이지 설정 및 디자인
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="SNOWISE DISCOVER | 비교과 데이터 대시보드",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .block-container {padding-top: 1.6rem; padding-bottom: 2.5rem;}
    [data-testid="stMetric"] {
        background: #f6f8fc; border: 1px solid #e6eaf2;
        padding: 14px 16px; border-radius: 12px;
    }
    .main-title {font-size: 2.05rem; font-weight: 800; letter-spacing: -0.04em;}
    .sub-title {color: #667085; font-size: 0.96rem; margin-top: -8px;}
    .insight-box {
        background: #f4f7fb; border-left: 4px solid #4778c7;
        padding: 14px 16px; border-radius: 8px; margin: 8px 0 16px 0;
    }
    .small-note {color: #667085; font-size: 0.82rem;}
    </style>
    """,
    unsafe_allow_html=True,
)

KEY = ["프로그램ID", "세부ID"]
VALID_GRADES = ["1학년", "2학년", "3학년", "4학년"]
TYPE_ORDER = ["A 참여 저조형", "B 중앙 저이용형", "C 참여 격차형", "D 참여 활발형"]
TYPE_SHORT = {
    "A 참여 저조형": "A형",
    "B 중앙 저이용형": "B형",
    "C 참여 격차형": "C형",
    "D 참여 활발형": "D형",
}
CATEGORY_ORDER = ["학습", "취업", "진로", "학생활동", "심리", "창업"]


# -----------------------------------------------------------------------------
# 데이터 로딩 및 전처리
# -----------------------------------------------------------------------------
def _read_csv_bytes(raw: bytes) -> pd.DataFrame:
    """한글 CSV의 인코딩 차이를 순서대로 시도한다."""
    errors = []
    for enc in ("utf-8-sig", "cp949", "euc-kr", "utf-8"):
        try:
            return pd.read_csv(io.BytesIO(raw), encoding=enc, low_memory=False)
        except Exception as exc:  # 인코딩 실패 시 다음 인코딩 시도
            errors.append(f"{enc}: {exc}")
    raise ValueError("CSV 인코딩을 읽지 못했습니다. UTF-8 또는 CP949로 저장해 주세요.\n" + "\n".join(errors))


def _normalise_id(series: pd.Series) -> pd.Series:
    """CSV의 ID가 123과 123.0으로 다르게 읽히는 문제를 줄인다."""
    out = series.astype("string").str.strip()
    return out.str.replace(r"\.0$", "", regex=True)


def _mode_or_first(values: pd.Series) -> Any:
    values = values.dropna()
    if values.empty:
        return "미상"
    modes = values.mode()
    return modes.iloc[0] if not modes.empty else values.iloc[0]


def _require_columns(df: pd.DataFrame, required: list[str], label: str) -> None:
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"{label} 파일에 필요한 열이 없습니다: {', '.join(missing)}")


@st.cache_data(show_spinner=False)
def prepare_data(program_bytes: bytes, participant_bytes: bytes) -> dict[str, Any]:
    """프로그램 파일과 이수자 파일을 기존 EDA 기준에 맞춰 정리한다."""
    programs_raw = _read_csv_bytes(program_bytes)
    participants_raw = _read_csv_bytes(participant_bytes)

    _require_columns(
        programs_raw,
        KEY + ["프로그램명", "카테고리1", "법정의무_기타플래그", "프로그램그룹", "운영기관", "운영부서"],
        "03_비교과_프로그램",
    )
    _require_columns(
        participants_raw,
        KEY + ["가명학번", "참여학년도", "신청상태", "이수여부", "학년", "학과전공"],
        "04_비교과_이수자",
    )

    programs = programs_raw.copy()
    participants = participants_raw.copy()

    for col in KEY:
        programs[col] = _normalise_id(programs[col])
        participants[col] = _normalise_id(participants[col])

    # 원본 EDA와 동일하게 TEST 프로그램 ID 6280 제외
    programs = programs.loc[programs["프로그램ID"].ne("6280")].copy()
    participants = participants.loc[participants["프로그램ID"].ne("6280")].copy()

    # 원본 EDA와 동일하게 프로그램명/세부주제에 '폐강'이 명시된 회차 제외
    cancel_mask = programs["프로그램명"].fillna("").astype(str).str.contains("폐강", na=False)
    if "세부주제" in programs.columns:
        cancel_mask |= programs["세부주제"].fillna("").astype(str).str.contains("폐강", na=False)
    cancel_keys = set(map(tuple, programs.loc[cancel_mask, KEY].astype(str).to_numpy()))
    programs = programs.loc[~cancel_mask].copy()
    if cancel_keys:
        participant_keys = list(zip(participants[KEY[0]].astype(str), participants[KEY[1]].astype(str)))
        participants = participants.loc[[key not in cancel_keys for key in participant_keys]].copy()

    duplicate_program_keys = int(programs.duplicated(KEY).sum())
    if duplicate_program_keys:
        raise ValueError(
            f"03 파일에서 프로그램ID+세부ID 중복이 {duplicate_program_keys:,}건 발견됐습니다. "
            "기존 분석은 이 조합이 중복되지 않는다는 전제로 병합하므로 원본을 확인해 주세요."
        )

    # 분석에 사용하는 프로그램 메타데이터만 연결
    optional_meta = [
        "카테고리1", "카테고리2", "법정의무_기타플래그", "핵심역량",
        "신청대상_학년", "시작일", "프로그램명", "프로그램그룹",
        "운영기관", "운영부서", "운영학년도", "운영학기",
    ]
    meta_cols = KEY + [c for c in optional_meta if c in programs.columns]
    merged = participants.merge(
        programs[meta_cols], on=KEY, how="left", validate="m:1",
        suffixes=("", "_프로그램"), indicator="_program_match"
    )
    program_info_missing = int(merged["_program_match"].eq("left_only").sum())

    # 필수 파생 변수
    merged["승인"] = merged["신청상태"].astype(str).str.strip().eq("승인")
    merged["이수"] = merged["이수여부"].astype(str).str.strip().eq("이수")
    merged["참여학년도"] = pd.to_numeric(merged["참여학년도"], errors="coerce")
    merged["법정의무_기타플래그"] = pd.to_numeric(merged["법정의무_기타플래그"], errors="coerce")
    merged["법정"] = merged["법정의무_기타플래그"].eq(1)
    merged["학년"] = merged["학년"].fillna("기타·미상").astype(str).str.strip()
    merged["학년_정리"] = np.where(
        merged["학년"].isin(VALID_GRADES), merged["학년"],
        np.where(merged["학년"].eq("약대"), "약대", "기타·미상")
    )
    merged["가명학번"] = merged["가명학번"].astype("string").str.strip()
    merged["학과전공"] = merged["학과전공"].fillna("미상").astype(str).str.strip()
    merged["카테고리1"] = merged["카테고리1"].fillna("미분류").astype(str).str.strip()
    if "시작일" in merged.columns:
        merged["시작일"] = pd.to_datetime(merged["시작일"], errors="coerce")
    if "단과대학" not in merged.columns:
        merged["단과대학"] = "미상"

    # 자율 비교과 승인 기록 (기존 분석의 핵심 기준)
    voluntary = merged.loc[merged["승인"] & ~merged["법정"]].copy()
    approved = merged.loc[merged["승인"]].copy()

    # 프로그램별 운영 주체 분류: 기존 코드의 규칙을 그대로 사용
    org = programs[KEY + [c for c in ["운영기관", "운영부서"] if c in programs.columns]].copy()
    v2 = voluntary.merge(org, on=KEY, how="left", validate="m:1", suffixes=("", "_원본"))
    institution = v2["운영기관"].fillna("").astype(str) if "운영기관" in v2.columns else pd.Series("", index=v2.index)
    department = v2["운영부서"].fillna("").astype(str) if "운영부서" in v2.columns else pd.Series("", index=v2.index)
    is_dept = institution.str.endswith("대학", na=False) | department.str.contains("학과|학부|전공", na=False)
    v2["운영주체"] = np.where(is_dept, "학과·단과대", "중앙")

    # 학생×학년도 패널: 승인 기록이 관측된 학생만 포함
    panel_index = ["가명학번", "참여학년도"]
    panel = approved.groupby(panel_index, dropna=True).agg(
        학년=("학년_정리", _mode_or_first),
        학과=("학과전공", _mode_or_first),
    )
    voluntary_counts = voluntary.groupby(panel_index, dropna=True).size()
    panel["자율건수"] = voluntary_counts.reindex(panel.index, fill_value=0).astype(int)
    panel_reset = panel.reset_index()

    # 학과 진단: 100명 이상 학과만 분석 (기존 EDA 기준)
    excluded_majors = set(merged["단과대학"].dropna().astype(str).unique())
    excluded_majors.update({"기타", "기타(신설)", "미상", "nan", "None"})
    panel_major = panel_reset.loc[~panel_reset["학과"].isin(excluded_majors)].copy()

    dept_base = panel_major.groupby("학과").agg(
        학생수=("가명학번", "nunique"),
        자율0회_비율=("자율건수", lambda s: float((s == 0).mean())),
    )
    dept_base = dept_base.loc[dept_base["학생수"] >= 100].copy()

    all_n = v2.groupby("학과전공").size()
    dept_n = v2.loc[v2["운영주체"].eq("학과·단과대")].groupby("학과전공").size()
    students_all = approved.groupby("학과전공")["가명학번"].nunique()
    students_central = v2.loc[v2["운영주체"].eq("중앙")].groupby("학과전공")["가명학번"].nunique()

    diag = pd.DataFrame(index=dept_base.index)
    diag["학생수"] = dept_base["학생수"]
    diag["자율0회_비율"] = dept_base["자율0회_비율"]
    diag["중앙프로그램_참여율"] = (
        students_central.reindex(diag.index, fill_value=0)
        / students_all.reindex(diag.index).replace(0, np.nan)
    ).fillna(0)
    diag["학과운영_비중"] = (
        dept_n.reindex(diag.index, fill_value=0)
        / all_n.reindex(diag.index).replace(0, np.nan)
    ).fillna(0)

    if not diag.empty:
        x_median = float(diag["중앙프로그램_참여율"].median())
        y_median = float(diag["자율0회_비율"].median())
        x, y = diag["중앙프로그램_참여율"], diag["자율0회_비율"]
        diag["유형"] = np.select(
            [
                (x < x_median) & (y >= y_median),
                (x < x_median) & (y < y_median),
                (x >= x_median) & (y >= y_median),
            ],
            TYPE_ORDER[:3],
            default=TYPE_ORDER[3],
        )
    else:
        x_median, y_median = np.nan, np.nan
        diag["유형"] = pd.Series(dtype=str)

    # 학과별 분야 배수: 학생 기준, 전체 평균=1
    category_pairs = voluntary.loc[
        voluntary["학과전공"].isin(dept_base.index),
        ["학과전공", "가명학번", "카테고리1"],
    ].drop_duplicates()
    if not category_pairs.empty:
        n_by_major = category_pairs.groupby("학과전공")["가명학번"].nunique()
        category_rate = pd.crosstab(category_pairs["학과전공"], category_pairs["카테고리1"]).div(n_by_major, axis=0)
        category_base = category_pairs.groupby("카테고리1")["가명학번"].nunique() / category_pairs["가명학번"].nunique()
        category_lift = category_rate.div(category_base.replace(0, np.nan), axis=1)
    else:
        category_lift = pd.DataFrame(index=dept_base.index)

    # 학생별 이용 경로: 자율 참여 경험이 있는 학생 기준
    # 학과별 경로표(path_result)와 유형별 경로표(type_path_result)를 별도로 만든다.
    # 유형별 그래프에 학과 이름 인덱스를 그대로 reindex하던 오류를 방지한다.
    route_order = ["양쪽", "중앙만", "학과·단과대만"]
    v3 = v2.loc[v2["학과전공"].isin(dept_base.index)].copy()
    if not v3.empty:
        path_flags = pd.crosstab([v3["학과전공"], v3["가명학번"]], v3["운영주체"]) > 0
        for col in ["중앙", "학과·단과대"]:
            if col not in path_flags.columns:
                path_flags[col] = False
        path_flags["이용경로"] = np.select(
            [path_flags["중앙"] & path_flags["학과·단과대"], path_flags["중앙"]],
            ["양쪽", "중앙만"],
            default="학과·단과대만",
        )
        path_flags = path_flags.reset_index()
        path_result = pd.crosstab(path_flags["학과전공"], path_flags["이용경로"], normalize="index")
        path_result = path_result.reindex(columns=route_order, fill_value=0)

        # 한 학생을 한 번만 세어 학과 유형별 이용 경로 비율을 계산한다.
        path_flags["유형"] = path_flags["학과전공"].map(diag["유형"])
        type_path_result = pd.crosstab(
            path_flags["유형"], path_flags["이용경로"], normalize="index"
        ).reindex(index=TYPE_ORDER, columns=route_order, fill_value=0)
    else:
        path_result = pd.DataFrame(index=dept_base.index, columns=route_order, dtype=float)
        type_path_result = pd.DataFrame(index=TYPE_ORDER, columns=route_order, dtype=float)

    year_panel = panel_reset.loc[panel_reset["학과"].isin(dept_base.index)].copy()
    if not year_panel.empty:
        year_panel["학과유형"] = year_panel["학과"].map(diag["유형"])
        year_trend = year_panel.groupby(["참여학년도", "학과유형"])["자율건수"].apply(
            lambda s: float((s == 0).mean())
        ).unstack()
        year_major = year_panel.pivot_table(
            index="학과", columns="참여학년도", values="자율건수",
            aggfunc=lambda s: float((s == 0).mean())
        )
        year_major.columns = [f"{int(c)}학년도" for c in year_major.columns]
    else:
        year_trend = pd.DataFrame()
        year_major = pd.DataFrame(index=dept_base.index)

    # 학과별 진단 카드: 기본 지표 + 분야 배수 + 운영 주체 이용 경로 + 연도별 0회 비율
    card = diag.join(category_lift.add_suffix("_배수"), how="left")
    card = card.join(path_result, how="left")
    card = card.join(year_major, how="left")

    # 분야별 공급(개설 회차)과 수요(자율 승인 기록)
    program_voluntary = programs.loc[
        pd.to_numeric(programs["법정의무_기타플래그"], errors="coerce").eq(0)
    ].copy()
    program_voluntary["카테고리1"] = program_voluntary["카테고리1"].fillna("미분류").astype(str).str.strip()
    supply = program_voluntary.groupby("카테고리1").size().rename("개설회차")
    demand = voluntary.groupby("카테고리1").agg(
        자율승인건수=("가명학번", "size"),
        참여학생수=("가명학번", "nunique"),
        이수율=("이수", "mean"),
    )
    category_summary = pd.concat([supply, demand], axis=1).fillna(0)
    if not category_summary.empty:
        category_summary["공급비중"] = category_summary["개설회차"] / max(category_summary["개설회차"].sum(), 1)
        category_summary["참여비중"] = category_summary["자율승인건수"] / max(category_summary["자율승인건수"].sum(), 1)
        category_summary["회차당승인건수"] = category_summary["자율승인건수"] / category_summary["개설회차"].replace(0, np.nan)

    # 개인별 누적 참여 집중도: 원본 EDA와 동일하게 승인 경험 학생 전체를 분모로 두고
    # 자율 참여가 0건인 학생도 0으로 포함한다.
    students_all = approved["가명학번"].dropna().astype(str).unique()
    student_counts = (
        voluntary.groupby("가명학번").size()
        .reindex(students_all, fill_value=0)
        .sort_values(ascending=False)
    )
    if len(student_counts) and student_counts.sum() > 0:
        top_n = max(1, int(len(student_counts) * 0.10))
        top10_share = float(student_counts.iloc[:top_n].sum() / student_counts.sum())
    else:
        top10_share = np.nan

    # 연속한 학년도 간 참여 구간 전이
    transitions = pd.DataFrame()
    if not panel_reset.empty:
        pp = panel_reset.sort_values(["가명학번", "참여학년도"]).copy()
        pp["구간"] = pd.cut(
            pp["자율건수"], [-1, 0, 2, 9, 10**6],
            labels=["0회", "1~2회", "3~9회", "10회 이상"],
        )
        pp["전년구간"] = pp.groupby("가명학번")["구간"].shift()
        pp["학년도차"] = pp.groupby("가명학번")["참여학년도"].diff()
        consecutive = pp.loc[(pp["학년도차"] == 1) & pp["전년구간"].notna()]
        if not consecutive.empty:
            transitions = pd.crosstab(consecutive["전년구간"], consecutive["구간"], normalize="index")

    # 추천 실험용 데이터: 자율 승인 + 프로그램 그룹
    group_meta_cols = KEY + [c for c in ["프로그램명", "프로그램그룹", "카테고리1", "운영기관", "운영부서", "신청대상_학년", "시작일"] if c in programs.columns]
    vm = voluntary.merge(programs[group_meta_cols], on=KEY, how="left", validate="m:1", suffixes=("", "_프로그램"))
    if "프로그램그룹" not in vm.columns:
        vm["프로그램그룹"] = np.nan
    if "프로그램명" not in vm.columns:
        vm["프로그램명"] = np.nan

    summary = {
        "program_rows_raw": len(programs_raw),
        "program_rows_clean": len(programs),
        "cancelled_sessions": len(cancel_keys),
        "participant_rows_raw": len(participants_raw),
        "participant_rows_clean": len(participants),
        "merged_rows": len(merged),
        "program_info_missing": program_info_missing,
        "approved_rows": int(approved.shape[0]),
        "voluntary_approved_rows": int(voluntary.shape[0]),
        "approved_students": int(approved["가명학번"].nunique()),
        "voluntary_students": int(voluntary["가명학번"].nunique()),
        "voluntary_not_completed": float(1 - voluntary["이수"].mean()) if len(voluntary) else np.nan,
        "top10_share": top10_share,
        "panel_rows": len(panel),
        "department_count": len(diag),
        "central_median": x_median,
        "zero_median": y_median,
    }

    return {
        "programs": programs,
        "merged": merged,
        "approved": approved,
        "voluntary": voluntary,
        "v2": v2,
        "panel": panel_reset,
        "diag": diag,
        "card": card,
        "year_trend": year_trend,
        "path_result": path_result,
        "type_path_result": type_path_result,
        "category_summary": category_summary,
        "transitions": transitions,
        "vm": vm,
        "summary": summary,
    }


# -----------------------------------------------------------------------------
# 공통 표시 함수
# -----------------------------------------------------------------------------
def pct(value: Any, digits: int = 1) -> str:
    try:
        if pd.isna(value):
            return "-"
        return f"{float(value) * 100:.{digits}f}%"
    except (TypeError, ValueError):
        return "-"


def safe_chart(fig: go.Figure, key: str) -> None:
    st.plotly_chart(fig, use_container_width=True, key=key)


def add_download(df: pd.DataFrame, filename: str, label: str) -> None:
    csv = df.to_csv(index=True, encoding="utf-8-sig").encode("utf-8-sig")
    st.download_button(label, data=csv, file_name=filename, mime="text/csv")


# -----------------------------------------------------------------------------
# 추천 오프라인 검증 (개선안.py의 로직 재현)
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner="추천 규칙 오프라인 검증을 계산하고 있습니다...")
def run_recommendation_evaluation(vm: pd.DataFrame) -> dict[str, Any]:
    required = ["가명학번", "참여학년도", "프로그램그룹", "학과전공"]
    if any(c not in vm.columns for c in required):
        raise ValueError("추천 실험에 필요한 열(가명학번, 참여학년도, 프로그램그룹, 학과전공)이 없습니다.")

    all_records = vm.copy()
    all_records["참여학년도"] = pd.to_numeric(all_records["참여학년도"], errors="coerce")
    # 추천 행렬에서는 '없음'/결측 그룹을 제외하되, 과거 이력 유무의 분모는
    # 원래 실험 코드처럼 프로그램그룹 필터링 전 전체 기록으로 계산한다.
    data = all_records.loc[
        all_records["프로그램그룹"].notna()
        & all_records["프로그램그룹"].astype(str).ne("없음")
    ].copy()
    train = data.loc[data["참여학년도"] <= 2024].copy()
    test = data.loc[data["참여학년도"] == 2025].copy()
    if train.empty or test.empty:
        raise ValueError("2022~2024 학습 데이터 또는 2025 평가 데이터가 없습니다.")

    groups = sorted(train["프로그램그룹"].astype(str).unique().tolist())
    users_train = np.array(sorted(train["가명학번"].dropna().astype(str).unique().tolist()), dtype=object)
    if len(groups) == 0 or len(users_train) == 0:
        raise ValueError("추천 후보 프로그램 그룹 또는 학습 학생이 없습니다.")

    ui = {u: i for i, u in enumerate(users_train)}
    gi = {g: i for i, g in enumerate(groups)}

    def matrix_for(frame: pd.DataFrame, users: np.ndarray) -> np.ndarray:
        user_index = {u: i for i, u in enumerate(users)}
        mat = np.zeros((len(users), len(groups)), dtype=np.float32)
        pairs = frame[["가명학번", "프로그램그룹"]].dropna().drop_duplicates()
        pairs["가명학번"] = pairs["가명학번"].astype(str)
        pairs["프로그램그룹"] = pairs["프로그램그룹"].astype(str)
        pairs = pairs.loc[pairs["가명학번"].isin(user_index) & pairs["프로그램그룹"].isin(gi)]
        if not pairs.empty:
            rr = pairs["가명학번"].map(user_index).to_numpy(dtype=int)
            cc = pairs["프로그램그룹"].map(gi).to_numpy(dtype=int)
            mat[rr, cc] = 1.0
        return mat

    x_train = matrix_for(train, users_train)
    test_users = np.array(sorted(set(users_train).intersection(test["가명학번"].dropna().astype(str))), dtype=object)
    if len(test_users) == 0:
        raise ValueError("2025 평가 데이터에서 과거 이력이 있는 학생을 찾지 못했습니다.")

    x_past = x_train[np.searchsorted(users_train, test_users)]
    x_new_all = matrix_for(test, test_users)
    x_new = x_new_all * (1 - x_past)
    keep = x_new.sum(axis=1) > 0
    test_users, x_past, x_new = test_users[keep], x_past[keep], x_new[keep]
    if len(test_users) == 0:
        raise ValueError("2025년에 새롭게 참여한 프로그램 그룹이 있는 평가 학생이 없습니다.")

    popularity = x_train.sum(axis=0)
    score_pop = np.tile(popularity, (len(test_users), 1))

    # 학과별 인기: 학습 이력에서 마지막으로 관측된 학과를 사용하며 학과별 50명 이상일 때 적용
    dept_of = train.sort_values("참여학년도").groupby("가명학번")["학과전공"].last()
    dept_train = dept_of.reindex(users_train).fillna("미상").astype(str).to_numpy()
    dept_test = dept_of.reindex(test_users).fillna("미상").astype(str).to_numpy()
    score_dept = score_pop.copy()
    for major in pd.unique(dept_test):
        mask_train = dept_train == major
        if mask_train.sum() >= 50:
            score_dept[dept_test == major] = x_train[mask_train].mean(axis=0)

    # 함께 참여한 그룹 기반: 코사인 유사도
    co = x_train.T @ x_train
    norms = np.sqrt(np.diag(co))
    denom = np.outer(norms, norms)
    similarity = np.divide(co, denom, out=np.zeros_like(co), where=denom > 0)
    np.fill_diagonal(similarity, 0)
    score_sim = x_past @ similarity

    def hit_vector(scores: np.ndarray, k: int) -> np.ndarray:
        masked = np.where(x_past > 0, -np.inf, scores)
        k_eff = min(k, masked.shape[1])
        top = np.argsort(-masked, axis=1)[:, :k_eff]
        return np.take_along_axis(x_new, top, axis=1).sum(axis=1) > 0

    result_rows = []
    for method, scores in [
        ("① 전체 인기순", score_pop),
        ("② 학과별 인기순", score_dept),
        ("③ 함께 참여한 그룹 기반", score_sim),
    ]:
        result_rows.append({"추천 방식": method, **{f"Hit@{k}": hit_vector(scores, k).mean() * 100 for k in (3, 5, 10)}})

    pop_norm = popularity / popularity.max() if popularity.max() else popularity
    sim_max = score_sim.max(axis=1, keepdims=True)
    sim_norm = np.divide(score_sim, sim_max + 1e-9)
    score_mix = 0.5 * pop_norm[None, :] + 0.5 * sim_norm
    result_rows.append({"추천 방식": "④ 인기+유사도 혼합", **{f"Hit@{k}": hit_vector(score_mix, k).mean() * 100 for k in (3, 5, 10)}})
    result = pd.DataFrame(result_rows)

    # 신규 참여가 상위 인기 그룹에 집중되는 정도
    new_counts = x_new.sum(axis=0)
    order = np.argsort(-popularity)
    concentration = []
    for k in (10, 20, 50):
        kk = min(k, len(groups))
        share = float(new_counts[order[:kk]].sum() / new_counts.sum()) if new_counts.sum() else np.nan
        concentration.append({"인기 상위 그룹 수": kk, "후보 중 비중": kk / len(groups), "2025 신규 참여 비중": share})

    # 과거 참여 그룹 수별 Hit@10 비교
    h_pop = hit_vector(score_pop, 10)
    h_sim = hit_vector(score_sim, 10)
    h_mix = hit_vector(score_mix, 10)
    past_counts = x_past.sum(axis=1)
    band_defs = [(1, 2, "1~2개"), (3, 5, "3~5개"), (6, 10, "6~10개"), (11, np.inf, "11개 이상")]
    bands = []
    for low, high, label in band_defs:
        m = (past_counts >= low) & (past_counts <= high)
        if m.any():
            bands.append({
                "과거 참여 그룹 수": label,
                "학생 수": int(m.sum()),
                "① 인기순": float(h_pop[m].mean() * 100),
                "③ 함께참여": float(h_sim[m].mean() * 100),
                "④ 혼합": float(h_mix[m].mean() * 100),
            })
    band_result = pd.DataFrame(bands)

    all_2025_users = set(all_records.loc[all_records["참여학년도"].eq(2025), "가명학번"].dropna().astype(str).unique())
    all_past_users = set(all_records.loc[all_records["참여학년도"].le(2024), "가명학번"].dropna().astype(str).unique())
    users_with_past = all_2025_users.intersection(all_past_users)
    no_past = len(all_2025_users - users_with_past)

    return {
        "result": result,
        "concentration": pd.DataFrame(concentration),
        "bands": band_result,
        "candidate_count": len(groups),
        "evaluation_students": len(test_users),
        "new_2025_students": len(all_2025_users),
        "students_with_past": len(users_with_past),
        "students_without_past": no_past,
        "groups": groups,
    }


# -----------------------------------------------------------------------------
# 추천 시뮬레이터
# -----------------------------------------------------------------------------
def _target_grade_match(value: Any, grade: str) -> bool:
    if pd.isna(value):
        return True
    text = str(value).strip()
    compact = re.sub(r"\s+", "", text)
    if not compact or compact in {"전체", "전체학년", "전학년", "공통", "nan"}:
        return True
    if "전체" in compact or "전학년" in compact or "1~4" in compact or "1-4" in compact:
        return True
    if grade in compact:
        return True
    # 신청 대상이 학년 숫자만 가진 경우도 보수적으로 처리
    digit = grade[0]
    return bool(re.search(rf"(^|[^0-9]){digit}([^0-9]|$)", text))


def make_simulated_recommendations(
    data: dict[str, Any],
    major: str,
    grade: str,
    interest: str,
    experience: str,
    past_groups: list[str],
    current_route: str,
    top_n: int = 10,
) -> pd.DataFrame:
    """과거 데이터만으로 만든 설명 가능한 추천 예시. 현재 모집상태 판단은 하지 않는다."""
    programs = data["programs"].copy()
    vm = data["vm"].copy()
    diag = data["diag"]

    if "프로그램그룹" not in programs.columns:
        return pd.DataFrame()
    candidates = programs.loc[
        pd.to_numeric(programs["법정의무_기타플래그"], errors="coerce").eq(0)
        & programs["프로그램그룹"].notna()
        & programs["프로그램그룹"].astype(str).ne("없음")
    ].copy()
    if candidates.empty:
        return pd.DataFrame()

    # 후보 그룹별 대표 메타데이터
    for col in ["프로그램명", "카테고리1", "운영기관", "운영부서", "신청대상_학년"]:
        if col not in candidates.columns:
            candidates[col] = np.nan
    if "시작일" in candidates.columns:
        candidates["시작일"] = pd.to_datetime(candidates["시작일"], errors="coerce")
    else:
        candidates["시작일"] = pd.NaT

    meta = candidates.groupby("프로그램그룹").agg(
        대표프로그램명=("프로그램명", _mode_or_first),
        분야=("카테고리1", _mode_or_first),
        운영기관=("운영기관", _mode_or_first),
        운영부서=("운영부서", _mode_or_first),
        신청대상_학년=("신청대상_학년", _mode_or_first),
        최근시작일=("시작일", "max"),
    )
    meta.index = meta.index.astype(str)
    meta["운영기관"] = meta["운영기관"].fillna("").astype(str)
    meta["운영부서"] = meta["운영부서"].fillna("").astype(str)
    is_dept = meta["운영기관"].str.endswith("대학", na=False) | meta["운영부서"].str.contains("학과|학부|전공", na=False)
    meta["운영주체"] = np.where(is_dept, "학과·단과대", "중앙")

    # 과거 데이터에서 학생별 그룹 참여를 집계
    historical = vm.loc[
        vm["프로그램그룹"].notna() & vm["프로그램그룹"].astype(str).ne("없음")
    ].copy()
    historical["프로그램그룹"] = historical["프로그램그룹"].astype(str)
    historical["가명학번"] = historical["가명학번"].astype(str)
    group_pop = historical.groupby("프로그램그룹")["가명학번"].nunique().reindex(meta.index, fill_value=0).astype(float)
    pop_score = group_pop / max(float(group_pop.max()), 1.0)

    # 학과 적합도: 해당 학과 학생 중 해당 그룹에 참여한 비율
    major_hist = historical.loc[historical["학과전공"].astype(str).eq(major)]
    major_student_count = major_hist["가명학번"].nunique()
    if major_student_count:
        major_counts = major_hist.groupby("프로그램그룹")["가명학번"].nunique().reindex(meta.index, fill_value=0)
        major_score = major_counts / major_student_count
        major_score = major_score / max(float(major_score.max()), 1e-9)
    else:
        major_score = pd.Series(0.0, index=meta.index)

    # 관심 분야와 일치 여부
    interest_score = meta["분야"].astype(str).eq(interest).astype(float)

    # 선택한 과거 그룹과 함께 관측된 그룹의 유사도
    sim_score = pd.Series(0.0, index=meta.index)
    if past_groups:
        available_groups = sorted(set(historical["프로그램그룹"].unique()).intersection(meta.index))
        users = sorted(historical["가명학번"].unique())
        user_index = {u: i for i, u in enumerate(users)}
        group_index = {g: i for i, g in enumerate(available_groups)}
        matrix = np.zeros((len(users), len(available_groups)), dtype=np.float32)
        pairs = historical[["가명학번", "프로그램그룹"]].drop_duplicates()
        pairs = pairs.loc[pairs["프로그램그룹"].isin(group_index)]
        if not pairs.empty:
            rr = pairs["가명학번"].map(user_index).to_numpy(dtype=int)
            cc = pairs["프로그램그룹"].map(group_index).to_numpy(dtype=int)
            matrix[rr, cc] = 1.0
        co = matrix.T @ matrix
        norms = np.sqrt(np.diag(co))
        denom = np.outer(norms, norms)
        sim = np.divide(co, denom, out=np.zeros_like(co), where=denom > 0)
        np.fill_diagonal(sim, 0)
        group_to_idx = {g: i for i, g in enumerate(available_groups)}
        valid_past = [g for g in past_groups if g in group_to_idx]
        if valid_past:
            seed = np.zeros(len(available_groups), dtype=np.float32)
            for g in valid_past:
                seed[group_to_idx[g]] = 1.0
            scores = seed @ sim
            sim_score.loc[available_groups] = scores
            if sim_score.max() > 0:
                sim_score = sim_score / sim_score.max()

    # 기존 이용 경로와 반대 운영 주체에 보너스
    cross_score = pd.Series(0.0, index=meta.index)
    if current_route == "학과·단과대 중심":
        cross_score = meta["운영주체"].eq("중앙").astype(float)
    elif current_route == "중앙부서 중심":
        cross_score = meta["운영주체"].eq("학과·단과대").astype(float)

    # 학년 대상 필터는 명시적으로 확인할 수 있는 경우에만 적용
    grade_ok = meta["신청대상_학년"].apply(lambda x: _target_grade_match(x, grade))
    eligible = meta.loc[grade_ok].copy()
    if eligible.empty:
        eligible = meta.copy()
        grade_ok = pd.Series(True, index=meta.index)

    # 이력 단계에 따라 점수 비중을 다르게 둔다. 사전 검증된 최적 가중치는 아님.
    if experience == "이력 없음":
        weights = {"major": 0.45, "pop": 0.35, "interest": 0.20, "similarity": 0.0, "cross": 0.08}
    elif experience == "이력 적음 (1~2개)":
        weights = {"major": 0.35, "pop": 0.25, "interest": 0.20, "similarity": 0.20, "cross": 0.08}
    else:
        weights = {"major": 0.20, "pop": 0.15, "interest": 0.15, "similarity": 0.50, "cross": 0.08}

    score = (
        weights["major"] * major_score
        + weights["pop"] * pop_score
        + weights["interest"] * interest_score
        + weights["similarity"] * sim_score
        + weights["cross"] * cross_score
    )

    # 이미 경험했다고 입력한 그룹은 추천 후보에서 제외
    if past_groups:
        score.loc[score.index.isin(past_groups)] = -1

    # 추천 사유 생성
    reasons = {}
    for group in eligible.index:
        why = []
        if float(major_score.get(group, 0)) > 0:
            why.append("소속 학과의 과거 참여 이력")
        if str(meta.loc[group, "분야"]) == interest:
            why.append(f"관심 분야({interest})와 일치")
        if float(pop_score.get(group, 0)) >= 0.5:
            why.append("전체 참여 이력에서 상대적으로 많이 관측")
        if float(sim_score.get(group, 0)) > 0.2:
            why.append("선택한 과거 그룹과 함께 참여된 이력")
        if float(cross_score.get(group, 0)) > 0:
            why.append("기존과 다른 운영 주체")
        reasons[group] = " · ".join(why) if why else "과거 프로그램 그룹 데이터 기반 후보"

    ranked = score.loc[eligible.index].sort_values(ascending=False)
    ranked = ranked.loc[ranked >= 0].head(top_n)
    result = eligible.loc[ranked.index].copy()
    result.insert(0, "추천점수(시뮬레이션)", ranked.values)
    result["추천이유"] = [reasons.get(g, "과거 데이터 기반 후보") for g in result.index]
    result.index.name = "프로그램그룹"
    result = result.reset_index()
    result["추천점수(시뮬레이션)"] = (result["추천점수(시뮬레이션)"] * 100).round(1)
    return result


# -----------------------------------------------------------------------------
# UI: 업로드
# -----------------------------------------------------------------------------
st.sidebar.markdown("## 🎓 SNOWISE DISCOVER")
st.sidebar.caption("비교과 참여 분석 · 학과 진단 · 추천 시뮬레이션")
st.sidebar.markdown("---")
st.sidebar.markdown("### 1. 데이터 업로드")
program_file = st.sidebar.file_uploader("03_비교과_프로그램.csv", type=["csv"], key="program_csv")
participant_file = st.sidebar.file_uploader("04_비교과_이수자.csv", type=["csv"], key="participant_csv")
st.sidebar.markdown("---")
st.sidebar.markdown("### 분석 기준")
st.sidebar.caption("2022~2025학년도 · 승인 기록 · 법정의무 프로그램 제외")
st.sidebar.caption("학과 유형은 중앙 프로그램 참여율과 자율 0회 비율의 중앙값을 이용한 규칙 기반 분류입니다.")

st.markdown('<div class="main-title">SNOWISE DISCOVER</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">학생과 학과의 참여 특성을 바탕으로 비교과 프로그램의 발견과 연결을 돕는 분석 프로토타입</div>', unsafe_allow_html=True)
st.write("")

if not program_file or not participant_file:
    st.info("시작하려면 왼쪽 사이드바에서 CSV 파일 2개를 업로드해 주세요.")
    st.markdown("#### 필요한 파일")
    st.markdown("- `03_비교과_프로그램.csv`: 프로그램ID, 세부ID, 프로그램명, 카테고리1, 법정의무_기타플래그, 프로그램그룹, 운영기관, 운영부서")
    st.markdown("- `04_비교과_이수자.csv`: 프로그램ID, 세부ID, 가명학번, 참여학년도, 신청상태, 이수여부, 학년, 학과전공")
    st.warning("이 앱은 실제 SNOWISE 시스템에 연결된 서비스가 아닙니다. 과거 CSV를 이용한 분석 및 추천 시뮬레이션입니다.")
    st.stop()

try:
    with st.spinner("데이터를 정리하고 학과 진단 지표를 계산하고 있습니다..."):
        data = prepare_data(program_file.getvalue(), participant_file.getvalue())
except Exception as exc:
    st.error(f"데이터 처리 중 오류가 발생했습니다: {exc}")
    st.stop()

summary = data["summary"]
programs = data["programs"]
voluntary = data["voluntary"]
panel = data["panel"]
diag = data["diag"]

if voluntary.empty:
    st.error("전처리 후 자율 비교과 승인 기록이 없습니다. 법정의무 플래그와 신청상태 값을 확인해 주세요.")
    st.stop()

st.success(
    f"데이터 처리 완료 · 병합 {summary['merged_rows']:,}건 · 자율 승인 {summary['voluntary_approved_rows']:,}건 · 학과 진단 {summary['department_count']}개"
)
if summary.get("program_info_missing", 0) > 0:
    st.warning(
        f"프로그램 정보가 연결되지 않은 신청 기록이 {summary['program_info_missing']:,}건 있습니다. "
        "프로그램ID·세부ID 키와 CSV 전처리 기준을 확인해 주세요."
    )

# -----------------------------------------------------------------------------
# 주요 탭
# -----------------------------------------------------------------------------
tabs = st.tabs([
    "① 전체 현황",
    "② 학과 진단",
    "③ 공급·이용 구조",
    "④ 추천 실험",
    "⑤ 추천 시뮬레이터",
])


# =============================================================================
# ① 전체 현황
# =============================================================================
with tabs[0]:
    st.subheader("비교과 참여 현황 한눈에 보기")
    st.caption("제공된 신청 기록에서 관측된 학생을 기준으로 계산합니다. 신청 기록이 전혀 없는 재학생은 데이터에 포함되지 않습니다.")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("자율 승인 기록", f"{summary['voluntary_approved_rows']:,}건")
    c2.metric("자율 참여 경험 학생", f"{summary['voluntary_students']:,}명")
    c3.metric("승인 대비 미이수", pct(summary["voluntary_not_completed"]))
    c4.metric("학과별 진단 대상", f"{summary['department_count']}개")

    c5, c6, c7 = st.columns(3)
    c5.metric("상위 10% 학생의 참여 비중", pct(summary["top10_share"]))
    c6.metric("관측 학생-학년도", f"{len(panel):,}건")
    observed_zero = float((panel["자율건수"] == 0).mean()) if len(panel) else np.nan
    c7.metric("관측 학생-학년도 중 자율 0회", pct(observed_zero))

    st.markdown(
        '<div class="insight-box"><b>핵심 해석</b><br>비교과 참여 횟수는 일부 학생에게 집중되어 있습니다. 다만 자율 0회는 제공 데이터에서 승인 기록이 관측된 학생-학년도 중 법정의무 프로그램만 승인받은 경우를 뜻하며, 전체 재학생의 미참여율과는 다릅니다.</div>',
        unsafe_allow_html=True,
    )

    left, right = st.columns(2)
    with left:
        st.markdown("##### 학생별 누적 자율 참여 횟수")
        count_by_student = voluntary.groupby("가명학번").size().rename("자율 승인 횟수")
        hist_df = count_by_student.clip(upper=20).reset_index()
        hist_df["구간"] = np.where(hist_df["자율 승인 횟수"] >= 20, "20회 이상", hist_df["자율 승인 횟수"].astype(str))
        hist_order = [str(i) for i in range(1, 20)] + ["20회 이상"]
        h = hist_df["구간"].value_counts().reindex(hist_order, fill_value=0).rename_axis("참여 횟수").reset_index(name="학생 수")
        fig = px.bar(h, x="참여 횟수", y="학생 수", title="학생별 참여 횟수 분포 (20회 이상 묶음)")
        fig.update_layout(xaxis_title="누적 자율 승인 횟수", yaxis_title="학생 수")
        safe_chart(fig, "student_count_distribution")
    with right:
        st.markdown("##### 연속 학년도 참여 구간 유지")
        transitions = data["transitions"]
        if not transitions.empty:
            fig = px.imshow(
                transitions,
                text_auto=".1%",
                aspect="auto",
                color_continuous_scale="Blues",
                labels={"x": "올해 참여 구간", "y": "전년도 참여 구간", "color": "비율"},
                title="전년도 → 올해 참여 구간 이동 (연속 학년도 관측 학생)",
            )
            safe_chart(fig, "participation_transition")
            st.caption("연속한 두 학년도가 모두 관측된 학생-학년도만 포함합니다.")
        else:
            st.info("연속 학년도 참여 이력을 계산할 데이터가 부족합니다.")

    # 연도별 학생-학년도 참여 분포
    st.markdown("##### 학년도별 자율 참여 0회 비율")
    year_zero = panel.groupby("참여학년도")["자율건수"].apply(lambda s: (s == 0).mean() * 100).rename("자율 0회 비율").reset_index()
    if not year_zero.empty:
        fig = px.line(year_zero, x="참여학년도", y="자율 0회 비율", markers=True, text=year_zero["자율 0회 비율"].map(lambda x: f"{x:.1f}%"))
        fig.update_traces(textposition="top center")
        fig.update_layout(xaxis_title="참여학년도", yaxis_title="자율 0회 비율(%)", yaxis_range=[0, 100])
        safe_chart(fig, "year_zero_rate")


# =============================================================================
# ② 학과 진단
# =============================================================================
with tabs[1]:
    st.subheader("학과별 참여 진단")
    st.caption("중앙 프로그램 참여율과 자율 0회 비율의 학과별 중앙값을 기준으로 4개 유형을 구분합니다. 머신러닝 군집화가 아닌 규칙 기반 유형화입니다.")

    if diag.empty:
        st.warning("학생 수 100명 이상인 분석 대상 학과가 없습니다. 학과전공·단과대학 열을 확인해 주세요.")
    else:
        type_summary = diag.groupby("유형").agg(
            학과수=("학생수", "size"),
            학생수=("학생수", "sum"),
            자율0회_평균=("자율0회_비율", "mean"),
            중앙참여율_평균=("중앙프로그램_참여율", "mean"),
        ).reindex(TYPE_ORDER).dropna(how="all").reset_index()

        cols = st.columns(4)
        for i, type_name in enumerate(TYPE_ORDER):
            row = type_summary.loc[type_summary["유형"].eq(type_name)]
            if not row.empty:
                cols[i].metric(TYPE_SHORT[type_name], f"{int(row.iloc[0]['학과수'])}개 학과")
                cols[i].caption(f"자율 0회 평균 {pct(row.iloc[0]['자율0회_평균'])}")

        left, right = st.columns([1.15, 0.85])
        with left:
            st.markdown("##### 학과별 참여 유형 사분면")
            fig = px.scatter(
                diag.reset_index(),
                x="중앙프로그램_참여율", y="자율0회_비율",
                color="유형", hover_name="학과",
                size="학생수", size_max=22,
                category_orders={"유형": TYPE_ORDER},
                labels={"중앙프로그램_참여율": "중앙 프로그램 참여율", "자율0회_비율": "자율 0회 비율", "학생수": "학생 수"},
            )
            fig.add_vline(x=summary["central_median"], line_dash="dash", line_color="gray")
            fig.add_hline(y=summary["zero_median"], line_dash="dash", line_color="gray")
            fig.update_xaxes(tickformat=".0%", range=[0, 1])
            fig.update_yaxes(tickformat=".0%", range=[0, 1])
            fig.update_layout(legend_title="학과 유형")
            safe_chart(fig, "department_quadrant")
        with right:
            st.markdown("##### 유형별 요약")
            display_summary = type_summary.copy()
            display_summary["자율 0회 평균"] = display_summary["자율0회_평균"].map(pct)
            display_summary["중앙 참여율 평균"] = display_summary["중앙참여율_평균"].map(pct)
            st.dataframe(
                display_summary[["유형", "학과수", "학생수", "자율 0회 평균", "중앙 참여율 평균"]],
                hide_index=True, use_container_width=True,
            )
            st.markdown("**유형별 안내 전략**")
            strategy = {
                TYPE_ORDER[0]: "첫 참여가 쉬운 입문 프로그램과 분야 탐색을 안내합니다.",
                TYPE_ORDER[1]: "학과·단과대 프로그램과 중앙부서 프로그램을 연결합니다.",
                TYPE_ORDER[2]: "참여 경험이 적은 학생의 첫 참여와 후속 참여를 지원합니다.",
                TYPE_ORDER[3]: "기존 관심을 바탕으로 심화·확장 프로그램을 제안합니다.",
            }
            for t in TYPE_ORDER:
                st.write(f"**{TYPE_SHORT[t]}** — {strategy[t]}")

        st.markdown("##### 학과별 진단표")
        type_filter = st.multiselect("유형 필터", TYPE_ORDER, default=TYPE_ORDER, key="dept_type_filter")
        dept_table = diag.reset_index().loc[lambda x: x["유형"].isin(type_filter)].copy()
        dept_table["자율 0회 비율"] = dept_table["자율0회_비율"].map(pct)
        dept_table["중앙 프로그램 참여율"] = dept_table["중앙프로그램_참여율"].map(pct)
        dept_table["학과·단과대 운영 비중"] = dept_table["학과운영_비중"].map(pct)
        # 내부 숫자형 열(자율0회_비율)로 먼저 정렬한 뒤 화면 표시용 열만 선택한다.
        # 표시용 열 선택 이후에는 내부 열이 제거되므로 기존 코드처럼 뒤에서 정렬하면 KeyError가 발생한다.
        dept_display = dept_table.sort_values("자율0회_비율", ascending=False)[
            ["학과", "학생수", "유형", "자율 0회 비율", "중앙 프로그램 참여율", "학과·단과대 운영 비중"]
        ]
        st.dataframe(
            dept_display,
            hide_index=True, use_container_width=True,
        )
        add_download(data["card"], "snowise_department_diagnosis_cards.csv", "학과 진단카드 CSV 다운로드")

        st.markdown("##### 학과 상세 진단")
        selected_major = st.selectbox("학과 선택", sorted(diag.index.tolist()), key="selected_major")
        selected = diag.loc[selected_major]
        m1, m2, m3 = st.columns(3)
        m1.metric("학과 유형", TYPE_SHORT.get(selected["유형"], selected["유형"]))
        m2.metric("학생 수", f"{int(selected['학생수']):,}명")
        m3.metric("자율 0회 비율", pct(selected["자율0회_비율"]))

        detail_left, detail_right = st.columns(2)
        with detail_left:
            lift_cols = [c for c in data["card"].columns if c.endswith("_배수")]
            if lift_cols:
                lift_values = data["card"].loc[selected_major, lift_cols].dropna()
                if not lift_values.empty:
                    lift_df = pd.DataFrame({"분야": [c.replace("_배수", "") for c in lift_values.index], "참여 배수": lift_values.values})
                    fig = px.bar(lift_df, x="분야", y="참여 배수", title="분야별 상대 참여 수준 (전체 평균=1.0)")
                    fig.add_hline(y=1, line_dash="dash", line_color="gray")
                    safe_chart(fig, "department_category_lift")
                else:
                    st.info("해당 학과의 분야별 참여 배수를 계산할 값이 없습니다.")
        with detail_right:
            path_row = data["path_result"].reindex([selected_major]).fillna(0)
            if not path_row.empty and path_row.to_numpy().sum() > 0:
                path_plot = path_row.T.reset_index()
                path_plot.columns = ["이용 경로", "학생 비율"]
                path_plot["학생 비율"] *= 100
                fig = px.bar(path_plot, x="이용 경로", y="학생 비율", title="운영 주체 이용 경로 (자율 참여 경험 학생 기준)", text=path_plot["학생 비율"].map(lambda x: f"{x:.1f}%"))
                fig.update_traces(textposition="outside")
                fig.update_layout(yaxis_title="학생 비율(%)")
                safe_chart(fig, "department_path")
            else:
                st.info("해당 학과의 이용 경로를 계산할 데이터가 없습니다.")

        dept_year_cols = [c for c in data["card"].columns if c.endswith("학년도")]
        if dept_year_cols:
            dept_year = data["card"].loc[selected_major, dept_year_cols].dropna()
            if not dept_year.empty:
                year_df = pd.DataFrame({
                    "학년도": [int(c[:4]) for c in dept_year.index],
                    "자율 0회 비율": dept_year.values * 100,
                })
                fig = px.line(year_df, x="학년도", y="자율 0회 비율", markers=True,
                              title=f"{selected_major} 학년도별 자율 0회 비율")
                fig.update_layout(yaxis_title="자율 0회 비율(%)")
                safe_chart(fig, "department_individual_year_trend")

        st.markdown("##### 유형별 학년도 변화")
        trend = data["year_trend"]
        if not trend.empty:
            trend_pct = trend * 100
            trend_long = trend_pct.reset_index().melt(id_vars="참여학년도", var_name="학과 유형", value_name="자율 0회 비율")
            trend_long["학과 유형"] = pd.Categorical(trend_long["학과 유형"], TYPE_ORDER, ordered=True)
            fig = px.line(trend_long, x="참여학년도", y="자율 0회 비율", color="학과 유형", markers=True, category_orders={"학과 유형": TYPE_ORDER})
            fig.update_layout(yaxis_title="자율 0회 비율(%)")
            safe_chart(fig, "type_year_trend")
        else:
            st.info("학년도별 추세를 계산할 데이터가 없습니다.")


# =============================================================================
# ③ 공급·이용 구조
# =============================================================================
with tabs[2]:
    st.subheader("프로그램 공급과 학생 이용 구조")
    cat = data["category_summary"].copy()
    if cat.empty:
        st.warning("분야별 공급·참여를 집계할 데이터가 없습니다.")
    else:
        c1 = float(cat.loc[cat.index.isin(["학습", "취업"]), "공급비중"].sum())
        c2 = float(cat.loc[cat.index.isin(["학습", "취업"]), "참여비중"].sum())
        k1, k2, k3 = st.columns(3)
        k1.metric("학습·취업 공급 비중", pct(c1))
        k2.metric("학습·취업 자율 승인 비중", pct(c2))
        k3.metric("두 비중의 차이", f"{(c2-c1)*100:+.1f}%p")

        chart_df = cat.reset_index().rename(columns={"카테고리1": "분야"})
        chart_long = chart_df.melt(
            id_vars="분야", value_vars=["공급비중", "참여비중"],
            var_name="구분", value_name="비중",
        )
        chart_long["비중"] *= 100
        fig = px.bar(
            chart_long, x="분야", y="비중", color="구분", barmode="group",
            text=chart_long["비중"].map(lambda x: f"{x:.1f}%"),
            title="분야별 프로그램 공급 비중과 자율 승인 비중 비교",
        )
        fig.update_traces(textposition="outside")
        fig.update_layout(yaxis_title="비중(%)")
        safe_chart(fig, "supply_demand_comparison")

        st.markdown("##### 분야별 공급·참여 집계")
        show_cat = cat.reset_index().rename(columns={"카테고리1": "분야"})
        for col in ["공급비중", "참여비중", "이수율"]:
            if col in show_cat:
                show_cat[col] = show_cat[col].map(pct)
        st.dataframe(show_cat, hide_index=True, use_container_width=True)

        st.markdown(
            '<div class="insight-box"><b>해석할 때 주의</b><br>공급 비중은 개설 회차 기준이고 참여 비중은 승인 기록 기준입니다. 두 값의 분모가 다르므로 학생 선호를 직접 측정한 비교는 아닙니다. 회차당 승인 건수가 높아도 정원·시간대·홍보량을 알 수 없어 공급 부족이라고 단정할 수 없습니다.</div>',
            unsafe_allow_html=True,
        )

    st.markdown("##### 학과 유형별 운영 주체 이용 경로")
    type_path_result = data.get("type_path_result", pd.DataFrame())
    if not type_path_result.empty:
        route_order = ["양쪽", "중앙만", "학과·단과대만"]
        pr = type_path_result.reindex(index=TYPE_ORDER, columns=route_order, fill_value=0).fillna(0)
        # index name이 어떤 pandas 버전에서도 확실히 '유형'이 되도록 지정한다.
        pr.index.name = "유형"
        pr_plot = pr.reset_index().melt(
            id_vars=["유형"],
            value_vars=route_order,
            var_name="이용 경로",
            value_name="학생 비율",
        )
        pr_plot["학생 비율"] *= 100
        fig = px.bar(
            pr_plot, x="유형", y="학생 비율", color="이용 경로", barmode="stack",
            category_orders={"유형": TYPE_ORDER, "이용 경로": route_order},
            title="학과 유형별 이용 경로 (자율 참여 경험 학생 기준)",
        )
        fig.update_layout(yaxis_title="학생 비율(%)", yaxis_range=[0, 100])
        safe_chart(fig, "type_route_comparison")
        st.caption("이용 경로 분석은 2022~2025 관측 기간에 자율 비교과 승인 경험이 있는 학생 기준입니다. 자율 0회 학생은 포함되지 않습니다.")
    else:
        st.info("학과 유형별 이용 경로를 계산할 수 있는 자율 참여 경험 학생이 없습니다.")


# =============================================================================
# ④ 추천 실험
# =============================================================================
with tabs[3]:
    st.subheader("추천 규칙 오프라인 검증")
    st.caption("2022~2024학년도 참여 이력으로 규칙을 만들고, 2025학년도 신규 프로그램 그룹 참여를 상위 K개 추천에 포함했는지 평가합니다.")

    if st.button("추천 실험 계산하기", type="primary", key="run_recommendation"):
        try:
            rec_eval = run_recommendation_evaluation(data["vm"])
            st.session_state["rec_eval_result"] = rec_eval
        except Exception as exc:
            st.error(f"추천 실험 계산에 실패했습니다: {exc}")

    rec_eval = st.session_state.get("rec_eval_result")
    if rec_eval is None:
        st.info("버튼을 누르면 추천 규칙별 Hit@K와 참여 이력 구간별 결과를 계산합니다. 데이터 규모에 따라 계산에 시간이 걸릴 수 있습니다.")
    else:
        r1, r2, r3, r4 = st.columns(4)
        r1.metric("추천 후보 그룹", f"{rec_eval['candidate_count']:,}개")
        r2.metric("평가 대상 학생", f"{rec_eval['evaluation_students']:,}명")
        r3.metric("2025 참여 학생", f"{rec_eval['new_2025_students']:,}명")
        r4.metric("과거 이력 없음", f"{rec_eval['students_without_past']:,}명")

        results = rec_eval["result"]
        result_long = results.melt(id_vars="추천 방식", value_vars=["Hit@3", "Hit@5", "Hit@10"], var_name="추천 범위", value_name="Hit 비율")
        fig = px.bar(result_long, x="추천 방식", y="Hit 비율", color="추천 범위", barmode="group", text=result_long["Hit 비율"].map(lambda x: f"{x:.1f}%"), title="추천 규칙별 Hit@K 비교")
        fig.update_traces(textposition="outside")
        fig.update_layout(yaxis_title="Hit 비율(%)", xaxis_title="추천 규칙")
        safe_chart(fig, "hit_comparison")
        st.dataframe(results.round(1), hide_index=True, use_container_width=True)

        st.markdown("##### 인기 프로그램 그룹으로의 신규 참여 집중")
        concentration = rec_eval["concentration"].copy()
        concentration["후보 중 비중"] = concentration["후보 중 비중"].map(pct)
        concentration["2025 신규 참여 비중"] = concentration["2025 신규 참여 비중"].map(pct)
        st.dataframe(concentration, hide_index=True, use_container_width=True)

        st.markdown("##### 과거 참여 그룹 수별 Hit@10")
        bands = rec_eval["bands"]
        if not bands.empty:
            st.dataframe(bands.round(1), hide_index=True, use_container_width=True)
            band_long = bands.melt(id_vars=["과거 참여 그룹 수", "학생 수"], value_vars=["① 인기순", "③ 함께참여", "④ 혼합"], var_name="추천 방식", value_name="Hit@10")
            fig = px.bar(band_long, x="과거 참여 그룹 수", y="Hit@10", color="추천 방식", barmode="group", title="과거 이력 구간에 따른 Hit@10")
            fig.update_layout(yaxis_title="Hit@10(%)")
            safe_chart(fig, "hit_by_history")

        st.markdown(
            '<div class="insight-box"><b>결과 해석</b><br>Hit@K는 추천 상위 K개 중 2025년에 실제 신규 참여한 프로그램 그룹이 하나 이상 포함된 학생의 비율입니다. 실제 추천 노출·클릭·신청 전환율이 아닙니다. 혼합 비중은 0.5:0.5 한 가지 설정만 시험했으므로 개인화 추천의 일반적 우위를 입증하지 않습니다.</div>',
            unsafe_allow_html=True,
        )
        st.caption("추천 후보는 프로그램그룹 단위입니다. 실제 서비스의 추천 로그와 비교한 결과가 아닌 과거 승인 이력 기반 오프라인 검증입니다.")


# =============================================================================
# ⑤ 추천 시뮬레이터
# =============================================================================
with tabs[4]:
    st.subheader("맞춤형 프로그램 추천 시뮬레이터")
    st.caption("학생 조건을 바꾸면 과거 참여 데이터에 근거한 추천 후보가 달라지는 모습을 시연합니다.")

    if "프로그램그룹" not in programs.columns or programs["프로그램그룹"].dropna().empty:
        st.warning("프로그램그룹 정보가 없어 추천 시뮬레이터를 사용할 수 없습니다.")
    elif diag.empty:
        st.warning("학과 진단 결과가 없어 추천 시뮬레이터를 사용할 수 없습니다.")
    else:
        major_options = sorted(diag.index.astype(str).tolist())
        left, right = st.columns(2)
        with left:
            sim_major = st.selectbox("소속 학과", major_options, key="sim_major")
            sim_grade = st.selectbox("학년", VALID_GRADES, key="sim_grade")
            sim_interest = st.selectbox("관심 분야", CATEGORY_ORDER, key="sim_interest")
        with right:
            experience = st.selectbox(
                "참여 이력 단계",
                ["이력 없음", "이력 적음 (1~2개)", "이력 많음 (11개 이상)"],
                key="sim_experience",
            )
            current_route = st.selectbox(
                "기존 이용 경로",
                ["이용 경험 없음", "학과·단과대 중심", "중앙부서 중심", "양쪽 모두 이용"],
                key="sim_route",
            )
            top_n = st.slider("추천 후보 수", min_value=5, max_value=15, value=10, key="sim_top_n")

        group_options = sorted(
            programs.loc[
                pd.to_numeric(programs["법정의무_기타플래그"], errors="coerce").eq(0)
                & programs["프로그램그룹"].notna()
                & programs["프로그램그룹"].astype(str).ne("없음"),
                "프로그램그룹",
            ].astype(str).unique().tolist()
        )
        past_groups = []
        if experience != "이력 없음":
            past_groups = st.multiselect(
                "과거에 참여했다고 가정할 프로그램 그룹",
                options=group_options,
                help="개인 식별정보 없이 추천 규칙을 시연하기 위한 가상 입력입니다. 실제 학생의 이력은 아닙니다.",
                key="sim_past_groups",
            )
            if experience.startswith("이력 적음") and len(past_groups) > 2:
                st.warning("이력 적음 구간은 1~2개 그룹을 선택하는 시나리오입니다. 앞의 2개만 사용합니다.")
                past_groups = past_groups[:2]
            if experience.startswith("이력 많음") and len(past_groups) < 3:
                st.info("이력 많음 시나리오를 더 잘 보여주려면 과거 그룹을 3개 이상 선택해 보세요. 최종 제안서의 11개 이상 기준을 그대로 재현하려면 11개 이상 입력이 필요합니다.")

        if st.button("맞춤형 추천 결과 보기", type="primary", key="run_simulator"):
            try:
                recs = make_simulated_recommendations(
                    data=data,
                    major=sim_major,
                    grade=sim_grade,
                    interest=sim_interest,
                    experience=experience,
                    past_groups=past_groups,
                    current_route=current_route,
                    top_n=top_n,
                )
                st.session_state["sim_recommendations"] = recs
                st.session_state["sim_context"] = {
                    "major": sim_major, "grade": sim_grade, "interest": sim_interest,
                    "experience": experience, "route": current_route,
                }
            except Exception as exc:
                st.error(f"추천 시뮬레이션 중 오류가 발생했습니다: {exc}")

        recs = st.session_state.get("sim_recommendations")
        context = st.session_state.get("sim_context")
        if recs is not None and context is not None:
            st.markdown("---")
            st.markdown(
                f"**추천 시나리오:** {context['major']} · {context['grade']} · {context['interest']} 관심 · {context['experience']}"
            )
            st.info("아래는 과거 2022~2025 참여 이력으로 만든 프로그램 그룹 추천 예시입니다. 현재 모집 중인 회차나 신청 가능 여부를 뜻하지 않습니다.")
            if recs.empty:
                st.warning("조건에 맞는 추천 후보가 없습니다. 관심 분야나 학년 조건을 바꾸어 보세요.")
            else:
                display_recs = recs.copy()
                if "최근시작일" in display_recs.columns:
                    display_recs["최근시작일"] = pd.to_datetime(display_recs["최근시작일"], errors="coerce").dt.strftime("%Y-%m-%d")
                show_cols = [c for c in [
                    "프로그램그룹", "대표프로그램명", "분야", "운영주체", "운영기관", "운영부서",
                    "신청대상_학년", "추천점수(시뮬레이션)", "추천이유", "최근시작일",
                ] if c in display_recs.columns]
                st.dataframe(display_recs[show_cols], hide_index=True, use_container_width=True)
                add_download(display_recs.set_index("프로그램그룹"), "snowise_simulated_recommendations.csv", "추천 결과 CSV 다운로드")

        st.markdown("---")
        st.markdown("##### 실제 서비스 적용 시 추가로 필요한 데이터")
        st.markdown(
            "- 실시간 모집 시작일·마감일·정원·모집 상태\n"
            "- 학생이 직접 입력한 관심 키워드와 성향 진단 정보\n"
            "- 추천 노출·클릭·신청·이수 로그\n"
            "- 학생이 추천을 숨기거나 관심 없다고 표시한 피드백"
        )
        st.caption("추천 점수의 가중치는 시연용 휴리스틱이며, 검증된 최적 알고리즘이나 실제 SNOWISE 개인 추천 결과가 아닙니다.")


# -----------------------------------------------------------------------------
# 푸터 및 데이터 한계
# -----------------------------------------------------------------------------
st.markdown("---")
st.caption(
    "분석 기준: 2022~2025학년도 제공 CSV · 신청상태 승인 · 법정의무 프로그램 제외. "
    "자율 0회는 관측된 승인 학생-학년도 중 자율 승인 기록이 없는 경우입니다. "
    "신청 기록이 전혀 없는 재학생은 제공 데이터에서 확인할 수 없습니다. "
    "추천 시뮬레이터는 과거 데이터 기반 프로토타입이며 현재 모집 상태와 실제 서비스 효과를 검증하지 않습니다."
)
