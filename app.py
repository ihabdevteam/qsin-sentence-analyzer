import streamlit as st

st.set_page_config(
    page_title="Quick-SIN 데이터 관리",
    page_icon="🎧",
    layout="wide"
)

st.title("Quick-SIN 데이터 관리 및 분석 대시보드")

st.markdown(
    """   
    **👈 왼쪽 사이드바에서 원하는 페이지를 선택하세요.**

    ### 제공되는 페이지
    - **테스트 리포트 생성기**: 새로운 더미 테스트 데이터를 생성하고 데이터베이스에 저장합니다.
    - **점수 분석 페이지**: 저장된 데이터를 기반으로 문장별 SNR-50을 추정하고 시각적으로 분석합니다.

    """
)
