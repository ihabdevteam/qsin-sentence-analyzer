import streamlit as st
from supabase import create_client, Client
import pandas as pd

@st.cache_resource
def init_supabase_client():
    """Supabase 클라이언트를 초기화하고 반환합니다."""
    try:
        supabase_url = st.secrets["SUPABASE_URL"]
        supabase_key = st.secrets["SUPABASE_KEY"]
        return create_client(supabase_url, supabase_key)
    except KeyError:
        st.error("Supabase 연결 정보를 찾을 수 없습니다. `.streamlit/secrets.toml` 파일을 확인해주세요.")
        return None

@st.cache_data(ttl=300)
def get_profiles(_supabase: Client):
    """profiles 테이블에서 담당자 목록을 가져옵니다."""
    try:
        response = _supabase.table('profiles').select('user_id, tester_name').not_.is_('tester_name', 'null').execute()
        return response.data
    except Exception as e:
        st.error(f"담당자 목록을 불러오는 중 오류 발생: {e}")
        return []

@st.cache_data(ttl=300)
def get_patients(_supabase: Client):
    """patient_user 테이블에서 환자 목록을 가져옵니다."""
    try:
        response = _supabase.table('patient_user').select('id, name').execute()
        return response.data
    except Exception as e:
        st.error(f"환자 목록을 불러오는 중 오류 발생: {e}")
        return []

