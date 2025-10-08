import streamlit as st
import requests
import json
import time

# 페이지 설정
st.set_page_config(
    page_title="GroundingDINO Search",
    page_icon="🔍",
    layout="wide"
)

# 제목
st.title("🔍 GroundingDINO Object Detection")
st.markdown("---")

# 사이드바에 설정 옵션
with st.sidebar:
    st.header("설정")
    
    # 서버 URL 설정
    server_url = st.text_input(
        "백엔드 서버 URL",
        value="http://localhost:8000",
        help="백엔드 API 서버의 URL을 입력하세요"
    )
    
    # 임계값 설정
    box_threshold = st.slider(
        "Box Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.35,
        step=0.05
    )
    
    text_threshold = st.slider(
        "Text Threshold", 
        min_value=0.0,
        max_value=1.0,
        value=0.25,
        step=0.05
    )

# 메인 컨텐츠
col1, col2 = st.columns([1, 1])

with col1:
    st.header("검색 입력")
    
    # 검색 입력
    query = st.text_input(
        "검색할 객체",
        placeholder="예: person . car . dog",
        help="찾고 싶은 객체들을 입력하세요. 여러 객체는 점(.)으로 구분합니다."
    )
    
    # 이미지 업로드
    uploaded_file = st.file_uploader(
        "이미지 업로드",
        type=['jpg', 'jpeg', 'png'],
        help="분석할 이미지를 업로드하세요"
    )
    
    # 검색 버튼
    search_button = st.button(
        "🔍 검색하기",
        type="primary",
        disabled=not (query and uploaded_file),
        use_container_width=True
    )

with col2:
    st.header("검색 결과")
    
    # 결과 표시 영역
    result_container = st.empty()

# 검색 실행
if search_button and query and uploaded_file:
    with st.spinner("검색 중입니다..."):
        try:
            # 이미지를 base64로 인코딩하거나 파일로 전송
            files = {"image": uploaded_file.getvalue()}
            data = {
                "query": query,
                "box_threshold": box_threshold,
                "text_threshold": text_threshold
            }
            
            # API 호출
            response = requests.post(
                f"{server_url}/search",
                files=files,
                data=data,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                
                with result_container.container():
                    st.success("검색이 완료되었습니다!")
                    
                    # 결과 표시
                    st.subheader("검색 결과")
                    st.json(result)
                    
                    # 결과 이미지가 있다면 표시
                    if "annotated_image" in result:
                        st.subheader("검출 결과 이미지")
                        st.image(result["annotated_image"])
                    
                    # 검출된 객체 목록
                    if "detections" in result:
                        st.subheader("검출된 객체")
                        for i, detection in enumerate(result["detections"]):
                            with st.expander(f"객체 {i+1}: {detection.get('label', 'Unknown')}"):
                                st.write(f"**신뢰도:** {detection.get('confidence', 0):.2f}")
                                st.write(f"**위치:** {detection.get('bbox', [])}")
            else:
                with result_container.container():
                    st.error(f"서버 오류: {response.status_code}")
                    st.text(response.text)
                    
        except requests.exceptions.ConnectionError:
            with result_container.container():
                st.error("서버에 연결할 수 없습니다. 서버가 실행 중인지 확인해주세요.")
        except requests.exceptions.Timeout:
            with result_container.container():
                st.error("요청 시간이 초과되었습니다.")
        except Exception as e:
            with result_container.container():
                st.error(f"오류가 발생했습니다: {str(e)}")

# 사용법 안내
with st.expander("📖 사용법"):
    st.markdown("""
    ### 사용 방법
    1. **검색할 객체 입력**: 찾고 싶은 객체 이름을 입력하세요
       - 여러 객체를 찾으려면 점(.)으로 구분하세요
       - 예: `person . car . dog`
    
    2. **이미지 업로드**: 분석할 이미지를 업로드하세요
       - 지원 형식: JPG, JPEG, PNG
    
    3. **임계값 조정**: 사이드바에서 검출 민감도를 조정할 수 있습니다
       - Box Threshold: 객체 검출 임계값
       - Text Threshold: 텍스트 매칭 임계값
    
    4. **검색 실행**: 모든 설정이 완료되면 검색 버튼을 클릭하세요
    
    ### 백엔드 서버 요구사항
    - FastAPI 또는 Flask 서버가 실행 중이어야 합니다
    - `/search` 엔드포인트가 구현되어 있어야 합니다
    """)

# 실시간 상태 표시
st.markdown("---")
status_col1, status_col2, status_col3 = st.columns(3)

with status_col1:
    if query:
        st.success("✅ 검색어 입력됨")
    else:
        st.warning("⚠️ 검색어를 입력하세요")

with status_col2:
    if uploaded_file:
        st.success("✅ 이미지 업로드됨")
    else:
        st.warning("⚠️ 이미지를 업로드하세요")

with status_col3:
    try:
        # 서버 상태 확인
        response = requests.get(f"{server_url}/health", timeout=2)
        if response.status_code == 200:
            st.success("✅ 서버 연결됨")
        else:
            st.error("❌ 서버 오류")
    except:
        st.error("❌ 서버 연결 실패")
