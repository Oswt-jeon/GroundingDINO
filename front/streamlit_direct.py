import streamlit as st
import numpy as np
from PIL import Image
import cv2

# 이 방법은 Streamlit 앱 내에서 직접 GroundingDINO를 실행합니다
# 별도의 백엔드 서버가 필요하지 않습니다

st.set_page_config(
    page_title="GroundingDINO Direct",
    page_icon="🎯",
    layout="wide"
)

st.title("🎯 GroundingDINO Direct Processing")

# 세션 상태로 모델 캐싱
@st.cache_resource
def load_grounding_dino_model():
    """
    GroundingDINO 모델을 로드합니다 (한 번만 로드)
    """
    try:
        # TODO: 실제 모델 로딩 코드
        # from groundingdino.util.inference import load_model
        # model = load_model("config_path", "checkpoint_path")
        # return model
        
        # 현재는 더미 모델 반환
        st.success("✅ GroundingDINO 모델이 로드되었습니다")
        return "dummy_model"
    except Exception as e:
        st.error(f"❌ 모델 로딩 실패: {str(e)}")
        return None

# 모델 로드
model = load_grounding_dino_model()

if model:
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("입력")
        
        # 텍스트 입력
        text_prompt = st.text_area(
            "검색할 객체",
            value="person . car . dog",
            help="찾고 싶은 객체들을 입력하세요"
        )
        
        # 이미지 업로드
        uploaded_file = st.file_uploader(
            "이미지 선택",
            type=['jpg', 'jpeg', 'png']
        )
        
        # 설정
        with st.expander("고급 설정"):
            box_threshold = st.slider("Box Threshold", 0.0, 1.0, 0.35)
            text_threshold = st.slider("Text Threshold", 0.0, 1.0, 0.25)
        
        # 처리 버튼
        if st.button("🔍 객체 검출", type="primary", disabled=not uploaded_file):
            if uploaded_file and text_prompt:
                with st.spinner("처리 중..."):
                    # 이미지 처리
                    image = Image.open(uploaded_file)
                    image_array = np.array(image)
                    
                    # TODO: 실제 GroundingDINO 추론
                    # from groundingdino.util.inference import predict, annotate
                    # boxes, logits, phrases = predict(
                    #     model=model,
                    #     image=image_array,
                    #     caption=text_prompt,
                    #     box_threshold=box_threshold,
                    #     text_threshold=text_threshold
                    # )
                    # annotated_image = annotate(image_array, boxes, logits, phrases)
                    
                    # 더미 결과
                    st.session_state['results'] = {
                        'original_image': image,
                        'text_prompt': text_prompt,
                        'detections': [
                            {'label': 'person', 'confidence': 0.95, 'bbox': [100, 100, 200, 300]},
                            {'label': 'car', 'confidence': 0.87, 'bbox': [300, 200, 500, 400]}
                        ]
                    }
                    
                    st.success("검출 완료!")
    
    with col2:
        st.header("결과")
        
        if 'results' in st.session_state:
            results = st.session_state['results']
            
            # 원본 이미지 표시
            st.subheader("원본 이미지")
            st.image(results['original_image'], use_column_width=True)
            
            # 검출 결과
            st.subheader("검출된 객체")
            for i, detection in enumerate(results['detections']):
                with st.container():
                    col_a, col_b, col_c = st.columns([2, 1, 1])
                    with col_a:
                        st.write(f"**{detection['label']}**")
                    with col_b:
                        st.write(f"신뢰도: {detection['confidence']:.2f}")
                    with col_c:
                        st.write(f"위치: {detection['bbox']}")
        else:
            st.info("이미지를 업로드하고 검출을 실행하세요.")

else:
    st.error("모델을 로드할 수 없습니다. 환경을 확인해주세요.")
