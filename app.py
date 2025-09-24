import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import plotly.graph_objects as go
import time

st.set_page_config(
    page_title="Sentiment Analysis App",
    layout="centered"
)

st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    .sentiment-positive {
        background: linear-gradient(90deg, #4CAF50, #8BC34A);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
        margin: 1rem 0;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    
    .sentiment-negative {
        background: linear-gradient(90deg, #F44336, #FF5722);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
        margin: 1rem 0;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    
    .prediction-score {
        font-size: 1.2rem;
        color: #333;
        text-align: center;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Cache the model and tokenizer loading
@st.cache_resource
def load_model_and_tokenizer():
    """Tải mô hình LSTM và tokenizer đã huấn luyện"""
    try:
        # Load the trained model
        model = load_model("lstm_text_classifier.h5")
        
        # Load the tokenizer
        with open("tokenizer.pkl", "rb") as f:
            tokenizer = pickle.load(f)
        
        return model, tokenizer
    except Exception as e:
        st.error(f"Lỗi khi tải mô hình hoặc tokenizer: {str(e)}")
        return None, None

def predict_sentiment(text, model, tokenizer, max_length=100):
    """Dự đoán cảm xúc cho văn bản đầu vào"""
    try:
        # Convert text to sequence
        sequence = tokenizer.texts_to_sequences([text])
        
        # Pad sequence
        padded_sequence = pad_sequences(sequence, maxlen=max_length, padding='post', truncating='post')
        
        # Make prediction
        prediction = model.predict(padded_sequence, verbose=0)
        
        return prediction[0][0]
    except Exception as e:
        st.error(f"Lỗi khi thực hiện dự đoán: {str(e)}")
        return None

def create_sentiment_gauge(score):
    """Tạo biểu đồ đồng hồ đẹp cho điểm cảm xúc"""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = score * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Điểm Cảm xúc"},
        delta = {'reference': 50},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 50], 'color': "lightcoral"},
                {'range': [50, 100], 'color': "lightgreen"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    fig.update_layout(height=300)
    return fig

# Main app
def main():
    # Header
    st.markdown('<h1 class="main-header">Ứng dụng Phân tích Cảm xúc</h1>', unsafe_allow_html=True)
    
    # Load model and tokenizer
    with st.spinner("Đang tải mô hình AI..."):
        model, tokenizer = load_model_and_tokenizer()
    
    if model is None or tokenizer is None:
        st.error("Không thể tải mô hình. Vui lòng kiểm tra các tệp có tồn tại và hợp lệ không.")
        return
    
    st.success("Mô hình đã được tải thành công!")
    
    # Main content area
    st.header("Nhập văn bản để phân tích")
    
    # Text input
    user_text = st.text_area(
        "Nhập hoặc dán văn bản của bạn vào đây:",
        height=150,
        placeholder="ví dụ: Bộ phim này thật tuyệt vời! Tôi yêu từng khoảnh khắc của nó.",
        help="Nhập bất kỳ văn bản nào bạn muốn phân tích cảm xúc (tích cực hoặc tiêu cực)"
    )
    
    # Analyze button
    analyze_button = st.button("Phân tích Cảm xúc", type="primary", use_container_width=True)
    
    st.header("Văn bản mẫu")
    
    example_texts = [
        "Bộ phim này thật tuyệt vời! Tôi yêu từng khoảnh khắc của nó.",
        "Dịch vụ rất tệ và đồ ăn thật khủng khiếp.",
        "Hôm nay tôi cảm thấy rất tuyệt, mọi thứ đều ổn!",
        "Đây là trải nghiệm tệ nhất mà tôi từng có.",
        "Thời tiết hôm nay rất đẹp.",
        "Tôi ghét phải chờ đợi trong hàng dài.",
        "Cuốn sách này đã thay đổi cuộc đời tôi theo hướng tích cực.",
        "Ứng dụng liên tục bị lỗi và rất khó chịu."
    ]
    
    cols = st.columns(2)
    for i, example in enumerate(example_texts):
        col_idx = i % 2
        with cols[col_idx]:
            if st.button(f"Thử mẫu {i+1}", key=f"example_{i}", use_container_width=True):
                st.session_state.selected_example = example
    
    # Use selected example if available
    if hasattr(st.session_state, 'selected_example'):
        user_text = st.session_state.selected_example
        st.text_area("Mẫu đã chọn:", value=user_text, height=100, disabled=True)
        analyze_button = True  # Auto-analyze when example is selected
        del st.session_state.selected_example  # Clear the selection
    
    # Analysis results
    if analyze_button and user_text.strip():
        with st.spinner("Đang phân tích cảm xúc..."):
            # Add a small delay for better UX
            time.sleep(0.5)
            
            # Get prediction
            prediction_score = predict_sentiment(user_text, model, tokenizer)
            
            if prediction_score is not None:
                # Determine sentiment
                is_positive = prediction_score > 0.5
                sentiment_label = "Tích cực" if is_positive else "Tiêu cực"
                confidence = prediction_score if is_positive else (1 - prediction_score)
                
                # Display results
                st.header("Kết quả Phân tích")
                
                # Show sentiment with styling
                if is_positive:
                    st.markdown(f'<div class="sentiment-positive">Cảm xúc: {sentiment_label}</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="sentiment-negative">Cảm xúc: {sentiment_label}</div>', unsafe_allow_html=True)
                
                # Show prediction score
                st.markdown(f'<div class="prediction-score">Điểm dự đoán: {prediction_score:.4f}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="prediction-score">Độ tin cậy: {confidence:.2%}</div>', unsafe_allow_html=True)
                
                # Create columns for metrics and chart
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric(
                        label="Phân loại Cảm xúc",
                        value=sentiment_label,
                        delta=f"{confidence:.2%} tin cậy"
                    )
                
                with col2:
                    # Show gauge chart
                    fig = create_sentiment_gauge(prediction_score)
                    st.plotly_chart(fig, use_container_width=True)
                
                # Show interpretation
                st.header("Giải thích")
                if is_positive:
                    st.success(f"Văn bản thể hiện **cảm xúc tích cực** với độ tin cậy {confidence:.2%}. Mô hình phát hiện cảm xúc lạc quan, vui vẻ hoặc thuận lợi trong văn bản.")
                else:
                    st.error(f"Văn bản thể hiện **cảm xúc tiêu cực** với độ tin cậy {confidence:.2%}. Mô hình phát hiện cảm xúc bi quan, buồn bã hoặc bất lợi trong văn bản.")
                
                # Technical details in expander
                with st.expander("Chi tiết Kỹ thuật"):
                    st.write(f"**Điểm dự đoán thô:** {prediction_score:.6f}")
                    st.write(f"**Ngưỡng quyết định:** 0.5")
                    st.write(f"**Độ dài văn bản:** {len(user_text)} ký tự")
                    st.write(f"**Số từ:** {len(user_text.split())} từ")
    
    elif analyze_button and not user_text.strip():
        st.warning("Vui lòng nhập một số văn bản để phân tích!")
    
    # Footer
    st.markdown("---")
    st.markdown("Được xây dựng bằng Streamlit và TensorFlow | Mạng Neural LSTM để Phân tích Cảm xúc")

if __name__ == "__main__":
    main()
