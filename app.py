import streamlit as st
import time
from model_utils import load_model_and_tokenizer, predict_sentiment, create_sentiment_gauge
from ui_components import (
    render_text_input, render_example_buttons, render_sentiment_result,
    render_prediction_scores, render_interpretation, render_technical_details
)

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

def handle_selected_example():
    if hasattr(st.session_state, 'selected_example'):
        user_text = st.session_state.selected_example
        st.text_area("Selected example:", value=user_text, height=100, disabled=True)
        del st.session_state.selected_example
        return user_text, True
    return None, False

def display_analysis_results(prediction_score, user_text):
    is_positive = prediction_score > 0.5
    sentiment_label = "Positive" if is_positive else "Negative"
    confidence = prediction_score if is_positive else (1 - prediction_score)
    
    st.header("Analysis Results")
    
    render_sentiment_result(sentiment_label, is_positive)
    render_prediction_scores(prediction_score, confidence)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric(
            label="Sentiment Classification",
            value=sentiment_label,
            delta=f"{confidence:.2%} confidence"
        )
    
    with col2:
        fig = create_sentiment_gauge(prediction_score)
        st.plotly_chart(fig, use_container_width=True)
    
    render_interpretation(is_positive, confidence)
    render_technical_details(prediction_score, user_text)

def main():
    st.markdown('<h1 class="main-header">Sentiment Analysis App</h1>', unsafe_allow_html=True)
    
    with st.spinner("Loading AI model..."):
        model, tokenizer = load_model_and_tokenizer()
    
    if model is None or tokenizer is None:
        st.error("Failed to load model. Please check if the files exist and are valid.")
        return
    
    st.success("Model loaded successfully!")
    
    st.header("Enter Text for Analysis")
    
    user_text = render_text_input()
    analyze_button = st.button("Analyze Sentiment", type="primary", use_container_width=True)
    
    render_example_buttons()
    
    selected_text, auto_analyze = handle_selected_example()
    if selected_text:
        user_text = selected_text
        analyze_button = auto_analyze
    
    if analyze_button and user_text.strip():
        with st.spinner("Analyzing sentiment..."):
            time.sleep(0.5)
            prediction_score = predict_sentiment(user_text, model, tokenizer)
            
            if prediction_score is not None:
                display_analysis_results(prediction_score, user_text)
    
    elif analyze_button and not user_text.strip():
        st.warning("Please enter some text to analyze!")
    
    st.markdown("---")
    st.markdown("Built with Streamlit and TensorFlow | LSTM Neural Network for Sentiment Analysis")

if __name__ == "__main__":
    main()
