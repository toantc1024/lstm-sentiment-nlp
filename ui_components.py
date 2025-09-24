import streamlit as st

def get_example_texts():
    return [
        "This movie was absolutely fantastic! I loved every moment of it.",
        "The service was terrible and the food was awful.",
        "I'm feeling great today, everything is going well!",
        "This is the worst experience I've ever had.",
        "The weather is nice today.",
        "I hate waiting in long lines.",
        "This book changed my life in the most positive way.",
        "The app crashes constantly and is very frustrating."
    ]

def render_text_input():
    user_text = st.text_area(
        "Enter or paste your text here:",
        height=150,
        placeholder="e.g., This movie was absolutely fantastic! I loved every moment of it.",
        help="Enter any text you want to analyze for sentiment (positive or negative)"
    )
    return user_text

def render_example_buttons():
    st.header("Example Texts")
    example_texts = get_example_texts()
    
    cols = st.columns(2)
    for i, example in enumerate(example_texts):
        col_idx = i % 2
        with cols[col_idx]:
            if st.button(f"Try Example {i+1}", key=f"example_{i}", use_container_width=True):
                st.session_state.selected_example = example

def render_sentiment_result(sentiment_label, is_positive):
    if is_positive:
        st.markdown(f'<div class="sentiment-positive">Sentiment: {sentiment_label}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="sentiment-negative">Sentiment: {sentiment_label}</div>', unsafe_allow_html=True)

def render_prediction_scores(prediction_score, confidence):
    st.markdown(f'<div class="prediction-score">Prediction Score: {prediction_score:.4f}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="prediction-score">Confidence: {confidence:.2%}</div>', unsafe_allow_html=True)

def render_interpretation(is_positive, confidence):
    st.header("Interpretation")
    if is_positive:
        st.success(f"The text expresses **positive sentiment** with {confidence:.2%} confidence. The model detected optimistic, happy, or favorable emotions in the text.")
    else:
        st.error(f"The text expresses **negative sentiment** with {confidence:.2%} confidence. The model detected pessimistic, sad, or unfavorable emotions in the text.")

def render_technical_details(prediction_score, text):
    with st.expander("Technical Details"):
        st.write(f"**Raw prediction score:** {prediction_score:.6f}")
        st.write(f"**Decision threshold:** 0.5")
        st.write(f"**Text length:** {len(text)} characters")
        st.write(f"**Word count:** {len(text.split())} words")
