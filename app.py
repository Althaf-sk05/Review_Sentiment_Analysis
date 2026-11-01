import pickle
import streamlit as st
from preprocessing import clean_text

st.set_page_config(
    page_title="Review Sentiment Analysis",
    page_icon="💬",
    layout="centered"
)

@st.cache_resource
def load_model():
    with open("sentiment_model.pkl", "rb") as model_file:
        return pickle.load(model_file)

model = load_model()

st.title("💬 Review Sentiment Analysis")
st.write(
    "Enter a customer review below. The Naïve Bayes model will classify it "
    "as positive or negative."
)

review = st.text_area(
    "Customer review",
    placeholder="Example: The food was fantastic!",
    height=130
)

if st.button("Predict sentiment", type="primary", use_container_width=True):
    if not review.strip():
        st.warning("Please enter a review before predicting.")
    else:
        cleaned_review = clean_text(review)
        if not cleaned_review.strip():
            st.warning("This review contains no usable words. Please enter a longer review.")
        else:
            prediction = int(model.predict([cleaned_review])[0])
            probabilities = model.predict_proba([cleaned_review])[0]
            classes = list(model.classes_)
            positive_probability = (
                float(probabilities[classes.index(1)]) if 1 in classes else 0.0
            )

            if prediction == 1:
                st.success("😊 Positive review")
            else:
                st.error("😞 Negative review")

            st.caption(
                f"Estimated probability of positive sentiment: "
                f"{positive_probability:.1%}"
            )

with st.expander("Try sample reviews"):
    st.write("Positive: The food was fantastic and the service was excellent.")
    st.write("Negative: The food was cold and the service was terrible.")

st.divider()
st.caption("Model: Bernoulli Naïve Bayes | Dataset: Reviews_data")
