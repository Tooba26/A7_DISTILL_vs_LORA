import streamlit as st
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from peft import PeftModel
from datasets import load_dataset
import numpy as np
from collections import Counter
import os

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load model and tokenizer
@st.cache_resource
def load_model():
    # model_path = "./best_lora_model"
    # tokenizer = AutoTokenizer.from_pretrained(model_path)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Construct the full path to best_lora_model
    model_path = os.path.join(script_dir, "best_lora_model")
    
    # Load tokenizer and model from the local directory
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    base_model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=28)
    model = PeftModel.from_pretrained(base_model, model_path)
    model = model.to(device)
    model.eval()
    return model, tokenizer

model, tokenizer = load_model()

# Load GoEmotions dataset for examples
@st.cache_data
def load_dataset_examples():
    dataset = load_dataset("google-research-datasets/go_emotions", "simplified")
    return dataset["test"]

test_dataset = load_dataset_examples()

# Emotion labels
emotion_labels = [
    "admiration", "amusement", "anger", "annoyance", "approval", "caring", "confusion", "curiosity",
    "desire", "disappointment", "disapproval", "disgust", "embarrassment", "excitement", "fear",
    "gratitude", "grief", "joy", "love", "nervousness", "optimism", "pride", "realization",
    "relief", "remorse", "sadness", "surprise", "neutral"
]

# Prediction function
def predict_emotion(text):
    inputs = tokenizer(text, padding="max_length", truncation=True, max_length=128, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
        prediction = torch.argmax(outputs.logits, dim=-1).item()
        probabilities = torch.softmax(outputs.logits, dim=-1)[0].cpu().numpy()
    
    return emotion_labels[prediction], probabilities

# Batch prediction function
def batch_predict(texts):
    inputs = tokenizer(texts, padding=True, truncation=True, max_length=128, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.argmax(outputs.logits, dim=-1).cpu().numpy()
    
    return [emotion_labels[pred] for pred in predictions]

# Streamlit app
st.title("Emotion Prediction with LoRA Model")

# Sidebar
st.sidebar.header("Options")
example_mode = st.sidebar.checkbox("Use Example Texts", value=False)
num_examples = st.sidebar.slider("Number of Examples", 1, 10, 5) if example_mode else 0

# Main content
tab1, tab2, tab3 = st.tabs(["Prediction", "Distribution", "Errors"])

# Prediction Tab
with tab1:
    st.header("Emotion Prediction")
    
    if example_mode:
        st.subheader("Random Examples from GoEmotions")
        random_samples = test_dataset.shuffle(seed=1234).select(range(num_examples))
        sample_texts = random_samples['text']
        true_labels = [emotion_labels[label[0]] if label else "neutral" for label in random_samples['labels']]
        
        for i, (text, true_label) in enumerate(zip(sample_texts, true_labels)):
            predicted_emotion, probs = predict_emotion(text)
            st.write(f"**Example {i+1}**")
            st.write(f"Text: {text}")
            st.write(f"True Emotion: {true_label}")
            st.write(f"Predicted Emotion: {predicted_emotion}")
            st.bar_chart(dict(zip(emotion_labels, probs)))
            st.write("---")
    
    else:
        user_input = st.text_area("Enter your text here:", "I can't believe how amazing this day turned out!")
        if st.button("Predict"):
            predicted_emotion, probs = predict_emotion(user_input)
            st.write(f"Predicted Emotion: **{predicted_emotion}**")
            st.bar_chart(dict(zip(emotion_labels, probs)))

# Distribution Tab
with tab2:
    st.header("Emotion Distribution Analysis")
    num_samples = st.slider("Number of samples for distribution", 10, 200, 100)
    
    if st.button("Analyze Distribution"):
        samples = test_dataset.shuffle(seed=1234).select(range(num_samples))
        texts = samples['text']
        predictions = batch_predict(texts)
        
        dist = Counter(predictions)
        total = sum(dist.values())
        dist_dict = {emotion: count/total for emotion, count in dist.items()}
        
        st.bar_chart(dist_dict)
        st.write("Distribution Details:")
        for emotion, freq in sorted(dist_dict.items(), key=lambda x: x[1], reverse=True):
            st.write(f"{emotion}: {freq:.3f}")

# Error Analysis Tab
with tab3:
    st.header("Error Analysis")
    num_error_samples = st.slider("Number of samples for error analysis", 10, 100, 50)
    
    if st.button("Analyze Errors"):
        samples = test_dataset.shuffle(seed=1234).select(range(num_error_samples))
        texts = samples['text']
        true_labels = [label[0] if label else 0 for label in samples['labels']]
        predicted_labels = [emotion_labels.index(pred) for pred in batch_predict(texts)]
        
        errors = []
        for text, true, pred in zip(texts, true_labels, predicted_labels):
            if true != pred:
                errors.append({
                    'text': text,
                    'true': emotion_labels[true],
                    'predicted': emotion_labels[pred]
                })
        
        st.write(f"Found {len(errors)} errors in {num_error_samples} samples")
        for i, error in enumerate(errors[:5], 1):
            st.write(f"**Error {i}**")
            st.write(f"Text: {error['text']}")
            st.write(f"True Emotion: {error['true']}")
            st.write(f"Predicted Emotion: {error['predicted']}")
            st.write("---")

# Footer
st.sidebar.markdown("---")
st.sidebar.write("Built with Streamlit and Hugging Face Transformers")