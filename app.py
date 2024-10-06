import streamlit as st
from transformers import pipeline

# Load the NER model
ner_pipeline = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english")

# Load the Sentence Completion model
fill_mask_pipeline = pipeline("fill-mask", model="bert-base-uncased")


# NER Feature
def ner_feature():
    st.header("Named Entity Recognition")
    user_input = st.text_area("Enter text for NER:")
    if st.button("Recognize Entities"):
        if user_input:
            entities = ner_pipeline(user_input)
            st.write("### Recognized Entities:")
            for entity in entities:
                st.write(f"{entity['word']} - {entity['entity']}")
        else:
            st.warning("Please enter some text.")


# Sentence Completion Feature
def sentence_completion_feature():
    st.header("Sentence Completion")
    sentence_input = st.text_area("Enter a sentence with [MASK]:")
    if st.button("Complete Sentence"):
        if sentence_input:
            completions = fill_mask_pipeline(sentence_input)
            st.write("### Suggested Completions:")
            for completion in completions:
                st.write(f"{completion['sequence']}")
        else:
            st.warning("Please enter a sentence with [MASK].")


# Main Function
def main():
    st.title("NER and Sentence Completion App")
    st.write("Choose a feature to use:")

    option = st.selectbox("Select Feature:", ("Named Entity Recognition", "Sentence Completion"))

    if option == "Named Entity Recognition":
        ner_feature()
    elif option == "Sentence Completion":
        sentence_completion_feature()


if __name__ == "__main__":
    main()
