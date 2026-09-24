import streamlit as st
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression


# -------------------------------
# 1. Load the dataset
# -------------------------------

data = pd.read_csv("spam.csv")


# -------------------------------
# 2. Separate messages and labels
# -------------------------------

X = data["message"]
y = data["label"]


# -------------------------------
# 3. Convert text into numbers
# -------------------------------

vectorizer = CountVectorizer()

X = vectorizer.fit_transform(X)


# -------------------------------
# 4. Create Logistic Regression model
# -------------------------------

model = LogisticRegression()

model.fit(X, y)


# -------------------------------
# 5. Streamlit Interface
# -------------------------------

st.title("📧 Spam Message Detector")

st.write("Enter a message below to check whether it is Spam or Not Spam.")


# Input box

message = st.text_area(
    "Enter your message:"
)


# Prediction button

if st.button("Check Message"):

    if message.strip() == "":
        st.warning("Please enter a message.")

    else:

        # Convert the new message into numbers
        message_number = vectorizer.transform([message])

        # Predict
        prediction = model.predict(message_number)

        # Get probability
        probability = model.predict_proba(message_number)


        # Display result

        if prediction[0] == "spam":

            st.error("🚨 This message is SPAM!")

            st.write(
                "Spam Probability:",
                round(probability[0][1] * 100, 2),
                "%"
            )

        else:

            st.success("✅ This message is NOT SPAM!")

            st.write(
                "Not Spam Probability:",
                round(probability[0][0] * 100, 2),
                "%"
            )
