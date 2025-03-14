import os
import json
import datetime
import csv
import nltk
import ssl
import streamlit as st
import random
import time  # Added for typing effect
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from newsapi import NewsApiClient  # Import NewsAPI for trending news

NEWS_API_KEY = "3eed1acd617c470b96573baab0a713fa"    #visit this -> https://newsapi.org/ website and create your own API key and add    
newsapi = NewsApiClient(api_key=NEWS_API_KEY)

ssl._create_default_https_context = ssl._create_unverified_context
nltk.data.path.append(os.path.abspath("nltk_data"))
nltk.download('punkt')

# Load intents from the JSON file
file_path = os.path.abspath("D:\Internship\ChatBot_AICTE Cycle 4 Green skills & AI Skills4Future\Chatbot\intents.json")
with open(file_path, "r") as file:
    intents = json.load(file)

# Create the vectorizer and classifier
vectorizer = TfidfVectorizer()
clf = LogisticRegression(random_state=0, max_iter=10000)

# Preprocess the data
tags = []
patterns = []
for intent in intents:
    for pattern in intent['patterns']:
        tags.append(intent['tag'])
        patterns.append(pattern)

# Training the model
x = vectorizer.fit_transform(patterns)
y = tags
clf.fit(x, y)

def chatbot(input_text):
    input_text = vectorizer.transform([input_text])
    tag = clf.predict(input_text)[0]
    for intent in intents:
        if intent['tag'] == tag:
            response = random.choice(intent['responses'])
            return response

counter = 0

def get_trending_news():
    """Fetch trending news headlines and images."""
    news_data = newsapi.get_top_headlines(language="en", country="us")
    articles = news_data["articles"][:10]  # Fetch top 10 trending articles

    trending_news = []
    for article in articles:
        title = article["title"]
        url = article["url"]
        image = article["urlToImage"]
        trending_news.append({"title": title, "url": url, "image": image})
    return trending_news

def main():
    global counter
    st.title("Intents of Chatbot using NLP")

    # Create a sidebar menu with options
    menu = ["Home", "Conversation History", "Trending News", "About"]
    choice = st.sidebar.selectbox("Menu", menu)

    # Home Menu
    if choice == "Home":
        st.write("Welcome to the chatbot. Please type a message and press Enter to start the conversation.")

        # Check if the chat_log.csv file exists, and if not, create it with column names
        if not os.path.exists('chat_log.csv'):
            with open('chat_log.csv', 'w', newline='', encoding='utf-8') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow(['User Input', 'Chatbot Response', 'Timestamp'])

        counter += 1
        user_input = st.text_input("You:", key=f"user_input_{counter}")

        if user_input:
            user_input_str = str(user_input)
            response = chatbot(user_input)
            
            response_placeholder = st.empty()  
            response_text = ""  
            for word in response.split():
                response_text += word + " "  
                response_placeholder.markdown(f"**Chatbot:** {response_text}", unsafe_allow_html=True)  
                time.sleep(0.2)  




            # Get the current timestamp
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # Save the user input and chatbot response to the chat_log.csv file
            with open('chat_log.csv', 'a', newline='', encoding='utf-8') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow([user_input_str, response, timestamp])

            if response.lower() in ['goodbye', 'bye']:
                st.write("Thank you for chatting with me. Have a great day!")
                st.stop()

    #trending news
    elif choice == "Trending News":
        st.header("Trending News")
        trending_news = get_trending_news()

        if trending_news:
            for news in trending_news:
                st.subheader(f"📰 {news['title']}")
                st.write(f"[Read more]({news['url']})")
                if news["image"]:
                    st.image(news["image"], use_container_width=True)
                st.markdown("---")
        else:
            st.write("No trending news available at the moment.")

    # Conversation History Menu
    elif choice == "Conversation History":
        st.header("Conversation History")
        with open('chat_log.csv', 'r', encoding='utf-8') as csvfile:
            csv_reader = csv.reader(csvfile)
            next(csv_reader)  # Skip the header row
            for row in csv_reader:
                st.text(f"User: {row[0]}")
                st.text(f"Chatbot: {row[1]}")
                st.text(f"Timestamp: {row[2]}")
                st.markdown("---")

    # About Section
    elif choice == "About":
        st.write("The goal of this project is to create a chatbot that can understand and respond to user input based on intents. The chatbot is built using Natural Language Processing (NLP) library and Logistic Regression.")

        st.subheader("Project Overview:")
        st.write("""
        The project is divided into two parts:
        1. NLP techniques and Logistic Regression algorithm are used to train the chatbot on labeled intents and entities.
        2. The chatbot interface is built using the Streamlit web framework.
        3. There is use of news API for trending news.
        """)

        st.subheader("Streamlit Chatbot Interface:")
        st.write("The chatbot interface is built using Streamlit. The interface includes a text input box for users to input their text and a chat window to display the chatbot's responses.")

        st.subheader("Conclusion:")
        st.write("In this project, a chatbot is built that can understand and respond to user input based on intents. The chatbot was trained using NLP and Logistic Regression.")

if __name__ == '__main__':
    main()
