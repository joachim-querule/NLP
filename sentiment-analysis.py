import gradio as grd
from textblob import TextBlob
from textblob.sentiments import NaiveBayesAnalyzer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from flair.models import TextClassifier
classifier = TextClassifier.load('en-sentiment')
from flair.data import Sentence
 
 
 
def textblob(text):
    output = TextBlob(text)
    Polarity        = output.sentiment.polarity
    Subjectivity    = output.sentiment.subjectivity
 
    if Polarity > 0:
        sentiment = "Positive 😊"
    elif Polarity < 0:
        sentiment = "Negative 😞"
    else:
        sentiment = "Neutral 😐"
 
    return (
        f"Sentiment: {sentiment}\n"
        f"Polarity: {Polarity:.3f}\n"
        f"Subjectivity: {Subjectivity:.3f}"
    )
 
def naive_bayes_textblob(text):
    naive = TextBlob(text=text, analyzer=NaiveBayesAnalyzer())
    sentiment = naive.sentiment
 
    label = "Positive 😊" if sentiment.classification == "pos" else "Negative 😞"
 
    return (
        f"Sentiment: {label}\n"
        f"Positive Probability: {sentiment.p_pos:.3f}\n"
        f"Negative Probability: {sentiment.p_neg:.3f}"
    )
 
def vader(text):
    vader_sentiment = SentimentIntensityAnalyzer
 
 
 
 
 
with grd.Blocks(title="Sentiment Analysis", theme=grd.themes.Soft()) as analyse_system:
    grd.Markdown("""
            # 🤖 Sentiment Analysis App
            Analyze text using Different models.
            """)
   
    with grd.Tabs():
        with grd.TabItem("Home Page"):
            grd.Markdown(
                """
                # Sentiment analysis System
                So yo can enter your text
                You can try with 4 different models
                """
            )
 
 
        with grd.TabItem("TextBlob Model"):
            grd.Markdown(
                """
                # Sentiment analysis with the textblob model
                Enter your text and try to find the sentiment analysis by using this system and this model
                """
            )
            # Input and Output Components
            input_text = grd.Textbox(label="Input Text", placeholder="Enter your text here...", lines=4)
            output_text = grd.Textbox(label="Sentiment Analysis Result", interactive=False)
 
            # Buttons
            with grd.Row():
                submit_button = grd.Button("Submit")
                clear_button = grd.Button("Clear")
 
            # Examples
            grd.Examples(
                examples=[
                    ["I absolutely love this product!"],
                    ["This movie was terrible and boring."],
                    ["The service was okay, nothing special."],
                    ["Amazing experience, highly recommend it."],
                    ["I am disappointed with the quality."]
                ],
                inputs=input_text,
                label="Try these examples!"
            )
 
            # Button Actions
            submit_button.click(fn=textblob, inputs=input_text, outputs=output_text)
            clear_button.click(lambda: ("", ""),outputs=[input_text, output_text])
 
 
 
 
 
        with grd.TabItem("TextBlob with Naive Bayes Model"):
            grd.Markdown(
                """
                # Sentiment analysis with the textblob Naive Bayes model
                Enter your text and try to find the sentiment analysis by using this system and this model
                """
            )
            # Input and Output Components
            input_text = grd.Textbox(label="Input Text", placeholder="Enter your text here...", lines=4)
            output_text = grd.Textbox(label="Sentiment Analysis Result", interactive=False)
 
            # Buttons
            with grd.Row():
                submit_button = grd.Button("Submit")
                clear_button = grd.Button("Clear")
 
            # Examples
            grd.Examples(
                examples=[
                    ["I absolutely love this product!"],
                    ["This movie was terrible and boring."],
                    ["The service was okay, nothing special."],
                    ["Amazing experience, highly recommend it."],
                    ["I am disappointed with the quality."]
                ],
                inputs=input_text,
                label="Try these examples!"
            )
 
            # Button Actions
            submit_button.click(fn=naive_bayes_textblob, inputs=input_text, outputs=output_text)
            clear_button.click(lambda: ("", ""),outputs=[input_text, output_text])
 
        with grd.TabItem("Vader Model"):
            grd.Markdown(
                """
                # Sentiment analysis with the textblob model
                Enter your text and try to find the sentiment analysis by using this system and this model
                """
            )
 
 
        with grd.TabItem("Flair Model"):
            grd.Markdown(
                """
                # Sentiment analysis with the textblob model
                Enter your text and try to find the sentiment analysis by using this system and this model
                """
            )
 
 
 
 
 
analyse_system.launch(share=True, debug=True)
 