
import os
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.nn.functional import softmax
from sklearn.metrics import f1_score



def generate_new_ner():
    import re
    import spacy
    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    from nltk.stem import WordNetLemmatizer
    import gensim
    from gensim import corpora
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    from itertools import chain

    nltk.download('punkt')
    nltk.download('punkt_tab')
    nltk.download('stopwords')
    nltk.download('wordnet')

    nlp = spacy.load("en_core_web_sm")
    sid = SentimentIntensityAnalyzer()

    df = pd.read_csv("data/India_sector_news_articles.csv")

    df = df[
        df['title'].str.contains("India", case=False, na=False) |
        df['description'].str.contains("India", case=False, na=False) |
        df['content'].str.contains("India", case=False, na=False)
    ].copy()

    def clean_text(text):
        text = re.sub(r'<.*?>|\[.*?\]|[^a-zA-Z\s]', '', str(text).lower())
        tokens = word_tokenize(text)
        tokens = [t for t in tokens if t not in set(stopwords.words('english'))]
        lemmatizer = WordNetLemmatizer()
        return ' '.join([lemmatizer.lemmatize(t) for t in tokens])

    df['cleaned_text'] = (df['title'].astype(str) + ' ' + df['description'].astype(str) + ' ' + df['content'].astype(str)).apply(clean_text)

    manual_mapping = {
        "trump": "Donald Trump", "donald trump": "Donald Trump",
        "donald trump's": "Donald Trump", "rbi": "RBI",
        "reserve bank of india": "RBI", "pm modi": "Narendra Modi",
        "narendra modi": "Narendra Modi"
    }

    def normalize(ent): return manual_mapping.get(ent.lower().strip().replace("’s", "").replace("’", "").replace("'", ""), ent.title())
    def is_valid_entity(ent): return len(ent.split()) <= 4 and not any(x in ent.lower() for x in ["href", "window", "http", "cookie", "terms", "login", "march", "april"])
    def extract_entities(text): return [normalize(ent.text) for ent in nlp(text).ents if ent.label_ in ['PERSON', 'ORG'] and is_valid_entity(ent.text)]

    df['entities'] = df['cleaned_text'].apply(extract_entities)

    economic_terms = ["tariff", "inflation", "export", "import", "policy", "reform", "gst", "subsidy", "gdp", "trade deficit", "rupee", "rbi", "interest rate"]
    df['econ_terms'] = df['cleaned_text'].apply(lambda text: [term.title() for term in economic_terms if term in text.lower()])
    df['combined_influencers'] = df.apply(lambda row: row['entities'] + row['econ_terms'], axis=1)

    texts = [text.split() for text in df['cleaned_text']]
    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]
    lda_model = gensim.models.ldamodel.LdaModel(corpus, num_topics=5, id2word=dictionary, passes=10)

    df['topic'] = [lda_model[corpus[i]][0][0] for i in range(len(corpus))]
    df['topic_probability'] = [lda_model[corpus[i]][0][1] for i in range(len(corpus))]
    topic_labels = {
        0: "US Tariff Market Impact",
        1: "India Export Policy",
        2: "India Economic Disruption",
        3: "Trade Diplomacy & Leadership",
        4: "Trump’s Tariff Legacy"
    }
    df['topic_label'] = df['topic'].map(topic_labels)

    def get_adjusted_sentiment(text):
        score = sid.polarity_scores(text)['compound']
        if -0.05 < score < 0.05:
            return 0.0
        return score

    df['sentiment'] = df['cleaned_text'].apply(get_adjusted_sentiment)

    df.to_csv("data/new_NER.csv", index=False)
    st.success("Extracted high-impact topics from India sector news.")


def display():
    st.title("\U0001F4C8 Temporal Trend Analysis")
    # Reddit Sentiment Trends
    st.header("Public Sentiment Over Time(Country-wise)")
    df_reddit = pd.read_csv("data/reddit_sentiment_labeled.csv")
    df_reddit['date'] = pd.to_datetime(df_reddit['date'], errors='coerce')
    df_reddit = df_reddit.dropna(subset=['date'])

    subreddit_to_country = {
        "india": "India", "mexico": "Mexico", "china": "China",
        "AskEconomics": "USA", "economics": "USA",
        "geopolitics": "Global", "Sino": "China", "canada": "Canada"
    }

    df_reddit['country'] = df_reddit['subreddit'].map(subreddit_to_country).fillna("Other")
    df_reddit['month'] = df_reddit['date'].dt.to_period("M").astype(str)
    df_reddit = df_reddit[~df_reddit['country'].isin(['Global', 'Other'])]

    grouped = df_reddit.groupby(['month', 'country', 'sentiment']).size().unstack(fill_value=0).reset_index()
    countries = df_reddit['country'].unique()
    cols = st.columns(2)
    for i, country in enumerate(countries):
        cdata = grouped[grouped['country'] == country].set_index('month')[['negative', 'neutral', 'positive']]
        with cols[i % 2]:
            st.subheader(f"{country}")
            st.line_chart(cdata)
        if i % 2 == 1 and i < len(countries) - 1:
            cols = st.columns(2)
    

    

                

        



    st.header("📢 Statement-Level Stance Detection")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @st.cache_resource
    def load_stance_model():
        model_path = "stance_xlmr_model"
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        model.to(device)
        model.eval()
        return tokenizer, model

    def predict_stance(text, tokenizer, model, threshold=0.6):
        tokens = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=128).to(device)
        with torch.no_grad():
            outputs = model(**tokens)
        probs = softmax(outputs.logits, dim=1)
        conf, pred = torch.max(probs, dim=1)
        conf = conf.item()
        pred = pred.item()
        label = "Uncertain" if conf < threshold else ("Government" if pred == 1 else "Public")
        return label, {
            "Public": round(probs[0][0].item(), 2),
            "Government": round(probs[0][1].item(), 2),
            "Confidence": round(conf, 2)
        }

    tokenizer, model = load_stance_model()
    user_input = st.text_area("Enter a policy-related or opinion-based statement:")
    if st.button("Classify Stance"):
        if not user_input.strip():
            st.warning("⚠️ Please enter some text.")
        else:
            label, probs = predict_stance(user_input, tokenizer, model)
            st.markdown(f"### 🧾 Prediction: `{label}`")
            st.markdown(
                f"**Public**: `{probs['Public']}` &nbsp;&nbsp;&nbsp; "
                f"**Government**: `{probs['Government']}` &nbsp;&nbsp;&nbsp; "
                f"✅ **Confidence**: `{probs['Confidence']}`"
            )
            if label != "Uncertain":
                st.progress(probs["Confidence"])


# ---------------------------------

    st.header("\U0001F1EE\U0001F1F3 India News — Topic & Sentiment Trends")
    if not os.path.exists("data/new_NER.csv"):
        with st.spinner("⏳ Generating enriched India topic data..."):
            generate_new_ner()


    df_india = pd.read_csv("data/new_NER.csv")

    st.subheader("Top Influencers in India-Focused Articles")
    from itertools import chain
    all_influencers = list(chain.from_iterable(df_india['combined_influencers'].dropna().apply(eval)))
    influencer_counts = Counter(all_influencers)
    top_influencers = influencer_counts.most_common(15)
    top_names, top_counts = zip(*top_influencers)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=list(top_counts), y=list(top_names), palette="viridis", ax=ax)
    ax.set_title("Top Influencers in India-Focused Articles")
    ax.set_xlabel("Mentions")
    ax.set_ylabel("Entity")
    plt.tight_layout()
    st.pyplot(fig)

    st.subheader("Average Sentiment by Topic")
    avg_sentiment = df_india.groupby('topic_label')['sentiment'].mean().sort_values()
    fig2, ax2 = plt.subplots(figsize=(10, 4))
    sns.barplot(x=avg_sentiment.values, y=avg_sentiment.index, palette="coolwarm", ax=ax2)
    ax2.set_title("Average Sentiment per Topic", fontsize=14)
    ax2.set_xlabel("Sentiment Score", fontsize=12)
    ax2.set_ylabel("Topic", fontsize=12)
    ax2.tick_params(labelsize=10)
    plt.tight_layout()
    st.pyplot(fig2)

    st.subheader("Weekly Topic Trends")
    df_india['week'] = pd.to_datetime(df_india['publish_date'], errors='coerce').dt.to_period("W").apply(lambda r: r.start_time)
    weekly_trends = df_india.groupby(['week', 'topic_label']).size().unstack(fill_value=0)
    fig3, ax3 = plt.subplots(figsize=(12, 5))
    weekly_trends.plot(marker='o', ax=ax3)
    ax3.set_title("Weekly Frequency of Topics", fontsize=14)
    ax3.set_xlabel("Week", fontsize=12)
    ax3.set_ylabel("Articles", fontsize=12)
    ax3.tick_params(axis='x', labelrotation=45, labelsize=9)
    ax3.legend(title="Topic", bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9, title_fontsize=10)
    plt.tight_layout()
    st.pyplot(fig3)

   