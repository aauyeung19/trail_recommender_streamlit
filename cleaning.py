"""
@Andrew
Cleaning Methods for NLP Pipeline
"""
import pandas as pd
import numpy as np
import sqlalchemy
import spacy
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, RegexpTokenizer
from spacy.lemmatizer import Lemmatizer
from spacy.lookups import Lookups
import re
import pickle
import string
sp = spacy.load('en_core_web_sm')
stopwords_list = stopwords.words('english')
stopwords_list.remove('no')
stopwords_list.remove('not')

# remove stop words
def remove_stopwords(text):
    tokens = word_tokenize(text)
    tokens = [token.strip() for token in tokens]
    filtered_tokens = [token for token in tokens if token not in stopwords_list]
    return ' '.join(filtered_tokens)

# remove numbers and punctuation
def remove_digits(text):
    text = re.sub(r'[^A-Za-z\s]', '', text)
    return text

# lemmatize
def lemma(text):
    text = sp(text)
    text = [word.lemma_ for word in text]
    text = ' '.join(text)
    return text

# Remove states/named entities
def remove_loc(text):
    sentence = sp(text)
    filtered_sentence = [word for word in sentence if word.ent_type=='GPE']
    text = ' '.join(filtered_sentence)
    return text
    
# Remove Pronouns:
def remove_pron(text):
    text = re.sub(r'[-PRON-]', '', text)
    return text


def clean_corpus(corpus):
    cleaned_corpus = []

    for doc in corpus:
        # lower case
        doc = doc.lower()
        # remove numbers and punctuation
        doc = remove_digits(doc)
        # lemmatize
        doc = lemma(doc)
        # remove stop words
        doc = remove_stopwords(doc)
        # remove pronouns
        doc = remove_pron(doc)

        cleaned_corpus.append(doc)
    
    return cleaned_corpus

def clean_reviews_by_sentence(df):
    """
    Tokenizes corpus by sentence, lemmatizes words, and remove punctuation. 
    """
    clean_rev = []
    for _, curr_row in df.iterrows():
        hike_id = curr_row['hike_id']
        user_id = curr_row['user_id']
        date = curr_row['date']
        user_desc = curr_row['user_desc']
        user_rating = curr_row['user_rating']
        
        review_sents = sp(str(user_desc).lower()).sents
        for sent in review_sents:
            curr_sent = [word.lemma_ for word in sent if not word.is_punct]
            curr_sent = [word for word in curr_sent if word not in stopwords_list]
            curr_sent = " ".join(curr_sent)
            clean_rev.append({
                'hike_id': hike_id,
                'user_id': user_id,
                'date': date,
                'user_desc': user_desc,
                'user_rating': user_rating,
                'clean_review': curr_sent
            })

    clean_rev = pd.DataFrame(data=clean_rev, columns=['hike_id', 'user_id', 'date', 'user_desc', 'user_rating', 'cleaned_desc'])
    return clean_rev


def convert_rating(rating):
    first = rating[0]
    if (first =='N') or (first == '-'):
        return 0
    else:
        return int(first)

def clean_measurements(text):
    try:
        text = re.sub(r'[^0-9.\s]', '', text)
        number = float(text)
        return number
    except:
        return None

def add_link_base(text):
    if "https" in text:
        return text
    else:
        text = "https://www.alltrails.com"+text
        return text

if __name__ == "__main__":
    import psycopg2

    conn=psycopg2.connect(database='alltrails', user='postgres', host='127.0.0.1', port= '5432')
    q = "SELECT * FROM hikes;"
    hikes_df = pd.read_sql_query(q, conn)
    hikes_df['cleaned_descriptions'] = clean_corpus(hikes_df['trail_description'])
    hikes_df['trail_length'] = hikes_df['trail_length'].apply(clean_measurements)
    hikes_df['trail_elevation'] = hikes_df['trail_elevation'].apply(clean_measurements)
    hikes_df['link'] = hikes_df['link'].apply(add_link_base)
    hikes_df.to_csv('../src/clean_all_hikes.csv')

    q = "SELECT * FROM reviews;"
    reviews_df = pd.read_sql_query(q, conn)
    reviews_df.dropna(inplace=True)
    reviews_df['cleaned_reviews'] = clean_corpus(reviews_df['user_desc'])
    reviews_df['user_rating'] = reviews_df['user_rating'].apply(convert_rating)
    reviews_df.to_csv('../src/clean_reviews.csv')

    # split by period into individual rows while keeping hike_id and user_id combination 
    reviews_by_sent = clean_reviews_by_sentence(reviews_df)
    reviews_by_sent.to_csv('../src/reviews_by_sent.csv')
    