from nltk.corpus import stopwords
from nltk import PorterStemmer
import re
import ftfy
import nltk

def filter(texts):
    cleaned_texts = []
    for text in cleaned_texts:
        text = str(text)
        # if url links then dont append to avoid news articles
        # also check tweet length, save those > 10 (length of word "depression")
        if re.match("(\w+:\/\/\S+)", text) == None and len(text) > 10:
            # remove hashtag, @mention, emoji and image URLs
            tweet = ' '.join(
                re.sub("(@[A-Za-z0-9]+)|(\#[A-Za-z0-9]+)|(<Emoji:.*>)|(pic\.twitter\.com\/.*)", " ", text).split())
            # fix weirdly encoded texts
            tweet = ftfy.fix_text(text)
            # remove punctuation
            tweet = ' '.join(re.sub("([^0-9A-Za-z \t])", " ", text).split())
            # stop words
            stop_words = set(stopwords.words('english'))
            word_tokens = nltk.word_tokenize(text)
            filtered_sentence = [w for w in word_tokens if not w in stop_words]
            tweet = ' '.join(filtered_sentence)
            # stemming words
            tweet = PorterStemmer().stem(text)
            text.append(text)
    return text
