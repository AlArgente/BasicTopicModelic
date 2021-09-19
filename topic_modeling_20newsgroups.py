"""Topic Modeling Fetch_20newsgroups

Script for topic modeling the Fetch_20newsgroups from sklearn.
In this script I will be using LDA as topic modeling algorithm.
I will use spacy and gensim, but I won't use NLTK for preprocessing
the text. 

I usually use nltk for all the text preprocessing, but I willuse
other libraries such as spacy and gensim to learn new ways of
preprocessing the text. For lemmatizing I will use spacy, but if
stemming is selected, I will use nltk. I've found that nltk stemmer
and lemmatizer are better thatn spacy's one, because nltk has
better algorithms/rules for that. Also it could be a nice option to use
stanza, a NLP package made by Stanford University.
"""
import spacy
import gensim
from gensim import models
from nltk.stem import SnowballStemmer
from sklearn.datasets import fetch_20newsgroups

def preprocessing(text, remove_stopwords=True, stemming=False):
    """Function to preprocess the text.

    Args:
        text (list): List with all the text to preprocess.
        remove_stopwords (bool, optional): Remove stopwords. Defaults to True.
        stemming (bool, optional: Stemming (True) or lemmatize (False). Defaults to False.
    """
    text_tokenized = tokenize(text, remove_stopwords)
    return do_stem(text_tokenized) if stemming else lemmatize(text_tokenized)

def tokenize(text, remove_stopwords=True):
    """Function that tokenize the text. This function also lower the
    text, and can remove the stopwords.

    Args:
        text (list): List with all the text to tokenize
        remove_stopwords (bool, optional): Remove stopwords. Defaults to True.

    Returns:
        list: text tokenized
    """
    nlp = spacy.load('en_core_web_sm')
    if remove_stopwords:
        return [[word for word in nlp(news) if word.is_stop is False] for news in text]
    else:
        return [[word for word in nlp(news)] for news in text]

def do_stem(text):
    """Function that stem the text

    Args:
        text (list): text to stem
    Returns:
        list: text stemmed
    """
    stemmer = SnowballStemmer('english')
    return [[stemmer.stem(word.text) for word in news_tokenized] for news_tokenized in text]

def lemmatize(text):
    """Funtion that lemmatize the text

    Args:
        text (list): text to lemmatize

    Returns:
        list: text lemmatized
    """
    return [[word.lemma_ for word in news_tokenized] for news_tokenized in text]

def main():
    print('Topic Modeling fetch_20newsgroups with LDA.\n')
    # Load train and test data
    newsgroups_train = fetch_20newsgroups(subset='train', shuffle=True)
    newsgroups_test = fetch_20newsgroups(subset='test', shuffle=True)
    
    # To see the complete information about the dataset, uncommnet the next line.
    # print(train.DESCR)
    
    # There are twenty classes in this dataset.
    print('Number of targets for the fetch_20newsgroup:', len(newsgroups_train.target_names))
    print('The targets are:', newsgroups_train.target_names)
    print('\n\nExample of news in this problem:')
    print(newsgroups_train.data[0])
    
    # Take news data and target
    train_text, y_train = newsgroups_train.data, newsgroups_train.target
    test_text, y_test = newsgroups_test.data, newsgroups_test.target
    
    # Now preprocess the data
    train_text = preprocessing(train_text)
    test_text = preprocessing(test_text)
    print(train_text[0])
    
    # Once the text is preprocesed, have to create the vocabulary (id2word)
    id2word = gensim.corpora.Dictionary(train_text)
    corpus = [id2word.doc2bow(text) for text in train_text]
    
    # Build the LDA model using Gensim.
    lda_model = models.ldamodel.LdaModel(
        corpus=corpus,
        id2word=id2word,
        num_topics=20,
        random_state=42,
        update_every=1,
        chunksize=100,
        passes=10,
        alpha='auto',
        per_word_topics=True
    )
    

if __name__ == '__main__':
    main()
