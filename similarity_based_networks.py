import pandas as pd
import numpy as np
import networkx as nx

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfTransformer
from scipy.sparse import csr_matrix
from pandas.api.types import CategoricalDtype

from datetime import datetime
from datetime import timedelta

from nltk.corpus import stopwords
import nltk
import spacy

def get_tweet_timestamp(tid):
    """
    Computes tweet's original timestamp
    
    Args:
        tid: tweetid (int)
    Returns:
        timestamp  
    """
    try:
        offset = 1288834974657
        tstamp = (tid >> 22) + offset
        utcdttime = datetime.utcfromtimestamp(tstamp/1000)
        return utcdttime
    except:
        return None   


def coSharing(data):
    """
    Builds the similarity network based on any kind of feature
    
    Args:
        data: Pandas dataframe with columns ['userid', 'feature_shared'] 
    Returns:
        undirected graph  
    """

    temp = data.groupby('feature_shared', as_index=False).count()
    data = data.loc[data['feature_shared'].isin(temp.loc[temp['userid']>1]['feature_shared'].to_list())]

    data['value'] = 1
    
    ids = dict(zip(list(data.feature_shared.unique()), list(range(data.feature_shared.unique().shape[0]))))
    data['feature_shared'] = data['feature_shared'].apply(lambda x: ids[x]).astype(int)
    del ids

    userid = dict(zip(list(data.userid.astype(str).unique()), list(range(data.userid.unique().shape[0]))))
    data['userid'] = data['userid'].astype(str).apply(lambda x: userid[x]).astype(int)
    
    person_c = CategoricalDtype(sorted(data.userid.unique()), ordered=True)
    thing_c = CategoricalDtype(sorted(data.feature_shared.unique()), ordered=True)
    
    row = data.userid.astype(person_c).cat.codes
    col = data.feature_shared.astype(thing_c).cat.codes
    sparse_matrix = csr_matrix((data["value"], (row, col)), shape=(person_c.categories.size, thing_c.categories.size))
    del row, col, person_c, thing_c
    
    vectorizer = TfidfTransformer()
    try:
        tfidf_matrix = vectorizer.fit_transform(sparse_matrix)
    except:
        return None
    similarities = cosine_similarity(tfidf_matrix, dense_output=False)


    df_adj = pd.DataFrame(similarities.toarray())
    del similarities
    df_adj.index = userid.keys()
    df_adj.columns = userid.keys()
    G = nx.from_pandas_adjacency(df_adj)
    del df_adj

    G.remove_edges_from(nx.selfloop_edges(G))
    G.remove_nodes_from(list(nx.isolates(G)))

    return G

def coRetweet_sim(data):
    """
    Builds the similarity network based on co-retweet
    
    Args:
        data: Pandas dataframe with columns ['userid', 'tweet_id', 'retweet_tweetid'] 
    Returns:
        undirected graph  
    """
    
    # discards users with low activity of retweets
    filt = dataset[['userid', 'tweetid']].groupby(['userid'],as_index=False).count()
    filt = list(filt.loc[filt['tweetid'] >= 20]['userid'])
    dataset = dataset.loc[dataset['userid'].isin(filt)]
    
    dataset = dataset[['userid', 'retweet_tweetid', 'tweetid']].groupby(['userid', 'retweet_tweetid'],as_index=False).size()

    dataset = dataset[['userid', 'retweet_tweetid']].dropna()
    dataset.drop_duplicates(inplace=True)

    dataset.columns = ['userid', 'feature_shared']

    return coSharing(dataset)

def fastRetweet_sim(data):
    """
    Builds the similarity network based on fast retweeting a source
    
    Args:
        data: Pandas dataframe with columns ['userid', 'tweet_id', 'retweet_tweetid'] 
    Returns:
        undirected graph  
    """
    
    dataset['retweet_tweetid'] = dataset['retweet_tweetid'].astype(int,errors='ignore')
    dataset['tweet_timestamp'] = dataset['tweetid'].apply(lambda x: get_tweet_timestamp(int(x)))
    dataset['retweet_timestamp'] = dataset['retweet_tweetid'].apply(lambda x: get_tweet_timestamp(int(float(x))))
    dataset.dropna(inplace=True)

    dataset['tweet_timestamp'] = pd.to_datetime(dataset['tweet_timestamp'])
    dataset['retweet_timestamp'] = pd.to_datetime(dataset['retweet_timestamp'])

    dataset['delta'] = (dataset['tweet_timestamp'] - dataset['retweet_timestamp']).dt.seconds

    dataset = dataset[['userid', 'retweet_userid', 'delta']]
    dataset['userid'].astype(int).astype(str)
    dataset = dataset.loc[dataset['delta'] <= 10]

    dataset = dataset.groupby(['userid', 'retweet_userid'],as_index=False).count()
    dataset = dataset.loc[dataset['delta'] > 1]

    dataset = dataset[['userid', 'retweet_userid']]

    dataset.columns = ['userid', 'feature_shared']
    
    return coSharing(dataset)

def coURL_sim(data):
    """
    Builds the similarity network based on URL shared
    
    Args:
        data: Pandas dataframe with columns ['userid', 'tweet_id', 'urls'] 
    Returns:
        undirected graph  
    """
    
    # explodes list of urls
    dataset['urls'] = dataset['urls'].astype(str).replace('[]', '').apply(lambda x: x[1:-1].replace("'", '').split(',') if len(x) != 0 else '')
    dataset = dataset.loc[dataset['urls'] != ''].explode('urls')

    dataset = dataset[['userid', 'urls']].dropna()
    dataset.drop_duplicates(inplace=True)

    dataset.columns = ['userid', 'feature_shared']
    
    return coSharing(dataset)
    
def coHashtag_sim(data):
    """
    Builds the similarity network based on hashtags shared
    
    Args:
        data: Pandas dataframe with columns ['userid', 'tweet_id', 'hashtags'] 
    Returns:
        undirected graph  
    """
    
    # explodes list of hashtags
    dataset['hashtags'] = dataset['hashtags'].astype(str).replace('[]', '').apply(lambda x: x[1:-1].replace("'", '').split(',') if len(x) != 0 else '')
    dataset = dataset.loc[dataset['hashtags'] != ''].explode('hashtags')

    dataset = dataset[['userid', 'hashtags']].dropna()
    dataset.drop_duplicates(inplace=True)

    dataset.columns = ['userid', 'feature_shared']

    return coSharing(dataset)
    
def textSimilarity_sim(data):
    """
    Builds the similarity network based on text similarity
    
    Args:
        data: Pandas dataframe with columns ['userid', 'tweet_id', 'retweet_tweetid', 'tweet_text'] 
    Returns:
        undirected graph  
    """
    
    dataset['tweet_text'] = dataset['tweet_text'].apply(lambda x: x.split(' ') if len(x) != 0 else '')
    dataset = dataset.loc[dataset['tweet_text'] != ''].explode('tweet_text')

    dataset = dataset[['userid', 'tweet_text']].dropna()
    dataset.drop_duplicates(inplace=True)

    dataset.columns = ['userid', 'feature_shared']
    
    return coSharing(dataset)
    