import os
import math
from pymongo import MongoClient
from gensim.corpora import Dictionary
from gensim.models import LdaModel
import dotenv
import datetime
import argparse
import pprint
import logging
import json

parser = argparse.ArgumentParser()

parser.add_argument("--start", default="2020-03-03 00:00:00.0", type=str, help="Lower date to filter tweets")
parser.add_argument("--end", default="{}".format(datetime.datetime.now()), type=str, help="Most recent date to filter tweets")
parser.add_argument("--accounts", default="", type=str, help="List of accounts to filter tweets")
parser.add_argument("--topics", required=True, type=int, help="Number of topics to execute the model")
parser.add_argument("--logging", default=0, type=int, help="Whether or not to log")
parser.add_argument("--hashtagmodel", default=0, type=int, help="Whether to execute LDA based on hashtags or based on tweet text")

args = parser.parse_args()

start = datetime.datetime.strptime(args.start, "%Y-%m-%d %H:%M:%S.%f")
end = datetime.datetime.strptime(args.end, "%Y-%m-%d %H:%M:%S.%f")
accounts = args.accounts
topics = args.topics
is_logging = args.logging == 1
is_hashtag_model = args.hashtagmodel == 1

if is_logging:
  logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
  
def log(args):
  if is_logging:
    print(args)

def get_tweets(isTest=False):
  # MongoDB parameters
  host = "localhost"
  port = 27017

  client = MongoClient(host, port)
  db = client["tweet_miner" if not isTest else "tweet_miner_test"]
  
  query = { "created_at": { "$gte": start, "$lte": end } }
  
  if accounts:
    account_list = accounts.split()
    query["screen_name"] = { "$in": account_list }
  
  return db["tweets"].find(query)

def train_model(dictionary, corpus):
  chunksize = int(math.ceil(len(corpus) / 1000.0)) * 1000
  passes = 20
  iterations = 400
  eval_every = None  # Don't evaluate model perplexity, takes too much time.

  # Make a index to word dictionary.
  temp = dictionary[0] # This is only to "load" the dictionary.
  id2word = dictionary.id2token
  model = LdaModel(
    corpus=corpus,
    id2word=id2word,
    chunksize=chunksize,
    alpha='auto',
    eta='auto',
    iterations=iterations,
    num_topics=topics,
    passes=passes,
    eval_every=eval_every
  )
  
  return model

def remap_topics(result):
  mapped_topics = list()
  for i in range(0, len(result)):
    words = list()
    for topic in result[i][0]:
      word = {
        "word": topic[1],
        "score": float(topic[0])
      }
      words.append(word)
    topic_object = {
      "topics": words,
      "score": float(result[i][1])
    }
    mapped_topics.append(topic_object)
  return mapped_topics

if __name__ == "__main__":
  dotenv.load_dotenv(".env")
  
  docs = [[hashtag["text"].lower() for hashtag in tweet["hashtags"]] for tweet in get_tweets(isTest=True)] if is_hashtag_model else [tweet["tokenized_text"] for tweet in get_tweets(isTest=True)]
  
  # Create a dictionary representation of the documents.
  dictionary = Dictionary(docs)
  
  # Filter out words that occur less than 20 documents, or more than 80% of the documents.
  dictionary.filter_extremes(no_below=5, no_above=0.8)
  corpus = [dictionary.doc2bow(doc) for doc in docs]
  
  log('Number of unique tokens: %d' % len(dictionary))
  log('Number of documents: %d' % len(corpus))
  try:
    if len(dictionary) > 1:
      model = train_model(dictionary, corpus)
      top_topics = model.top_topics(corpus)  # , num_words=20)
      # Average topic coherence is the sum of topic coherences of all topics, divided by the number of topics.
      avg_topic_coherence = sum([t[1] for t in top_topics]) / topics
      log('Average topic coherence: %.4f.' % avg_topic_coherence)
      mapped_topics = remap_topics(top_topics)
      result = json.dumps(mapped_topics)
      print(result)
    else:
      print([])
  except:
    print([])