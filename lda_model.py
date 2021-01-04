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

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

parser = argparse.ArgumentParser()

parser.add_argument("--start", default="2020-03-03 00:00:00.0", type=str, help="Lower date to filter tweets")
parser.add_argument("--end", default="{}".format(datetime.datetime.now()), type=str, help="Most recent date to filter tweets")
parser.add_argument("--accounts", default="", type=str, help="List of accounts to filter tweets")
parser.add_argument("--topics", required=True, type=int, help="Number of topics to execute the model")

args = parser.parse_args()

start = datetime.datetime.strptime(args.start, "%Y-%m-%d %H:%M:%S.%f")
end = datetime.datetime.strptime(args.end, "%Y-%m-%d %H:%M:%S.%f")
accounts = args.accounts
topics = args.topics

def get_tweets(isTest=False):
  # MongoDB parameters
  host = os.getenv("MONGO_DB_HOST")
  port = int(os.getenv("MONGO_DB_PORT"))

  client = MongoClient(host, port)
  db = client["tweet_miner" if not isTest else "tweet_miner_test"]
  
  query = { "created_at": { "$gte": start, "$lte": end } }
  
  if accounts:
    account_list = accounts.split()
    query["screen_name"] = { "$in": account_list }
  
  print(query)
  return db["tweets"].find(query)

def train_model(dictionary, chunksize):
  chunksize = int(math.ceil(chunksize / 1000.0)) * 1000
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

if __name__ == "__main__":
  dotenv.load_dotenv(".env")
  
  docs = [tweet["tokenized_text"] for tweet in get_tweets(isTest=True)]
  
  # Create a dictionary representation of the documents.
  dictionary = Dictionary(docs)
  
  # Filter out words that occur less than 20 documents, or more than 80% of the documents.
  dictionary.filter_extremes(no_below=5, no_above=0.8)
  corpus = [dictionary.doc2bow(doc) for doc in docs]
  
  print('Number of unique tokens: %d' % len(dictionary))
  print('Number of documents: %d' % len(corpus))
  
  model = train_model(dictionary, len(corpus))
  top_topics = model.top_topics(corpus)  # , num_words=20)
  
  # Average topic coherence is the sum of topic coherences of all topics, divided by the number of topics.
  avg_topic_coherence = sum([t[1] for t in top_topics]) / topics
  print('Average topic coherence: %.4f.' % avg_topic_coherence)
  pprint.pprint(top_topics)