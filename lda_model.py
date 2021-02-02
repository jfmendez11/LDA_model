import sys
import math
from pymongo import MongoClient
from gensim.corpora import Dictionary
from gensim.models import LdaModel
import dotenv
import datetime
import argparse
import logging
import json
from bson import json_util
import matplotlib.colors as mcolors
from collections import Counter

parser = argparse.ArgumentParser()

base = '/Users/JuanFelipe/GitHub/'

parser.add_argument("--start", default="2020-03-03 00:00:00.0", type=str, help="Lower date to filter tweets")
parser.add_argument("--end", default="{}".format(datetime.datetime.now()), type=str, help="Most recent date to filter tweets")
parser.add_argument("--accounts", default="", type=str, help="List of accounts to filter tweets")
parser.add_argument("--topics", required=True, type=int, help="Number of topics to execute the model")
parser.add_argument("--logging", default=0, type=int, help="Whether or not to log")
parser.add_argument("--hashtagmodel", default=0, type=int, help="Whether to execute LDA based on hashtags, on tweet text oe both")
parser.add_argument("--keywords", default="", type=str, help="Interest words chosen by the user")

args = parser.parse_args()

start = datetime.datetime.strptime(args.start, "%Y-%m-%d %H:%M:%S.%f")
end = datetime.datetime.strptime(args.end, "%Y-%m-%d %H:%M:%S.%f")
accounts = args.accounts
topics = args.topics
is_logging = args.logging == 1
hashtag_model = args.hashtagmodel
keywords = args.keywords

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
    
  if hashtag_model == 1:
    query["hashtags"] = {'$exists': True, '$ne': [] }
  
  if keywords:
    query["$text"] = { '$search': keywords, '$language': 'es' }
  
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
    eval_every=eval_every,
    per_word_topics=True
  )
  
  return model

def process_data(model, dictionary, docs, tweets, num_topics=topics):
  result = dict()
  model_info = dict()
  # Get colors for graphing pourpuses
  colors = [color for name, color in mcolors.XKCD_COLORS.items()]
  # Flatten docs to use counter
  docs_flat = [w for w_list in docs for w in w_list]
  counter = Counter(docs_flat)
  
  # Get all the relevant info from the model
  for topic, words in model.show_topics(formatted=False, num_topics=num_topics):
    words_dict = dict()
    for word, importance in words:
      # Importance and word count
      word_obj = {
        'count': counter[word],
        'importance': float(importance)
      }
      words_dict[word] = word_obj
    # Set some parameters: words, document count by dominant topic, document count by topic (a document can belong to
    # multiple docs and the color used to graph the relevant topic
    model_info[topic] = {
      'words': words_dict,
      'document_count': 0,
      'weight_count': 0,
      'color': colors[topic]
    }
  # Relevant tweet info
  for tweet in tweets:
    if hashtag_model == 0:
      doc = tweet["tokenized_text"]
    elif hashtag_model == 1:
      doc = [hashtag["text"].lower() for hashtag in tweet["hashtags"]]
    else:
      doc = tweet["tokenized_text"] + [hashtag["text"].lower() for hashtag in tweet["hashtags"]]
      
    # Dominant topic in the doc
    dominant_topic = (-1, 0.0)
    # Dictionary of topics with their importance
    topics = {}
    topic_percentages, word_id_topics, word_id_phivalues = model[dictionary.doc2bow(doc)]
    for topic in topic_percentages:
      topics[topic[0]] = float(topic[1])
      # Increase the topic's weight count
      model_info[topic[0]]['weight_count'] += 1
      # Set the dominant topic
      if topic[1] > dominant_topic[1]:
        dominant_topic = topic
    # Increase the dominant topic's document count
    model_info[dominant_topic[0]]['document_count'] += 1
    tweet["dominant_topic"] = dominant_topic[0]
    tweet["topics"] = topics
    word_dominant_topic = [(model.id2word[wd], topic[0]) for wd, topic in word_id_topics]
    # Set dictionary of the docs words in the relevant topic
    tweet['text_topic'] = dict(word_dominant_topic)

  now = datetime.datetime.now()
  result["time_stamp"] = now.strftime("%d/%m/%Y %H:%M:%S")
  result["tweets"] = tweets
  result["model_info"] = model_info
  json_result = json.dumps(result, default=json_util.default)
  with open(base + "/thesis_web_app/backend/files/model_result.json", "w") as outfile:
    outfile.write(json_result)
    print('{"success":true, "message": "Success running the model and processing data"}')

if __name__ == "__main__":
  dotenv.load_dotenv(".env")
  # Get the tweets stored in the Mongo repo
  tweets = list(get_tweets(isTest=True))
  
  # Collect the documents depending on the model wished to analyze
  if hashtag_model == 0:
    docs = [tweet["tokenized_text"] for tweet in tweets]
  elif hashtag_model == 1:
    docs = [[hashtag["text"].lower() for hashtag in tweet["hashtags"]] for tweet in tweets]
  else:
    docs = list()
    for tweet in tweets:
      docs.append(tweet["tokenized_text"] + [hashtag["text"].lower() for hashtag in tweet["hashtags"]])
  # Create a dictionary representation of the documents.
  dictionary = Dictionary(docs)
  
  # Filter out words that occur less than 20 documents, or more than 80% of the documents.
  # dictionary.filter_extremes(no_below=5, no_above=0.8)
  corpus = [dictionary.doc2bow(doc) for doc in docs]
  log('Number of unique tokens: %d' % len(dictionary))
  log('Number of documents: %d' % len(corpus))
  try:
    if len(dictionary) > 1:
      model = train_model(dictionary, corpus)
      process_data(model, dictionary, docs, tweets)
      top_topics = model.top_topics(corpus)  # , num_words=20)
      # Average topic coherence is the sum of topic coherences of all topics, divided by the number of topics.
      avg_topic_coherence = sum([t[1] for t in top_topics]) / topics
      log('Average topic coherence: %.4f.' % avg_topic_coherence)
    else:
      print('{"success": false, "message": "Dataset too small, please increase number of topics, include more accounts or increase the date range"}')
  except Exception as e:
    #print(e.with_traceback())
    print('{"success": false, "message": "An error occured running the model ' + str(e.with_traceback(sys.exc_info()[2])) + '"}')
