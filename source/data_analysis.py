import re
import os
import pandas as pd
import numpy as np
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from html.parser import HTMLParser
import contractions
import matplotlib.pyplot as plt
from brokenaxes import brokenaxes
from scipy.stats import norm

os.chdir('Documents/projects/fitness_article_topic_modeling')

############################
# functions

# clean html tags out of documents
def remove_html_tags(text):
    clean = re.compile('<.*?>')
    return re.sub(clean, '', text)

# display the top words in each topic
def display_topics(model, feature_names, no_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print ("Topic %d:" % (topic_idx))
        print (" ".join([feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]]))

# univariate t-test for difference in means
def t_diff_means(sample_1,sample_2):
    xbar_1 = np.mean(sample_1)
    xbar_2 = np.mean(sample_2)
    shat_1 = np.std(sample_1)
    shat_2 = np.std(sample_2)
    n_1 = len(sample_1)
    n_2 = len(sample_2)
    t = (xbar_1 - xbar_2)/np.sqrt(((shat_1**2)/n_1) + ((shat_2**2)/n_2))
    if t > 0:
        p = 1 - norm.cdf(t)
    elif t <= 0:
        p = norm.cdf(t)
    return [t,p]

############################
# import and clean data

# 37 womens' articles
womens_data = pd.read_csv('data/womens_articles.csv',index_col=0) 
womens_data['type'] = 'womens'

# 50 mens' articles
mens_data = pd.read_csv('data/mens_articles.csv',index_col=0)
mens_data['type'] = 'mens'

# 87 articles total
data = womens_data.append(mens_data).reset_index(drop=True)  

# need to get rid of any html, css, etc. in the articles
data['text'] = data['text'].apply(remove_html_tags)

# expand out contractions so that stop words can be fully stripped
data['text'] = data['text'].apply(contractions.fix)

############################
# preparations for LDA

# get documents for LDA
documents = list(data['text'].values)

# generate feature count vectors, i.e. counts for each vocab word in each document in the corpus
num_features = 2000 # use 2000 (or fewer) words for LDA

# max_df: document frequency threshold; remove words which appear in over 95% of articles, i.e. corpus-specific stop words
# min_df: ignore words that appear in fewer than 2 documents
# also performs pre-processing and removes stop words
tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2, max_features=num_features, stop_words='english')

# create document-feature matrix, with (no. documents) rows and (no. words) columns
# entry [i,j] is the number of times word j appears in document i
tf = tf_vectorizer.fit_transform(documents) 

# obtain the 1000 words in the document-feature matrix
tf_feature_names = tf_vectorizer.get_feature_names() 
print(tf_feature_names)

############################
# run LDA

# choose number of topics 
# NOTE: this decision is arbitary. future work may apply a systematic way of choosing this number
num_topics = 10

# train LDA model on corpus and return document-topic matrix
# random_state is seed used by random number generator
lda_model = LatentDirichletAllocation(n_components=num_topics,random_state=7).fit(tf)

# display the 10 top words in each topic, where "top" means highest p(word|topic)
num_top_words = 15
display_topics(lda_model, tf_feature_names, num_top_words) 

# NOTE: topic labels in quotations are subjectively assigned by the author
# Topic 0 ("Women"): hamstrings quads glutes works core muscles women legs work body stronger having help training week
# Topic 1 ("Bootcamp/group workouts"): right partner kettlebell bring reach group working butt sky walk way left training able pushing
# Topic 2 ("Hypertrophy"): biceps work training muscle exercises weight curls muscles growth failure use form great doing grip
# Topic 3 ("Functional fitness"): body plank jump high position push rope core upper glutes work like weight exercise rate
# Topic 4: ("Form") weight reps exercise time leg body just set rep make want work muscle position movement
# Topic 5: ("Arm training") weight reps exercise triceps arm arms bar pull set body shoulders work elbows press sets
# Topic 6: ("Weekend warrior") says foot ballenger weight biceps sets reps stairs need muscle stair training way body mobility
# Topic 7 ("Intensity"): failure drop perform set weight double time dropset rep shoulder triple total repeat end raises
# Topic 8 ("Push/chest workout"): chest body training reps weight set rep exercises push muscle just use time workouts sets
# Topic 9 ("Timing"): reps weight rest just sets time exercises minutes set like use body training 10 perform

############################
# analysis 

# goal: investigate whether there are significant differences in the topic probabilities for "womens" articles vs. "mens".
# null hypothesis: "womens" and "mens" articles have topic probabilities drawn from the same Dirichlet distribution
# alternative hypothesis: "womens" and "mens" articles are different categories, such that their topic probabilities are drawn from unique Dirichlet distributions
# NOTE: for a rigorous treatment of collections, see Paul, Michael. "Cross-collection topic models: Automatically comparing and contrasting text." Urbana 51 (2009): 61801

# get document-topic matrix
doc_top_mat = lda_model.transform(tf)

# get topic distributions for mens' and womens' articles
doc_top_womens = doc_top_mat[data['type'] == 'womens'] 
doc_top_mens = doc_top_mat[data['type'] == 'mens'] 

# looking at the above fitted topic distributions, we can see that the fitted probabilities of a given topic being drawn have highly skewed distributions across the set of articles.
# since the fitted probabilities are nowhere near normally distributed, joint inference on the mean topic probabilities (i.e. the alpha parameter of the Dirichlet distribution) between mens and womens articles will be difficult.
# however, we can at least compare the mean topic probabilities across mens and womens articles in a univariate test.
# e.g.: is the average probability of a word from the "Women" category in womens' articles greater than the average probability of a word from the "Women" category in mens' articles?

# we investigate the above hypotheses using a univariate t-test for difference in means
topics = ['Women','Bootcamp','Hypertrophy','Functional fitness','Form','Arm training','Weekend warrior','Intensity','Push workout','Timing']
for i in list(range(num_topics)):
    test = t_diff_means(doc_top_womens[:,i],doc_top_mens[:,i])
    print('Test of H0: E[prob. of drawing topic \"' + topics[i] + '\" for a word in womens\' articles] = E[prob. of drawing topic \"' + topics[i] + '\" for a word in mens\' articles] \n t-stat: ' + str(test[0]) + '; p-val: ' + str(test[1]))

# reject the null in the following cases:
# topic = "Functional fitness": avg. probability of a drawing this topic for a given word is significantly greater in womens' articles vs. mens'
# topic = "Arm training": avg. probability of drawing this topic for a given word is significantly greater in mens' articles vs. womens'
    
# it is interesting to note that the topic (subjectively) labeled "Women" does not quite reject the null (p ~= 0.07)
    
# the above provides some basic evidence that the topic probabilities for mens' and womens' articles may be drawn from different Dirichlet distributions.
