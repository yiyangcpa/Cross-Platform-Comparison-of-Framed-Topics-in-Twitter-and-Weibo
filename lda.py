#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# coding:utf-8
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import lda
import numpy as np
import numpy
import pyLDAvis
import pyLDAvis.sklearn
import pyLDAvis.gensim

corpus = []
for line in open('file_path','r').readlines():
    corpus.append(line.strip())
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(corpus)
word = vectorizer.get_feature_names()
print('特征个数:',len(word))

for n in range(len(word)):
    print(word[n]),
print(' ')
print (X.toarray())


transformer = TfidfTransformer()
print (transformer)
tfidf = transformer.fit_transform(X)
print (tfidf.toarray())
weight = tfidf.toarray()

model = lda.LDA(n_topics = 30, n_iter = 1000,random_state = 1)
model.fit(X)
doc_topic = model.doc_topic_
print("shape:{}".format(doc_topic.shape))
for n in range(1):
    topic_most_pr = doc_topic[n].argmax()
    print("文档:{},主题:{}".format(n,topic_most_pr))
    
a=doc_topic
numpy.savetxt('file_name.csv', a, delimiter = ',') 
    
word = vectorizer.get_feature_names()
topic_word = model.topic_word_
for w in word:
    print (w),
print (' ')
n = 30 
for i,topic_dist in enumerate(topic_word):
    topic_words = np.array(word)[np.argsort(topic_dist)][:-(n+1):-1]
    print(u'*Topic {}\n- {}'.format(i,' '.join(topic_words)))

print("shape:{}".format(topic_word.shape))
print(topic_word[:,:1]) 
for n in range(1): 
    sum_pr = sum(topic_word[n,:])
    print("topic:{}sum:{}".format(n,sum_pr))
