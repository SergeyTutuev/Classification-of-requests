def sygma(x):
	return 1/(1+np.exp(-x))
	
def prim_sygma(x):
	return x*(1-x)	

from nltk import word_tokenize
from nltk.corpus import stopwords
from pandas import read_csv, DataFrame
from gensim.models import FastText
import numpy as np

vector_model=FastText(vector_size=400, window=3, min_count=3)

dataset=read_csv("D:/Downloads/train.csv")
titles=list(dataset.title)
labels=np.eye(2)[list(dataset.label)]

extra_words=[",",".","\"",";",":","-","--","+","*","\'","!","?","#","@","=","(",")","[","]","÷","_","×","&","^"]+ stopwords.words()

sentences=[]
maxlen=0
for i in titles:
	words=word_tokenize(str(i).lower())
	sentence=set()
	for j in words:
		if j not in extra_words:
			sentence.add(j)
					
	maxlen=max(maxlen,len(sentence))		
	sentences.append(list(sentence))

vector_model.build_vocab(corpus_iterable=sentences)
vector_model.train(corpus_iterable=sentences, total_examples=len(sentences), epochs=30)

print(1)

def vectorize(x):
	vector=np.zeros((400))
	for i in x:
		vector+=(vector_model.wv[i])
	vector/=(len(x)+(len(x)==0))
	return vector
	
layers_sizes=[400,25,2]
count_layers=len(layers_sizes)
layers=[]
epochs=3
step_learning=0.01

matrix_of_weights=[np.random.uniform(-0.5,0.5,(layers_sizes[i],layers_sizes[i-1])) for i in range(1,count_layers)]
matrix_of_bias=[np.zeros((layers_sizes[i],1)) for i in range(1,count_layers)]
layers=[np.zeros((layers_sizes[i],1)) for i in range(count_layers)]

for i in range(epochs):
	e_correct=0
	
	for sentence,label in zip(sentences,labels):
		
		vector=vectorize(sentence)
		
		layers[0]=vector.reshape((-1,1))
		label=label.reshape((-1,1))
		
		for k in range(1,count_layers):
			layers[k]=sygma(matrix_of_weights[k-1] @ layers[k-1]+matrix_of_bias[k-1])
			
		err=2*(layers[-1]-label)
		
		e_correct+=int(np.argmax(layers[-1])==np.argmax(label))
		
		for k in range(count_layers-2,-1,-1):
		   	
		   	matrix_of_weights[k]-=step_learning*err @ np.transpose(layers[k])
		   	matrix_of_bias[k]-=step_learning*err
		   	err=np.transpose(matrix_of_weights[k])@ err * prim_sygma(layers[k])
		   	
	print(round((e_correct/len(sentences))*100,3))

testing=read_csv("D:/Downloads/test.csv")
testing_titles=list(testing.title)

answer={"ID":list(testing.ID),"label":[]}

for i in testing_titles:
	words=word_tokenize(str(i).lower())
	sentence=set()
	for j in words:
		if j not in extra_words:
			sentence.add(j)
	vector=vectorize(list(sentence))

	layers[0]=vector.reshape((-1,1))
	for k in range(1,count_layers):
		layers[k]=sygma(matrix_of_weights[k-1] @ layers[k-1]+matrix_of_bias[k-1])

	answer["label"].append(np.argmax(layers[-1]))

answer_df=DataFrame(answer)
answer_df.to_csv("file_answer.csv")

	
    







