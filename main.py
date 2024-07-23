'''

import tensorflow as tf

string = tf.Variable('This is string', tf.string)

number1 = tf.Variable(12 , tf.int16)
number2 = tf.Variable([1,2,3],tf.int16)
floating = tf.Variable(12.4 , tf.float64)
n1 = tf.ones([1,2,3])
n2 = tf.ones([5,5,5,5])
#Rank and degree are the dimension in tensorflow. Shape are the elements in tensor flow. Reshape change the dimension of the matrix
print(tf.rank(string))
#print(tf.rank(number1))
#print(tf.rank(number2))
#print(tf.shape(number2))
#print(n1)
#print(tf.reshape(n1,[2,1,3]))
#print(n2)


'''
#REGULAR EXPRESSION




import re
text =  'Hi there! Thank you for your interest in our products. If you have any questions or need assistance, please feel free to contact us at (555)-123-4567. Our customer support team is available Monday to Friday from 9434544543 We are here to help you with anything you may need. Looking forward to serving you soon!'
pattern = '\(\d{3}\)-\d{3}-\d{4}|\d{10}'
match = re. findall( pattern,text)
#print(match)


text =     ''' Topic 1 Ovewrview
           sasas as aksmasasasas akamskaksHi there! Thank you for your interest in our products. If you have any questions or need assistance, please feel free to contact us at (555)-123-4567. Our customer support team is available Monday to Friday from 9434544543 We are here to help you with anything you may need. Looking forward to serving you soon! pattern
           Topic 2 Car Machine
           sasas as aksmasasasas akamskaks  'Hi there! Thank you for your interest in our products. If you have any questions or need assistance, please feel free to contact us at (555)-123-4567. Our customer support team is available Monday to Friday from 9434544543 We are here to help you with anything you may need. Looking forward to serving you soon!pattern '''

#3pattern = 'Topic \d ([^\n]*)'
#match = re. findall( pattern,text)
#print(match)

text =  ''' If you have any questions or need assistance, please feel free to contact us at (555)-123-4567. Our customer FY2022 Q1 support team is available Monday to Friday from 9434544543 We are here to help you with anything you may need.
Looking forward to serving you Fy2023 Q4  soon!'pattern
 '''

#pattern = 'FY\d{4} Q[1234]'
#same as above pattern = 'FY\d{4} Q[1-4]'
#match = re. findall( pattern,text,flags=re.IGNORECASE)
#print(match)

text =  ''' If you have any questions or need price $3.342323 billion assistance, please feel free to contact us at (555)-123-4567. Our customer FY2022 Q1 support team is available Monday to Friday from 9434544543 We are here to help you with anything you may need amd price $4 billion.
Looking forward to serving you FY2023 Q4  soon!'pattern
 '''

#pattern = '\$[\d\.]+'
#same as above pattern = '\$[0-9\.]+'
#match = re. findall( pattern,text)
#print(match)

text =  ''' If you have any questions or need price assistance, please feel free to contact us at (555)-123-4567. Our customer FY2022 Q1 was $3.342323 billion support team is available Monday to Friday from 9434544543 We are here to help you with anything you may need amd .
Looking forward to serving you FY2023 Q4 price $4 billion soon!'pattern
 '''

#pattern = '(FY\d{4} Q[1-4])[^\$]+(\$[\d.]+)'
# if do like this (FY\d{4} Q[1-4])|(\$[\d.]+) then they belong to different matches
#match = re.findall( pattern,text)
# search method fund only 1 occurence match = re.search( pattern,text)
#print(match)

#pattern= '[a-zA-Z_0-9]*[a-z]'
#text='''Born    Lahore pakistan'''
#pattern = 'Born.*'


#text = '''Born    Lahore pakistan June 23 2023 (age 23)'''
#pattern = 'Born.*\n(.*)\('
#match = re.findall(pattern,text)
#print(match)

#def getinfo(text):
#  born_place =   re.findall('Born.*',text)
#  born_date =  re.findall( 'Born.*\n(.*)\(',text)
#  return { 'born_place' : born_place , 'born_date ': born_date   }

#text = '''Born    Lahore pakistan June 23 2023 (age 23)'''
#print(getinfo(text))






''''#agar ap apna final prize ka correlation check karna chahta ha ka dosra features ka kya effect ha hamara final

import numpy as np
import pandas as pd
data = pd.read_csv('data.csv')
print(data)
corr_matrix = data.corr()
print(corr_matrix['MEDV'].sort_values(ascending=False))
'''
import keras.layers

#import pandas as pd
#import numpy as np
#import sklearn as sk
#from sklearn import preprocessing
#import random
#import matplotlib.pyplot as plt
'''
#for ligistic regression we will use loss= binary_crossentropy
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
(x_train,y_train) , (x_test, y_test) = keras.datasets.mnist.load_data()
#for standaard scale
x_train = x_train/255
x_test =x_test/255
x_train_flatten = x_train.reshape(len(x_train),28*28)
x_test_flatten = x_test.reshape(len(x_test),28*28)

#model = keras.Sequential([keras.layers.Dense(10,input_shape=(784,), activation='sigmoid')])
model = keras.Sequential([ keras.layers.Dense(100,input_shape=(784,), activation='relu'),
                           keras.layers.Dense(10, activation='sigmoid')])

model.compile(optimizer='adam' , loss='sparse_categorical_crossentropy' , metrics=['accuracy'] )
model.fit(x_train_flatten,y_train,epochs=5)
model.evaluate(x_test_flatten,y_test)
pre = model.predict(x_test_flatten)
#plt.matshow(x_test[0])
print(np.argmax(pre[0]))
#plt.show()
pre = [np.argmax(i) for i in pre]
cm = tf.math.confusion_matrix(y_test,pre)
plt.figure(figsize=(10,8))
sns.heatmap(cm ,annot=True , fmt='d')

plt.show()
'''

#GRADIENT DESCEMT: IT USE TO REDUCE AND LOSS ANDENHACE THE ACCURACY. It adjust the weight nnd biases after each epoche to reduce loss
#in start when we develop model we intialize weight and biased randomly like:
#model.Squential([ keras.layer.Dense(1 m input_shape(2)) ,activatiom='sigmoid',kernal_initalizer='ones',bias_intializer='zeroes'])
#TYPES:
#BATCH GRADIENT DECENT ARE USE ALL TRAINING SAMPLE NEFORE ADJUSTING WEIGHT
#STOCASTIC GRADIENT DECENT ARE USE RANDOM TRAINING SAMPLE NEFORE ADJUSTING WEIGHT
#MINI BATCH GRADIENT DECENT ARE USE BATCH OF  TRAINING SAMPLE NEFORE ADJUSTING WEIGHT


#MINI BATCH GRADIENT DECENT IMPLEMENTATION

'''
home_df = pd.read_csv('home.csv')
X = home_df[['area','bedroome']]
Y = home_df['price']
#print(X)
scaled_X= preprocessing.MinMaxScaler()  
scaled_Y = preprocessing.MinMaxScaler()

sx = scaled_X.fit_transform(X)
sy = scaled_Y.fit_transform(Y.values.reshape(Y.shape[0],1))
#print(sy)
no_of_feature = X.shape[1]
total_sample = X.shape[0]
w = np.ones(shape=(no_of_feature))
b =0
batch_size=5
epoche = 100
learning_rate = 0.05
sy = sy.reshape(sy.shape[0])
#print(sy)
cost_list =[]
epoche_list =[]
for i in range(epoche):
    #random_indices = np.random.permutation(total_samples)
    rand_index = np.random.permutation(total_sample)
    X_temp = sx[rand_index]
    y_temp = sy[rand_index]
    #print(y_temp)

    for j in range(0,total_sample,batch_size):
         X_batch = X_temp[j:j + batch_size]

         Y_batch = y_temp[j:j + batch_size]

         y_predicted = np.dot(w, X_batch.T) + b
         w_grad = -(2 / total_sample) * X_batch.T.dot(Y_batch - y_predicted)
         b_grad = -(2/total_sample) * np.sum(Y_batch - y_predicted)

         w = w - learning_rate * w_grad
         b = b - learning_rate * b_grad
         cost = np.mean(np.square(Y_batch - y_predicted))

         if(i%10):
             cost_list.append(cost)
             epoche_list.append(i)

#print(w,b,cost)

plt.xlabel("epoch")
plt.ylabel("cost")
plt.plot(epoche_list,cost_list)
#plt.show()


def pred(area,bedroom,w,b):
    #this scaled_X.transform give data into 2D format but you want value of price and badroom for price cal so you donvert into 1 D
    scale_X = scaled_X.transform([[area, bedroom]])[0]
    #print(scale_X)
    scaled_price = w[0] * scale_X[0] + w[1] * scale_X[1] + b
    #this give scaled_Y.inverse_transform you price in 2 D format if you want value then you convert into 0D abd extract price
    return scaled_Y.inverse_transform([[scaled_price]])[0][0]


print(pred(2600,4,w,b))
'''

'''
#Recurrent Neural Network
RNN stands for Recurrent Neural Network. It is a type of artificial neural network designed to
process sequential data, where the output at each step is dependent not only on the current input but
also on the previously processed inputs. RNNs are particularly well-suited for tasks involving
sequential data, such as natural language processing, speech recognition, time series analysis, and more.
#VANISHING GRADIENT AND EXPLODING GRADIENT
Vanishing and exploding gradients are issues that can occur during the training of deep neural networks,
particularly in recurrent neural networks (RNNs) and deep feedforward networks with many layers.
Vanishing Gradients:
In the context of vanishing gradients, the gradients of the loss function with respect to the
networks parameters become very small as they are backpropagated from the output layer to the initial layers.
This means that the updates to the early layers parameters become extremely small, effectively slowing down
the learning process for these layers. Consequently, the network struggles to learn long-range dependencies
or capture complex patterns that span over many time steps or layers
Exploding Gradients:
Conversely, exploding gradients occur when the gradients grow exponentially as they are backpropagated
through the network. This leads to very large updates to the networks parameters, causing the network
to diverge during training and fail to converge to a good solution.
#Long Short-Term Memory (LSTM) or Gated Recurrent Unit (GRU) cells in RNNs, 
# are designed to address the vanishing gradient problem 
#LSTM (Long Short-Term Memory) and GRU (Gated Recurrent Unit) are two types of recurrent neural network (RNN) 
architectures designed to address the vanishing gradient problem 
'''

'''
#WORD EMBEDING:
#TWO TYPES:
#1- Supervised
# 2- Self Supervised

#WORD EMBEDING USING LAYERS (TYPE 1)
import tensorflow
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Embedding

reviews = ['nice food',
        'amazing restaurant',
        'too good',
        'just loved it!',
        'will go again',
        'horrible food',
        'never go there',
        'poor service',
        'poor quality',
        'needs improvement']

sentiment = np.array([1,1,1,1,1,0,0,0,0,0])

vocal_length = 30
#encode all reveiws
encoded_review =  [one_hot(d,vocal_length) for d in reviews]

print(encoded_review)

max_length = 4
padding_review= pad_sequences(encoded_review,maxlen= max_length,padding='post')

embeded_size = 5
model = Sequential()
model.add(Embedding(vocal_length,embeded_size,input_length = max_length,name='embedding'))
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy' , metrics=['accuracy'])


x= padding_review
y= sentiment


#print(model.summary())

#model.fit(x,y,epochs=50 , verbose=0)

#loss ,accuracy = model.evaluate(x,y)
#print(loss,accuracy)

weight = model.get_layer('embedding').get_weights()[0]
print(len(weight))

print(weight[25])
print(weight[2])
'''

'''
#2- Self Supervised
#* Word2Vec
# - CBOW( Conyinous Bag of Worda)
# - Skip Gram

#Implement Word2Vec
import gensim
import json


data = {
    "reviews": reviews
}

filename = "review.json"

with open(filename, "w") as f:
    json.dump(data, f)
    
with open('review.json') as f:
    df = json.load(f)
#reviews_text =
#print(df)

reviews_text = df['reviews']
processed_reviews = [gensim.utils.simple_preprocess(review) for review in reviews_text]
print(processed_reviews)
#print(processed_reviews )
model = gensim.models.Word2Vec( window=10 , min_count=2 , workers=7)
model.build_vocab(processed_reviews,progress_per=1000)
model.train(processed_reviews,total_examples=model.corpus_total_words,epochs=model.epochs)
print(model.corpus_total_words)
#print(model.wv.most_similar('great'))
#print(model.wv.most_similar('phone'))
print(model.wv.similarity('phone' , 'phone'))
    
    '''

    #BERT (Bidirectional Encoder Representations from Transformers
''' 1- it use un Contextual Word Embeddings: 
BERT generates contextual word embeddings, meaning that the meaning of a word is dependent on its context within the sentence. 
 2- Pre-training and Fine-tuning:'''
'''preprocess_url = "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3": This URL points to a pre-trained BERT model that is specifically designed for text preprocessing. It is part of the BERT family of models and is tailored to process English text in a specific way. This preprocessing model takes raw text inputs and performs tokenization, converting the text into token IDs, adding special tokens like [CLS] and [SEP], and generating attention masks and input type masks.'''
'''encoder_url = "https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/4": This URL points to a pre-trained BERT encoder model. It is a larger BERT model with multiple layers and parameters that have been trained on a large corpus of text data for various NLP tasks. The encoder takes the preprocessed inputs from the preprocessing model and produces contextualized word embeddings or representations that capture the meaning of the text in the context of the surrounding words. '''
'''
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
import pandas as pd
import keras

df = pd.read_csv("spam.csv")
#df.head(5)
#df.groupby('Category').describe()
df['spam']=df['Category'].apply(lambda x: 1 if x=='spam' else 0)
df.head()
bert_preprocess = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3")
bert_encoder = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/4")
def get_sentence_embeding(sentences):
    preprocessed_text = bert_preprocess(sentences)
    return bert_encoder(preprocessed_text)['pooled_output']

e = get_sentence_embeding([
    "banana",
    "grapes",
    "mango",
    "jeff bezos",
    "elon musk",
    "bill gates"
]
)
from sklearn.metrics.pairwise import cosine_similarity
print(cosine_similarity([e[0]],[e[1]]))

#print(get_sentence_embeding(["500$ discount. hurry up", "Bhavin, are you up for a volleybal game tomorrow?"]))
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(df['Message'],df['spam'])
#print(X_train.head(4))

text_input = tf.keras.layers.Input(shape=() , dtype = tf.string , name='text' )
preprocessed_text = bert_preprocess(text_input)
output = bert_encoder(preprocessed_text)
#. Dropout is a regularization technique used to prevent overfitting in neural networks. It randomly sets 
#a fraction of input units to 0 at each update during training, which helps to prevent the model from relying too much 
#on specific features , 0.1 represents the dropout rate, which means 10% of the input units will be randomly set to 0 during training.


l = tf.keras.layers.Dropout(0.1, name="dropout")(output['pooled_output'])
l = tf.keras.layers.Dense(1, activation='sigmoid', name="output")(l)

model = tf.keras.Model(inputs = [text_input] , outputs = [l]  )
#tf.keras.utils.plot_model(model)
#print(model.summary())
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(X_train, y_train, epochs=5)
print(model.evaluate(X_test, y_test))

y_predicted = model.predict(X_test)
y_predicted = y_predicted.flatten() #it convert intp 1D
import numpy as np

y_predicted = np.where(y_predicted > 0.5, 1, 0)
from sklearn.metrics import confusion_matrix, classification_report

cm = confusion_matrix(y_test, y_predicted)
print(classification_report(y_test, y_predicted))
reviews = [
    'Enter a chance to win $5000, hurry up, offer valid until march 31, 2021',
    'You are awarded a SiPix Digital Camera! call 09061221061 from landline. Delivery within 28days. T Cs Box177. M221BP. 2yr warranty. 150ppm. 16 . p pÂ£3.99',
    'it to 80488. Your 500 free text messages are valid until 31 December 2005.',
    'Hey Sam, Are you coming for a cricket game tomorrow',
    "Why don't you wait 'til at least wednesday to see if you get your ."
]
model.predict(reviews)  # first three a like spamy so i get output is >0.5 and other < 0.5 that are not spamy


'''