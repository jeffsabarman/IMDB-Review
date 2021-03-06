#Import the packages
import pandas as pd
from textblob import TextBlob

#Import the data
data = pd.read_csv('D:/Dataset/IMDB_Dataset.csv')
review = data['review']

#This is the list and sentiment to hold the sentiment
predict = []
sentiment = " "

#We use for loop to get every review in the data dataframe
for rev in range(0, len(review)):
    sentence = review.iloc[rev]
    text = TextBlob(sentence)
    score = text.sentiment.polarity
    if score >= 0:
        sentiment = 'positive'
        predict.append(sentiment)
    else:
        sentiment = 'negative'
        predict.append(sentiment)

#This is too look at the accuracy of TextBlob sentiment
right = 0
for i in range(0,len(review)):
    if predict[i] == data['sentiment'].values[i]:
        right +=1
print('Accuracy : ', right/len(review))
#Accuracy : 0.68828
