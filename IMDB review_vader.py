#Import the packages
import pandas as pd #To read the csv file
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer #To do Sentiment Analysis

#Import the data
data = pd.read_csv('D:/Dataset/IMDB_Dataset.csv')
review = data['review']

#Initialize analyser
analyser = SentimentIntensityAnalyzer()

#This is the list and sentiment to hold the sentiment
predict = []
sentiment = " "

#We use for loop to get every review in the data dataframe
for rev in range(0, len(review)):
    sentence = review.iloc[rev]
    score = analyser.polarity_scores(sentence)
    com_score = score.get('compound') #Compound is the overall score from the VADER Sentiment
    if com_score >= 0:
        sentiment = 'positive'
        predict.append(sentiment)
    else:
        sentiment = 'negative'
        predict.append(sentiment)

#This is too look at the accuracy of VADER Sentiment
right = 0
for i in range(0,len(review)):
    if predict[i] == data['sentiment'].values[i]:
        right +=1
print('Accuracy : ', right/len(review))
#Accuracy : 0.69634
