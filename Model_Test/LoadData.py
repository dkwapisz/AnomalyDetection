import csv
import nltk
from keras.preprocessing.text import Tokenizer
import numpy as np

nltk.download('punkt')


filepath = '../Model_Test/Data/SMSSpamCollection'

with open(filepath, encoding='utf8') as file:
    lines = file.readlines()
    lines = [line.rstrip() for line in lines]

sentencesOnlyAlpha = []

for sentence in lines:
    tokens = nltk.word_tokenize(sentence)
    words = [word for word in tokens if word.isalpha() or word.isalnum()]
    sentencesOnlyAlpha.append(words)

countHam = 0
countSpam = 0

with open('../Model_Test/Data/Spam_OnlyAlphaAndDigit.csv', 'w', newline='', encoding='utf8') as csvfile:
    fieldnames = ['label', 'sentence']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

    for sentence in sentencesOnlyAlpha:
        writer.writerow({'label': sentence[0], 'sentence': ' '.join(sentence[1:])})

        if sentence[0] == 'spam':
            countSpam += 1
        elif sentence[0] == 'ham':
            countHam += 1

print("Spams: {count}".format(count=countSpam))  # 747, (447, 150, 150) -> (train, val, test)
print("Normal: {count}".format(count=countHam))  # 4827, (2897, 985, 985) -> (train, val, test)

justWords = []

for i in range(0, len(sentencesOnlyAlpha), 1):
    for j in range(0, len(sentencesOnlyAlpha[i]), 1):
        if sentencesOnlyAlpha[i][j] != 'ham' and sentencesOnlyAlpha[i][j] != 'spam':
            justWords.append(sentencesOnlyAlpha[i][j])

tokenizer = Tokenizer(num_words=10000)
tokenizer.fit_on_texts(justWords)

#print(tokenizer.word_index)

def getData():
    x_train = []
    y_train = []
    x_val = []
    y_val = []
    x_test = []
    y_test = []
    #print([" ".join(sentencesOnlyAlpha[0][1:])])

    for i in range(0, len(sentencesOnlyAlpha), 1):
        temp = [tokenizer.texts_to_sequences([" ".join(sentencesOnlyAlpha[i][1:])])]

        if 0 < i < 3344:
            if sentencesOnlyAlpha[i][0] == 'ham':
                y_train.append(0)
            elif sentencesOnlyAlpha[i][0] == 'spam':
                y_train.append(1)
            x_train.append(temp[0][0])

        elif 3344 <= i < 4459:
            if sentencesOnlyAlpha[i][0] == 'ham':
                y_val.append(0)
            elif sentencesOnlyAlpha[i][0] == 'spam':
                y_val.append(1)
            x_val.append(temp[0][0])

        elif 4459 <= i < 5574:
            if sentencesOnlyAlpha[i][0] == 'ham':
                y_test.append(0)
            elif sentencesOnlyAlpha[i][0] == 'spam':
                y_test.append(1)
            x_test.append(temp[0][0])

    length = tokenizer.num_words

    return x_train, y_train, x_val, y_val, x_test, y_test, length