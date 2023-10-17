import numpy as np
import pandas as pd
import nltk
import string
from nltk.corpus import stopwords
from nltk.tokenize import (TreebankWordTokenizer,
                           word_tokenize,
                           wordpunct_tokenize,
                           TweetTokenizer,
                           MWETokenizer)
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import jaccard_score
from nltk.probability import FreqDist
import string

# Stop words download
#nltk.download('stopwords')

# def plot_confusion_heatmap(y_true, y_pred, title):
#     cm = confusion_matrix(y_true, y_pred)
#     print(cm)
#     plt.figure(figsize=(6, 4))
#     sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['NEGATIVE', 'POSITIVE'], yticklabels=['NEGATIVE', 'POSITIVE'])
#     plt.xlabel('Predicted')
#     plt.ylabel('Actual')
#     plt.title(title)
#     plt.show()

def extract_polarity(label):
    if 'POSITIVE' in label:
        return 'POSITIVE'
    elif 'NEGATIVE' in label:
        return 'NEGATIVE'
    
def extract_td(label):
    if 'TRUTHFUL' in label:
        return 'TRUTHFUL'
    elif 'DECEPTIVE' in label:
        return 'DECEPTIVE'

def preprocessing(review): 
    
    # Define the english stopwords
    stop_words = set(stopwords.words('english'))
    
    # Define the tokenizer
    tokenizer = TreebankWordTokenizer()
    
    # Define the lemmatizer
    lemmatizer = WordNetLemmatizer()
    
    # Tokenize each word
    words = tokenizer.tokenize(review)
    
    # Convert to lowercase
    words = [word.lower() for word in words]
    
    # Remove ponctuation
    words = [''.join(c for c in word if c not in string.punctuation) for word in words]
    
    # Delete the stop words
    filtered_words = [word for word in words if word not in stop_words] 
    
    # Apply lemmatization
    lemmatized_words = [lemmatizer.lemmatize(word) for word in filtered_words]

    processed_review = ' '.join(lemmatized_words)
    
    return processed_review


# Define the train.txt source path
training_data_file = 'train.txt'

# Decompose in label and review
with open(training_data_file, 'r') as file:
    training_data = [line.strip().split('\t') for line in file]
    
# Extract labels and reviews
labels, reviews = zip(*training_data)

# Preprocess the text
preprocessed_reviews = [preprocessing(review) for review in reviews]

# Define x_train and y_train
x_train = preprocessed_reviews
y_train = labels

# Define a dataset dataframe
dataset_df = pd.DataFrame({'review': x_train, 'label': y_train})
print(dataset_df.head())
class_counts = dataset_df['label'].value_counts()
print(class_counts)

# TFID Vectorize
tfidf_vectorizer = TfidfVectorizer()
x_train_tfidf = tfidf_vectorizer.fit_transform(x_train)

# Extract y train for polarity classifier
y_train_pol = [extract_polarity(label) for label in y_train]
# Extract y train for truthful/deceptive classifier
y_train_td = [extract_td(label) for label in y_train]

################################ CROSS VALIDATION ##################################

cross_val_size = 15

########## Polarity Classifier ###########
print("---------- Polarity Classifier -----------")

classifier_pol = MultinomialNB(alpha=0.65)
scores_polarity = cross_val_score(classifier_pol, x_train_tfidf, y_train_pol, cv=cross_val_size, scoring='accuracy')
print(f'Cross-Validation Acc : {scores_polarity}')
mean_accuracy = scores_polarity.mean()
std_accuracy = scores_polarity.std()
print(f'Mean Accuracy: {mean_accuracy}')
print(f'Standard Deviation: {std_accuracy}\n')

 ########## Truthful/Deceptive Classifier ###########  
print("---------- Deceptive Classifier -----------")
classifier_td = SVC(C=1.5, kernel='rbf')

scores = cross_val_score(classifier_td, x_train_tfidf, y_train_td, cv=cross_val_size, scoring='accuracy')
print(f'Cross-Validation Acc: {scores}')
mean_accuracy = scores.mean()
std_accuracy = scores.std()
print(f'Mean Accuracy: {mean_accuracy}')
print(f'Standard Deviation: {std_accuracy}\n')


################################ Training and Validation ######################################
# 80% training, 20% validation
x_train, x_val, y_train, y_val = train_test_split(x_train_tfidf, labels, test_size=0.2, random_state=3, stratify=labels)

y_train_pol = [extract_polarity(label) for label in y_train]
y_train_td = [extract_td(label) for label in y_train]
# y_train_pol = [0 if label == 'NEGATIVE' else 1 for label in y_train_pol]
# y_train_td = [0 if label == 'DECEPTIVE' else 1 for label in y_train

# Check the number of validation samples per category
class_counts_val = pd.Series(y_val).value_counts()
print(class_counts_val)

########## Polarity Classifier ###########

classifier_pol = MultinomialNB(alpha=0.65)
classifier_pol.fit(x_train, y_train_pol)
y_pred_pol = classifier_pol.predict(x_val)

########## Truthful/Deceptive Classifier ###########
  
classifier_td = SVC(C=1.4, kernel='rbf')
classifier_td.fit(x_train, y_train_td)
y_pred_td = classifier_td.predict(x_val)

################ Ensemble Classifier: Polarity Classifier + Truthful/Deceptive Classifier ################ 
print("---------- Ensemble Classifier -----------")
y_pred_ensembled = [f"{td}{pol}" for td, pol in zip( y_pred_td, y_pred_pol)]

# Accuracy
accuracy = accuracy_score(y_pred_ensembled, y_val)
print(f'Ensemble Approach Accuracy: {accuracy}')

# Classification Report
classification_rep = classification_report(y_val, y_pred_ensembled)
print(f'Classification report for Ensemble model:\n{classification_rep}')

# Jaccard Index
jaccard_index = jaccard_score(y_val, y_pred_ensembled, average='micro')
print(f'Jaccard index: {jaccard_index*100}')


# classifiers = [
#     ('Multinomial Naive Bayes', MultinomialNB(alpha=1)),
#     ('Logistic Regression', LogisticRegression(max_iter=1000)),
#     ('Support Vector Machine', SVC(C=1.4, kernel='rbf')),
#     ('Random Forest', RandomForestClassifier())

# ]

# for classifier_name, classifier in classifiers:
#     scores = cross_val_score(classifier, x_train_tfidf, y_train, cv=15, scoring='accuracy')

#     print(f'Cross-Validation Acc {classifier_name}: {scores}')

#     mean_accuracy = scores.mean()
#     std_accuracy = scores.std()
#     print(f'Mean Accuracy: {mean_accuracy}')
#     print(f'Standard Deviation: {std_accuracy}\n')
    
    
# for classifier_name, classifier in classifiers:
#     classifier.fit(x_train_tfidf, y_train)
#     y_pred = classifier.predict(x_val_tfidf)
#     accuracy = accuracy_score(y_val, y_pred)
#     print(f'{classifier_name} Accuracy: {accuracy}')
#     #classification_rep = classification_report(y_val, y_pred, target_names=['NEGATIVE', 'POSITIVE'])
#     classification_rep = classification_report(y_val, y_pred, target_names=['DECEPTIVE', 'TRUTHFUL'])
#     print(f'Classification report for {classifier_name}:\n{classification_rep}')
#     #plot_confusion_heatmap(y_val, y_pred, f'Confusion Matrix for {classifier_name}')
    
#     # Find misclassified examples
#     misclassified_indices = [i for i, (true, pred) in enumerate(zip(y_val, y_pred)) if true != pred]
    
#     # Print misclassified examples
#     # print(f'Misclassified Examples for {classifier_name}:\n')
#     # for index in misclassified_indices:
#     #     print(f'True Label: {y_val.iloc[index]}, Predicted Label: {y_pred[index]}')
#     #     print(f'Review: {x_val.iloc[index]}\n')
        
