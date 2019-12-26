import sys
# import libraries
import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])
import re
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sqlalchemy import create_engine
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from nltk.tokenize import word_tokenize, RegexpTokenizer
from sklearn.metrics import precision_recall_fscore_support
from sklearn.pipeline import Pipeline
from sklearn.metrics import make_scorer, accuracy_score, f1_score, fbeta_score, classification_report
import pickle
import warnings
warnings.filterwarnings("ignore")


def load_data(database_filepath):

    '''

Loading the data from a path file

Input:
------- database_filepath- it is used to import the database that we want to use. 

Output:
X: -------Return the features variable, so in this database, messages column will be returning from the dataset.
Y: -------Return the Label variables, so in this database, categories will be returning from the dataset.

    '''


    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql_table('Disasters', engine)
    df.head()
    # drop coulmn with null values
    df = df[~(df.isnull().any(axis=1))|((df.original.isnull())&~(df.offer.isnull()))]
    # Features 
    X= df['message'].values

    #Target 
    Y=df.iloc[:,4:]
    
    category_names=Y.columns
    
    return X , Y , category_names


url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'


    
def tokenize(text):

    '''
Input: 
text------- processing the text 

Output:
Return a processed text variable after lowering cased,stripping, tokenizing and lemmatizing 
    '''
   
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")
        
    # remove all punctuation
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = word_tokenize(text)
    
    # LEMMATIZING
    
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

def build_model():

    '''
Building a model that we created for training the dataset 
Return-----pipeline model.
    '''
    model_cl=MultiOutputClassifier(AdaBoostClassifier())
    model = Pipeline([
    ('vect', CountVectorizer(tokenizer=tokenize)),
    ('tfidf', TfidfTransformer()),
    ('clf', model_cl)
    ])
    parameters = {'vect__ngram_range':((1, 1), (1, 2)),
              'vect__max_features': (None, 5000),
              'tfidf__use_idf': (True, False),
              'clf__estimator__learning_rate': [0.1,0.2],
              'clf__estimator__random_state':[10, 4,None]}
    model= GridSearchCV(model, parameters)
    return model
    
    
def evaluate_model(model, X_test, Y_test, category_names):

    '''
Input:
model------- the model that need to be evaluated after we trained 
X_test------ Input Features 
Y_test------ Label Features
category_names--- The list of categories 

Output:
This function doesn't return anything. But F1-score, recall, and precision will be printed out 

    '''
    
    Y_pred = model.predict(X_test)
    Accuracy_overall=(Y_pred == Y_test).mean().mean()
    print('Avg Accuracy Overall {0:.3f}% \n'.format(Accuracy_overall*100))
    Y_pre = pd.DataFrame(Y_pred, columns = Y_test.columns)
    for i in Y_test.columns: 
        print('_______________________________________________________________\n\n')
        print('FEATURES: {}\n'.format(i))
        print(classification_report(Y_test[i],Y_pre[i]))
    
    pass

    
   
def save_model(model, model_filepath):
    '''

Input:
model--- model that created after training, testing and evaluating 
model_filpath---- File-path that the model should be saved in.
    '''
    
    filename = model_filepath
    pickle.dump(model, open(filename, 'wb'))
    pass



def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()