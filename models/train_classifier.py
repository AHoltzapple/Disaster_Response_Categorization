import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import nltk
nltk.download(['punkt', 'wordnet'])
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import KNeighborsClassifier
# from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import sklearn.metrics as skm
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import pickle

def load_data(database_filepath):
    engine = create_engine(r'sqlite:///'+database_filepath)
    df = pd.read_sql("SELECT * FROM mess", engine)
    X = df['message']
    Y = df.drop(['id','message','original','genre'], axis=1)
    category_names = list(Y.columns)
    return X, Y, category_names

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok)
        clean_tok = clean_tok.lower()
        clean_tok = clean_tok.strip()
        # lemmatize, normalize case, and remove leading/trailing white space
        clean_tokens.append(clean_tok)
    return clean_tokens

def build_model():
    model = Pipeline([
    ('vect', CountVectorizer(tokenizer=tokenize)),
    ('tfidf', TfidfTransformer()),
    ('clf', MultiOutputClassifier(KNeighborsClassifier()))
    ])
    return model

def evaluate_model(model, X_test, Y_test, category_names):
    y_pred = model.predict(X_test)
    for i, category in enumerate(category_names):
        y_test_label = Y_test[category]
        y_pred_label = y_pred[:,i]
        precision, recall, fscore, support = skm.precision_recall_fscore_support(y_pred=y_pred_label, y_true=y_test_label, 
                                                                                 average='weighted', zero_division=1)
        accuracy = skm.accuracy_score(y_pred=y_pred_label, y_true=y_test_label)
        print('\nCategory: '+category_names[i])
        print('Accuracy: '+ str(round(accuracy,2))+'\tPrecision: ' + str(round(precision,2)))
        print('Recall: ' + str(round(recall,2))+'\tF-Score: ' + str(round(fscore,2)))
    print('\nHamming Loss (lower is better): '+ str(round(np.sum(np.not_equal(np.array(Y_test), y_pred))/float(Y_test.size),3)))

def save_model(model, model_filepath):
    pickle.dump(model, open(model_filepath, 'wb'))

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