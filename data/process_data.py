import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    
    '''
Loading the data from a path file

Input:
messages_filepath- it is used to import the database from csv file. 
categories_filepath- it is used to import the database from csv file

Output:
df- Return df after merging both csv files messages_filepath and categories_filepath.
    '''
    messages = pd.read_csv(messages_filepath)
    categories=pd.read_csv(categories_filepath)
    df = messages.merge(categories, left_on='id', right_on='id', how='outer')
    
    return df


def clean_data(df):
    '''
Input: 
df- processing the df dataset 

Output:
Return a processed df variable after lowering cased,stripping, tokenizing and lemmatizing 

    '''
   

    categories = df['categories'].str.split(';', expand = True)
    First_row = categories.loc[0]
    category_colnames = First_row.apply(lambda x: x[:-2])
    categories.columns = category_colnames
    for i in categories:
        categories[i] = categories[i].str[-1]
        categories[i] = categories[i].astype(int)
    df.drop('categories', axis = 1, inplace = True)
    df = pd.concat([df, categories], axis = 1)
    df.drop_duplicates(inplace = True)
    df = df[df['related'] != 2]
    return df


def save_data(df, database_filename):
    '''

Input:
df--- model that created after training, testing and evaluating 
database_filename---- File-path that the model should be saved in.

    '''
    
    engine = create_engine('sqlite:///'+database_filename)
    df.to_sql('Disasters', engine,index=False)
    pass  


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()