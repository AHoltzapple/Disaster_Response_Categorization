import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    '''
    Parameters
    ----------
    messages_filepath : str
        Path to text or csv file with message inputs.
    categories_filepath : str
        Path to text or csv file with category outputs.

    Returns
    -------
    df : pandas.DataFrame
        Dataframe of merged message and category data from specified files.
    '''
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = pd.merge(messages, categories, how = 'inner', on = 'id')
    return df

def clean_data(df):
    '''
    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe of message and category data. Output from load_data().

    Returns
    -------
    df : pandas.DataFrame
        Cleaned dataframe with duplicates removed and categories in columns.
    '''
    categories = df['categories'].str.split(';', expand=True)
    row = categories.iloc[0]
    category_colnames = row.apply(lambda x: x[:-2])
    categories.columns = category_colnames
    for column in categories:
        # set each value to be the last character of the string (numeric value)
        categories[column] = categories[column].apply(lambda x: x[-1])
        # convert column from string to integer
        categories[column] = categories[column].astype(int)
    df = df.drop('categories', axis=1)
    df = pd.concat([df, categories], axis=1)
    df = df.drop_duplicates()
    # related column has some rows with value of 2, information on the meaning
    # of this value is not available, therefore those rows are dropped to
    # maintain binary data in all categories
    df = df.loc[df['related'] != 2]
    print(str('{} duplicates after cleaning.').format(df.duplicated().sum()))
    print(str('{} missing category values after cleaning.').format(categories.isnull().sum().sum()))
    return df

def save_data(df, database_filename):
    '''
    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe to save to SQL database. Output from clean_data().
    database_filename : str
        Location to save database file to.

    Returns
    -------
    None.
    Saves input dataframe to specified SQL database.
    '''
    engine = create_engine('sqlite:///'+database_filename)
    # saves to 'messages' table by default
    df.to_sql('messages', engine, index=False, if_exists='replace')
    
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