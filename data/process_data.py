import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    '''
    This function load messages and categories data from
    CSV files into dataframe 
    
    Input:
        messages_filepath: CSV messages filepath
        categories_filepath: CSV categories filepath
    
    Output:
        df: dataframe with message and category data
    '''
    
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories, on='id')
    return df

def clean_data(df):
    '''
    This function splits the categories data into separate columns 
    
    Input:
        df: dataframe containing raw messages and categories information 
        
    Output:
        df: dataframe containing messages and clean categories values 
    '''
    
    # create a dataframe of the 36 individual category columns
    categories = df.categories.str.split(";",expand=True)
    
    # select the first row of the categories dataframe
    category_colnames = [w.split("-")[0] for w in list(categories.iloc[0])]
    
    # rename the columns of `categories`
    categories.columns = category_colnames
    
    # Convert category values to just numbers 0 or 1
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = [w[1] for w in categories[column].str.split("-")]
    
        # convert column from string to numeric
        categories[column] = categories[column].astype('Int64')
        
    # drop the original categories column from `df`
    df.drop(columns=['categories'],inplace=True)

    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df,categories],axis=1)
    
    # handling multivalues in related class
    df['related']=df['related'].map(lambda x: 1 if x == 2 else x)

    # drop duplicates
    df.drop_duplicates(keep='first',inplace=True)

    return df

def save_data(df, database_filename):
    '''
    This function saves the dataframe into a SQLite database file
    
    Input:
        df: dataframe
        database_filename: SQLite db filepath 
    
    '''
    engine = create_engine('sqlite:///' + database_filename)
    table_name = database_filename.replace(".db","")+"_table"
    df.to_sql(table_name, engine, index=False,if_exists='replace')  


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