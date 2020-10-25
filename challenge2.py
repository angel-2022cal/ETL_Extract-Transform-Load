# Import dependencies.
import json
import pandas as pd
import numpy as np
import re
from sqlalchemy import create_engine
import psycopg2
import time
from config import db_password
# Define a variable file_dir for the directory that’s holding our data.
file_dir = 'C:/Users/User/Desktop/Class/Goddard_Shannon_Movies-ETL'
# Open the Wikipedia JSON file to be read into the variable file, and use json.load() to save the data to a new variable.
with open(f'{file_dir}/data/wikipedia-movies.json', mode='r') as file:
    wiki_movies_raw = json.load(file)
# Pull Kaggle data into Pandas DataFrames directly.
kaggle_metadata = pd.read_csv(f'{file_dir}/data/movies_metadata.csv', low_memory=False)
ratings = pd.read_csv(f'{file_dir}/data/ratings.csv')
# Create a list comprehension with the filter expression we created
# Save that to an intermediate variable wiki_movies.  

wiki_movies = [movie for movie in wiki_movies_raw
               if ('Director' in movie or 'Directed by' in movie)
                   and 'imdb_link' in movie
                   and 'No. of episodes' not in movie] # Filter TV shows out of movies.
# Create wiki_movies DataFrame.
wiki_movies_df = pd.DataFrame(wiki_movies_raw)
wiki_movies_df.head()
def clean_movie(movie):
    movie = dict(movie) #create a non-destructive copy
    alt_titles = {}
    # combine alternate titles into one list
    for key in ['Also known as','Arabic','Cantonese','Chinese','French',
                'Hangul','Hebrew','Hepburn','Japanese','Literally',
                'Mandarin','McCune-Reischauer','Original title','Polish',
                'Revised Romanization','Romanized','Russian',
                'Simplified','Traditional','Yiddish']:
        if key in movie:
            alt_titles[key] = movie[key]
            movie.pop(key)
    if len(alt_titles) > 0:
        movie['alt_titles'] = alt_titles

    # merge column names
    def change_column_name(old_name, new_name):
        if old_name in movie:
            movie[new_name] = movie.pop(old_name)
    change_column_name('Adaptation by', 'Writer(s)')
    change_column_name('Country of origin', 'Country')
    change_column_name('Directed by', 'Director')
    change_column_name('Distributed by', 'Distributor')
    change_column_name('Edited by', 'Editor(s)')
    change_column_name('Length', 'Running time')
    change_column_name('Original release', 'Release date')
    change_column_name('Music by', 'Composer(s)')
    change_column_name('Produced by', 'Producer(s)')
    change_column_name('Producer', 'Producer(s)')
    change_column_name('Productioncompanies ', 'Production company(s)')
    change_column_name('Productioncompany ', 'Production company(s)')
    change_column_name('Released', 'Release Date')
    change_column_name('Release Date', 'Release date')
    change_column_name('Screen story by', 'Writer(s)')
    change_column_name('Screenplay by', 'Writer(s)')
    change_column_name('Story by', 'Writer(s)')
    change_column_name('Theme music composer', 'Composer(s)')
    change_column_name('Written by', 'Writer(s)')

    return movie
# Rerun our list comprehension to clean wiki_movies and recreate wiki_movies_df.
clean_movies = [clean_movie(movie) for movie in wiki_movies]
wiki_movies_df = pd.DataFrame(clean_movies)
# Tell Python to treat our regular expression characters as a raw string of text by putting an r before the quotes.
# Drop any duplicates of IMDb IDs by using the drop_duplicates() method.
# To specify that we only want to consider the IMDb ID, use the subset argument, and set inplace equal to "True."
# We also want to see the new number of rows and how many rows were dropped. 
wiki_movies_df['imdb_id'] = wiki_movies_df['imdb_link'].str.extract(r'(tt\d{7})')
print(len(wiki_movies_df))
wiki_movies_df.drop_duplicates(subset='imdb_id', inplace=True)
print(len(wiki_movies_df))
wiki_movies_df.head()
# Remove Mostly Null Columns
wiki_columns_to_keep = [column for column in wiki_movies_df.columns if wiki_movies_df[column].isnull().sum() < len(wiki_movies_df) * 0.9]
wiki_movies_df = wiki_movies_df[wiki_columns_to_keep]
wiki_movies_df.head()
# Make a Plan to Convert and Parse the Data
box_office = wiki_movies_df['Box office'].dropna()
# Use the apply() method to see which values are not strings
def is_not_a_string(x):
    return type(x) != str
box_office[box_office.map(lambda x: type(x) != str)]
# Make a separator string and then call the join() method on it.
box_office = box_office.apply(lambda x: ' '.join(x) if type(x) == list else x)
# Create a variable form_one and set it equal to the finished regular expression string.
# Preface the string with an r for the escape characters to remain.
form_one = r'\$\d+\.?\d*\s*[mb]illion'
# Create a variable form_two and set it equal to the finished regular expression string.
# Preface the string with an r for the escape characters to remain.
form_two = r'\$\d{1,3}(?:,\d{3})+'
# Create two Boolean Series called matches_form_one and matches_form_two, 
# and then select the box office values that don’t match either
matches_form_one = box_office.str.contains(form_one, flags=re.IGNORECASE)
matches_form_two = box_office.str.contains(form_two, flags=re.IGNORECASE)
box_office[~matches_form_one & ~matches_form_two]
# “Million” is sometimes misspelled as “millon.” Make the second “i” optional in our match string with a question mark.
form_one = r'\$\s*\d+\.?\d*\s*[mb]illi?on'
# Change form_two to allow for either a comma or period as a thousands separator.
# Add a negative lookahead group that looks ahead for “million” or “billion” after the number and rejects the match.
form_two = r'\$\s*\d{1,3}(?:[,\.]\d{3})+(?!\s[mb]illion)'
# Some values are given as a range.To solve this problem, we’ll search for any string 
# that starts with a dollar sign and ends with a hyphen, and then replace it with just a 
# dollar sign using the replace() method.
box_office = box_office.str.replace(r'\$.*[-—–](?![a-z])', '$', regex=True)
# Use the expressions to extract only the parts of the strings that match. Do this with the str.extract() method.
# Make a regular expression that captures data when it matches either form_one or form_two with an f-string.
def parse_dollars(s):
    # if s is not a string, return NaN
    if type(s) != str:
        return np.nan

    # if input is of the form $###.# million
    if re.match(r'\$\s*\d+\.?\d*\s*milli?on', s, flags=re.IGNORECASE):

        # remove dollar sign and " million"
        s = re.sub('\$|\s|[a-zA-Z]','', s)

        # convert to float and multiply by a million
        value = float(s) * 10**6

        # return value
        return value

    # if input is of the form $###.# billion
    elif re.match(r'\$\s*\d+\.?\d*\s*billi?on', s, flags=re.IGNORECASE):

        # remove dollar sign and " billion"
        s = re.sub('\$|\s|[a-zA-Z]','', s)

        # convert to float and multiply by a billion
        value = float(s) * 10**9

        # return value
        return value

    # if input is of the form $###,###,###
    elif re.match(r'\$\s*\d{1,3}(?:[,\.]\d{3})+(?!\s[mb]illion)', s, flags=re.IGNORECASE):

        # remove dollar sign and commas
        s = re.sub('\$|,','', s)

        # convert to float
        value = float(s)

        # return value
        return value

    # otherwise, return NaN
    else:
        return np.nan
# Extract the values from box_office using str.extract. 
# Then apply parse_dollars to the first column in the DataFrame returned by str.extract.
wiki_movies_df['box_office'] = box_office.str.extract(f'({form_one}|{form_two})', flags=re.IGNORECASE)[0].apply(parse_dollars)
# Drop the Box Office column.
wiki_movies_df.drop('Box office', axis=1, inplace=True)
# Create a budget variable.
budget = wiki_movies_df['Budget'].dropna()
# Convert any lists to strings:
budget = budget.map(lambda x: ' '.join(x) if type(x) == list else x)
# Then remove any values between a dollar sign and a hyphen (for budgets given in ranges):
budget = budget.str.replace(r'\$.*[-—–](?![a-z])', '$', regex=True)
# Create two Boolean Series called matches_form_one and matches_form_two, 
# and then select the budget values that don’t match either
matches_form_one = budget.str.contains(form_one, flags=re.IGNORECASE)
matches_form_two = budget.str.contains(form_two, flags=re.IGNORECASE)
budget[~matches_form_one & ~matches_form_two]
# Remove the citation references
budget = budget.str.replace(r'\[\d+\]\s*', '')
budget[~matches_form_one & ~matches_form_two]
# Parse the budget values
wiki_movies_df['budget'] = budget.str.extract(f'({form_one}|{form_two})', flags=re.IGNORECASE)[0].apply(parse_dollars)
# Drop the original Budget column.
wiki_movies_df.drop('Budget', axis=1, inplace=True)
# Make the release_date variable.
release_date = wiki_movies_df['Release date'].dropna().apply(lambda x: ' '.join(x) if type(x) == list else x)
# Parse the forms.
date_form_one = r'(?:January|February|March|April|May|June|July|August|September|October|November|December)\s[123]\d,\s\d{4}'
date_form_two = r'\d{4}.[01]\d.[123]\d'
date_form_three = r'(?:January|February|March|April|May|June|July|August|September|October|November|December)\s\d{4}'
date_form_four = r'\d{4}'
# Use the built-in to_datetime() method in Pandas, set the infer_datetime_format option to True.
wiki_movies_df['release_date'] = pd.to_datetime(release_date.str.extract(f'({date_form_one}|{date_form_two}|{date_form_three}|{date_form_four})')[0], infer_datetime_format=True)
# Parse Running Time
running_time = wiki_movies_df['Running time'].dropna().apply(lambda x: ' '.join(x) if type(x) == list else x)
# Only extract digits and allow for both possible patterns by adding capture groups around the \d instances as well as add an alternating character.
running_time_extract = running_time.str.extract(r'(\d+)\s*ho?u?r?s?\s*(\d*)|(\d+)\s*m')
# This new DataFrame is all strings. Convert them to numeric values. Use the to_numeric() method and set the errors argument to 'coerce'. 
# Coercing the errors will turn the empty strings into Not a Number (NaN), then we can use fillna() to change all the NaNs to zeros.
running_time_extract = running_time_extract.apply(lambda col: pd.to_numeric(col, errors='coerce')).fillna(0)
# Apply a function that will convert the hour capture groups and minute capture groups to minutes if the pure minutes capture group is zero, and save the output to wiki_movies_df:
wiki_movies_df['running_time'] = running_time_extract.apply(lambda row: row[0]*60 + row[1] if row[2] == 0 else row[2], axis=1)
# Drop Running time from the dataset with the following code:
wiki_movies_df.drop('Running time', axis=1, inplace=True)
# Remove the bad data.
kaggle_metadata[~kaggle_metadata['adult'].isin(['True','False'])]
# Keep rows where the adult column is False, and then drop the adult column.
kaggle_metadata = kaggle_metadata[kaggle_metadata['adult'] == 'False'].drop('adult',axis='columns')
# Creates the Boolean column we want and assign it back to video.
kaggle_metadata['video'] = kaggle_metadata['video'] == 'True'
# Use the to_numeric() method from Pandas. Make sure the errors= argument is set to 'raise'. 
kaggle_metadata['budget'] = kaggle_metadata['budget'].astype(int)
kaggle_metadata['id'] = pd.to_numeric(kaggle_metadata['id'], errors='raise')
kaggle_metadata['popularity'] = pd.to_numeric(kaggle_metadata['popularity'], errors='raise')
# Convert release_date to to_datetime().
kaggle_metadata['release_date'] = pd.to_datetime(kaggle_metadata['release_date'])
# Assign it to the timestamp column.
ratings['timestamp'] = pd.to_datetime(ratings['timestamp'], unit='s')
# Merge by IMDb ID.
movies_df = pd.merge(wiki_movies_df, kaggle_metadata, on='imdb_id', suffixes=['_wiki','_kaggle'])
# Competing data:

# Wiki                     Movielens                Resolution
#--------------------------------------------------------------------------
# title_wiki               title_kaggle
# running_time             runtime
# budget_wiki              budget_kaggle
# box_office               revenue
# release_date_wiki        release_date_kaggle
# Language                 original_language
# Production company(s)    production_companies  
# Get the index.
movies_df[(movies_df['release_date_wiki'] > '1996-01-01') & (movies_df['release_date_kaggle'] < '1965-01-01')].index
# Drop the row.
movies_df = movies_df.drop(movies_df[(movies_df['release_date_wiki'] > '1996-01-01') & (movies_df['release_date_kaggle'] < '1965-01-01')].index)
# Drop the title_wiki, release_date_wiki, Language, and Production company(s) columns.
movies_df.drop(columns=['title_wiki','release_date_wiki','Language','Production company(s)'], inplace=True)
# Make a function that fills in missing data for a column pair and then drops the redundant column.
def fill_missing_kaggle_data(df, kaggle_column, wiki_column):
    df[kaggle_column] = df.apply(
        lambda row: row[wiki_column] if row[kaggle_column] == 0 else row[kaggle_column]
        , axis=1)
    df.drop(columns=wiki_column, inplace=True)
# Run the function for the three column pairs that we decided to fill in zeros.
fill_missing_kaggle_data(movies_df, 'runtime', 'running_time')
fill_missing_kaggle_data(movies_df, 'budget_kaggle', 'budget_wiki')
fill_missing_kaggle_data(movies_df, 'revenue', 'box_office')
movies_df
# Reorder the columns:
movies_df = movies_df[['imdb_id','id','title_kaggle','original_title','tagline','belongs_to_collection','url','imdb_link',
                       'runtime','budget_kaggle','revenue','release_date_kaggle','popularity','vote_average','vote_count',
                       'genres','original_language','overview','spoken_languages','Country',
                       'production_companies','production_countries','Distributor',
                       'Producer(s)','Director','Starring','Cinematography','Editor(s)','Writer(s)','Composer(s)','Based on'
                      ]]
# Rename the columns to be consistent.
    
movies_df.rename({'id':'kaggle_id',
                    'title_kaggle':'title',
                    'url':'wikipedia_url',
                    'budget_kaggle':'budget',
                    'release_date_kaggle':'release_date',
                    'Country':'country',
                    'Distributor':'distributor',
                    'Producer(s)':'producers',
                    'Director':'director',
                    'Starring':'starring',
                    'Cinematography':'cinematography',
                    'Editor(s)':'editors',
                    'Writer(s)':'writers',
                    'Composer(s)':'composers',
                    'Based on':'based_on'
                    }, axis='columns', inplace=True)
# Use a groupby on the “movieId” and “rating” columns and take the count for each group.
# Rename the “userId” column to “count.”
# Pivot this data so that movieId is the index, the columns will be all the rating values, and the rows will be the counts for each rating value.
rating_counts = ratings.groupby(['movieId','rating'], as_index=False).count() \
                .rename({'userId':'count'}, axis=1) \
                .pivot(index='movieId',columns='rating', values='count')
# Rename the columns so they’re easier to understand. Prepend rating_ to each column with a list comprehension:
rating_counts.columns = ['rating_' + str(col) for col in rating_counts.columns]
# Use a left merge, since we want to keep everything in movies_df:
movies_with_ratings_df = pd.merge(movies_df, rating_counts, left_on='kaggle_id', right_index=True, how='left')
# Fill 0 for all non values
movies_with_ratings_df[rating_counts.columns] = movies_with_ratings_df[rating_counts.columns].fillna(0)
# Make a connection string.
"postgres://[user]:[password]@[location]:[port]/[database]"
# Make a connection string for the local server.
db_string = f"postgres://postgres:{db_password}@127.0.0.1:5432/movies_challenge"
# Create the database engine.
engine = create_engine(db_string)
# Save the movies_df DataFrame to a SQL table using the to_sql() method.
# movies_df.to_sql(name='challenge', con=engine)
# Reimport the CSV using the chunksize= parameter in read_csv(). 
# Make a for loop and append the chunks of data to the new rows to the target SQL table.
#rows_imported = 0
# get the start_time from time.time()
#start_time = time.time()
#for data in pd.read_csv(f'{file_dir}/data/ratings.csv', chunksize=1000000):
#    print(f'importing rows {rows_imported} to {rows_imported + len(data)}...', end='')
#    data.to_sql(name='ratings', con=engine, if_exists='append')
#    rows_imported += len(data)

    # add elapsed time to final print out
#    print(f'Done. {time.time() - start_time} total seconds elapsed')       
