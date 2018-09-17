"""tmdb.py: A program to analyze the TMDb movie database."""

#pylint: disable-msg=too-many-arguments
#pylint: disable-msg=too-many-locals
#pylint: disable-msg=too-many-statements
#pylint: disable-msg=too-many-branches

import sqlite3
import csv
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn import linear_model

def escapestring(stng):
    """Returns an escaped string suitable for SQL commands."""
    return stng.replace("'", "''")

# dict_to_insert and dict_to_update take a Python dictionary and turn it into
# an SQL INSERT or UPDATE.
# Types on the values in the dictionary are assumed to be correct!
# I.e. numbers should be floats and ints, and strings should be strings
# (not numbers represented as strings or vice-versa or anything like that)
# This is useful because we can construct a Python dictionary once, then
# use it to either insert or update.
def dict_to_insert(tblname, insdct):
    """Converts a dictionary of fields to be inserted into an SQL INSERT."""
    fields = ''
    values = ''
    for fieldname in insdct:
        fields = fields + ', ' + fieldname
        if isinstance(insdct[fieldname], int):
            values = values + ', ' + str(insdct[fieldname])
        elif isinstance(insdct[fieldname], float):
            values = values + ', ' + str(insdct[fieldname])
        elif isinstance(insdct[fieldname], str):
            values = values + ", '" + escapestring(insdct[fieldname]) + "'"
        else:
            # this else should never happen
            tcba = type(insdct[fieldname])
            print('error: unrecognized type for:', tcba)
    sql = 'INSERT INTO ' + tblname + '(' + fields[2:] + ') VALUES (' + values[2:] + ');'
    return sql

def dict_to_update(tblname, updatedct, whereclause):
    """Converts a dictionary of fields to be updated into an SQL UPDATE."""
    setstmt = ''
    for fieldname in updatedct:
        setstmt = setstmt + ', ' + fieldname + ' = '
        if isinstance(updatedct[fieldname], int):
            setstmt = setstmt + str(updatedct[fieldname])
        elif isinstance(updatedct[fieldname], float):
            setstmt = setstmt + str(updatedct[fieldname])
        elif isinstance(updatedct[fieldname], str):
            setstmt = setstmt + "'" + escapestring(updatedct[fieldname]) + "'"
        else:
            # this else should never happen
            tcba = type(updatedct[fieldname])
            print('error: unrecognized type for:', tcba)
    sql = 'UPDATE ' + tblname + ' SET ' + setstmt[2:] + ' WHERE ' + whereclause + ';'
    return sql

# csv_to_database pulls a CSV file into a database.
# This function does NOT create the fields!
# It can't do that because it doesn't know the types for each column!
# YOU must create the table with the correct field names and types before calling this function.
# Column names much match field names in the CSV.
def csv_to_database(csv_file, tblname, rename_fields, dbcu):
    """Pulls a CSV file into a database table. The table needs to already be created."""
    fieldnames = []
    rownum = 0
    with open(csv_file, newline='') as csvfile:
        thereader = csv.reader(csvfile, delimiter=',', quotechar='"')
        for rowdat in thereader:
            if rownum == 0:
                for info in rowdat:
                    if info in rename_fields:
                        fieldnames.append(rename_fields[info])
                    else:
                        fieldnames.append(info)
            else:
                insdict = {}
                colnum = 0
                for info in rowdat:
                    insdict[fieldnames[colnum]] = info
                    colnum = colnum + 1
                sql = dict_to_insert(tblname, insdict)
                dbcu.execute(sql)
            rownum = rownum + 1

# Logarithm of the number "plus one" to keep 0s from blowing up the system.
# We are making the assumption that the "plus one" has little effect because we are dealing with
# movie budgets and revenues in millions of dollars.
def mylogp1(num):
    """Returns log(num + 1)"""
    return np.log(num + 1.0)

# In the Jupyter notebook, we convert numbers like "1.902723e+09" to numbers "in millions with just
# df_movies['revenue_adj_mil'] = df_movies['revenue_adj'] / 1000000
# We need something comparable here for our SQL queries.
def mytomillions(num):
    """Convert dollars to dollars in millions."""
    return num / 1000000.0

# map_column creates a new column with a value from an existing column mapped through a function
# This is one thing that's easier to do in Pandas than SQL.
# In Pandas this functionality is built-in but here we have to do it ourselves.
def map_colum(dbcu, tblname, primarykey, whereclause, col1, col2, mapfunc):
    """Maps col1 to col 2 by calling mapfunc."""
    # BUGBUG: This function only works for INTEGER primary keys
    sql = 'SELECT ' + primarykey + ', ' + col1 + ' FROM ' + tblname + ' WHERE ' + whereclause + ';'
    stufftodo = []
    for row in dbcu.execute(sql):
        stufftodo.append(row)
    for item in stufftodo:
        newval = mapfunc(item[1])
        updaterec = {col2 : newval}
        subwhere = primarykey + ' = ' + str(item[0])
        sql = dict_to_update(tblname, updaterec, subwhere)
        dbcu.execute(sql)

# The Laplace number assumes a prior "vote" of 5.0 (exactly average)
# and that will be the number assigned if the number of votes is zero.
# As the number of votes increases, the Laplace number will trend towards
# the actual average.
# In this manner the Laplace number combines both the vote count and vote average
# fields into a single number which reasonably represents how "good" the movie is
# according to the votes.
# The actual formula is:
# laplace_nums =
#     (5.0 + (df_with_revenue['vote_count'] * df_with_revenue['vote_average']))
#     /
#     (1.0 + df_with_revenue['vote_count'])
# The mathematical justification for this is explained here:
# https://en.wikipedia.org/wiki/Rule_of_succession
def calc_laplace_nums(dbcu):
    """Takes vote count and vote average and sets laplace_num."""
    sql = 'SELECT id, vote_count, vote_average FROM movies WHERE 1;'
    stufftodo = []
    for row in dbcu.execute(sql):
        stufftodo.append(row)
    for item in stufftodo:
        newval = (5.0 + (item[1] * item[2])) / (1.0 + item[1])
        updaterec = {'laplace_num': newval}
        subwhere = 'id = ' + str(item[0])
        sql = dict_to_update('movies', updaterec, subwhere)
        dbcu.execute(sql)

# sql_to_dataframe is the primary function -- it transforms any SQL query into a Pandas DataFrame!
# Magic! But it only works if you set row_factory = sqlite3.Row on your connection BEFORE
# you create your cursor!
# Otherwise row.keys() in here will fail.
def sql_to_dataframe(dbcu, sql):
    """Takes an SQL query and returns a Pandas dataframe."""
    count = 0
    datlist = []
    for row in dbcu.execute(sql):
        if count == 0:
            columns = []
            for colname in row.keys():
                columns.append(colname)
                datlist.append([])
        position = 0
        for item in row:
            datlist[position].append(item)
            position = position + 1
        count = count + 1
    dfdct = {}
    position = 0
    for colname in columns:
        dfdct[colname] = datlist[position]
        position = position + 1
    return pd.DataFrame(dfdct)

# get_field_entries_above_threshold is a helper function for the movie database so we can get
# genres, directors, cast members, etc, only if they appear above some threshold, because if they
# don't appear enough we can't assume any statistical significance. Cutting out everything below
# the threshold makes the math easier.
def get_field_entries_above_threshold(dbcu, fieldname, threshold):
    """Looks at pipe-separated field and finds only the entries above some arbitrary threshold."""
    item_counts = {}
    sql = 'SELECT ' + fieldname + ' FROM movies WHERE 1;'
    for row in dbcu.execute(sql):
        text_labels = row[0]
        txt_lbls_lst = text_labels.split('|')
        for the_item in txt_lbls_lst:
            if the_item in item_counts:
                item_counts[the_item] = item_counts[the_item] + 1
            else:
                item_counts[the_item] = 1
    results = []
    for the_item in item_counts:
        if item_counts[the_item] > threshold:
            trimmed = the_item.strip()
            if trimmed != '':
                results.append(trimmed)
    return results

# dbnameize is a helper function for the movie DB because the contents of some fields are
# unsuitable for use as DB column names.
# here we fix that
def dbnameize(stng):
    """Returns a name suitable for use as a database field name."""
    # make suitable for db field name by making spaces underscores
    # replace dashes with underscores, too, ("father-son") because those mess up the db
    # and apostrophes ("love of one's life")
    # and periods ("U.S. President")
    # and parenthesis ("(MGM)")
    # and plus sign ("Canal+")
    # we also make lower case for no reason
    return (((((((stng.replace(' ', '_')).replace('-', '_')).replace("'", '')).replace(
        '.', '')).replace('(', '')).replace(')', '')).replace('+', '')).lower()

# create_binomial_fields, zero_out_binomial_fields, and fill_in_binomial_fields are three helper
# functions that we use to create and set binomial (0 or 1) fields for genres, directors,
# cast members, keywords, etc, for the movie database, that we can feed into linear regression.
def create_binomial_fields(dbcu, tblname, prefix, item_list):
    """Breaks out pipe-separated fields into multiple fields."""
    for the_item in item_list:
        fieldname = prefix + dbnameize(the_item)
        sql = 'ALTER TABLE ' + tblname + ' ADD COLUMN ' + fieldname + ' INTEGER;'
        dbcu.execute(sql)

def zero_out_binomial_fields(dbcu, tblname, prefix, item_list):
    """Initializes binomial fields to 0."""
    for the_item in item_list:
        fieldname = prefix + dbnameize(the_item)
        dbcu.execute('UPDATE ' + tblname + ' SET ' + fieldname + ' = 0;')

def fill_in_binomial_fields(dbcu, tblname, primarykey, fieldname, prefix, item_list):
    """Fills in binomial fields according to the pipe-separated values in fieldname."""
    sql = 'SELECT ' + primarykey + ', ' + fieldname + ' FROM ' + tblname + ' WHERE 1;'
    stufftodo = []
    for row in dbcu.execute(sql):
        stufftodo.append(row)
    for item in stufftodo:
        key_value = item[0]
        text_labels = item[1]
        txt_lbls_lst = text_labels.split('|')
        for the_item in txt_lbls_lst:
            if the_item in item_list:
                sql = 'UPDATE ' + tblname + '                                 \
                       SET ' + prefix + dbnameize(the_item) + ' = 1           \
                       WHERE ' + primarykey + ' = ' + str(key_value) + ';'
                dbcu.execute(sql)

def get_field_names(dbcu, tblname):
    """Return list of fields from a DB table"""
    # Don't know any other way to get the field names except to do a SELECT *
    # Won't work if table is empty!
    sql = 'SELECT * FROM ' + tblname + ' WHERE 1 LIMIT 1;'
    columns = []
    for row in dbcu.execute(sql):
        for colname in row.keys():
            columns.append(colname)
    return columns

def calc_profit(dbcu):
    """Adds two profit fields, profit_adj_log and roi_adj_log"""
    sql = 'SELECT id, budget_adj, revenue_adj, budget_adj_log, revenue_adj_log FROM movies WHERE 1;'
    stufftodo = []
    for row in dbcu.execute(sql):
        stufftodo.append(row)
    for item in stufftodo:
        if item[2] >= item[1]:
            newval1 = np.log(item[2] - item[1] + 1.0)
        else:
            # hack to represent losses as negative numbers
            newval1 = -np.log(item[1] - item[2] + 1.0)
        newval2 = item[4] - item[3]
        updaterec = {'profit_adj_log': newval1, 'roi_adj_log': newval2}
        subwhere = 'id = ' + str(item[0])
        sql = dict_to_update('movies', updaterec, subwhere)
        dbcu.execute(sql)

# This function is like sql_to_dataframe except it just gives us a single number back
# useful if you're just selecting a count of something or the ID number for something.
def sql_to_scalar(dbcu, sql):
    """Return a single value from an SQL query"""
    for row in dbcu.execute(sql):
        for item in row:
            result = item
    return result

def set_up_movie_db(csv_file):
    """Pull in the TMDB movie database and add a bunch of fields we like"""
    # movieconn = sqlite3.connect('movies.db')
    movieconn = sqlite3.connect(':memory:')
    movieconn.row_factory = sqlite3.Row
    moviecu = movieconn.cursor()
    # We don't have a system in place to figure out data types from the CSV file, so we have to
    # explicitly tell SQL what our columns are and what type they are
    moviecu.execute('CREATE TABLE movies (    \
                id INTEGER PRIMARY KEY,       \
                imdb_id TEXT,                 \
                popularity REAL,              \
                budget REAL,                  \
                revenue REAL,                 \
                original_title TEXT,          \
                performers TEXT,              \
                homepage TEXT,                \
                director TEXT,                \
                tagline TEXT,                 \
                keywords TEXT,                \
                overview TEXT,                \
                runtime REAL,                 \
                genres TEXT,                  \
                production_companies TEXT,    \
                release_date TEXT,            \
                vote_count INTEGER,           \
                vote_average REAL,            \
                release_year INTEGER,         \
                budget_adj REAL,              \
                revenue_adj REAL              \
            );')
    csv_to_database(csv_file, 'movies', {'cast':'performers'}, moviecu)

    moviecu.execute('ALTER TABLE movies ADD COLUMN revenue_adj_log REAL;')
    map_colum(moviecu, 'movies', 'id', '1', 'revenue_adj', 'revenue_adj_log', mylogp1)

    moviecu.execute('ALTER TABLE movies ADD COLUMN laplace_num REAL;')
    calc_laplace_nums(moviecu)

    movieconn.commit()

    return movieconn, moviecu

def main():
    """Starting point for analyzing the TMDb database!"""
    # trailing slash is REQUIRED here
    path = '/Users/waynerad/Documents/project/udacity/nanodegree/datascience/p2test/'
    csv_file = path + 'tmdb-movies-modified.csv'

    print('reading: ', csv_file)
    movieconn, moviecu = set_up_movie_db(csv_file)

    num_movies = sql_to_scalar(moviecu, 'SELECT COUNT(*) FROM movies WHERE 1;')
    print('Number of movies:', num_movies)
    no_budget_count = sql_to_scalar(moviecu, 'SELECT COUNT(*) FROM movies WHERE budget_adj = 0;')
    print('Movies with zero budget:', no_budget_count)
    no_revenue_count = sql_to_scalar(moviecu, 'SELECT COUNT(*) FROM movies WHERE revenue_adj = 0;')
    print('Movies with zero revenue:', no_revenue_count)
    has_revenue_count = sql_to_scalar(moviecu, 'SELECT COUNT(*) FROM movies WHERE revenue_adj > 0;')
    print('Movies with non-zero revenue:', has_revenue_count)
    no_both_bdj_rev = sql_to_scalar(moviecu, 'SELECT COUNT(*)                                    \
                                              FROM movies                                        \
                                              WHERE (budget_adj = 0) AND (revenue_adj = 0);')
    print('Movies where both bugdget and revenue are zero:', no_both_bdj_rev)
    no_popularity = sql_to_scalar(moviecu, 'SELECT COUNT(*) FROM movies WHERE popularity = 0;')
    print('Movies where popularity is zero:', no_popularity)
    no_votes = sql_to_scalar(moviecu, 'SELECT COUNT(*) FROM movies WHERE vote_count = 0;')
    print('Movies where votes are zero:', no_votes)
    no_vote_avg = sql_to_scalar(moviecu, 'SELECT COUNT(*) FROM movies WHERE vote_average = 0;')
    print('Movies where vote average is zero:', no_vote_avg)
    no_release_year = sql_to_scalar(moviecu, 'SELECT COUNT(*) FROM movies WHERE release_year = 0;')
    print('Movies where release year is zero:', no_release_year)
    no_runtime = sql_to_scalar(moviecu, 'SELECT COUNT(*) FROM movies WHERE runtime = 0;')
    print('Movies where runtime is zero:', no_runtime)

    moviecu.execute('ALTER TABLE movies ADD COLUMN budget_adj_log REAL;')
    map_colum(moviecu, 'movies', 'id', '1', 'budget_adj', 'budget_adj_log', mylogp1)

    movieconn.commit()

    print('----------------------------------------------------------------------')
    print('------------------- budget vs revenue --------------------------------')
    print('----------------------------------------------------------------------')
    # for these, the WHERE and ORDER BY must match exactly!!!
    sql_x = 'SELECT budget_adj_log FROM movies WHERE (budget_adj_log > 0)     \
                    AND (revenue_adj_log > 0) ORDER BY id;'
    sql_y = 'SELECT revenue_adj_log FROM movies WHERE (budget_adj_log > 0)    \
                    AND (revenue_adj_log > 0) ORDER BY id;'

    df_x = sql_to_dataframe(moviecu, sql_x)
    df_y = sql_to_dataframe(moviecu, sql_y)

    df_x = sm.add_constant(df_x)
    model = sm.OLS(df_y, df_x).fit()
    print(model.summary())
    # fit_intercept has to be True in the Jupyter notebook
    lin_mod = linear_model.LinearRegression(fit_intercept=False)
    model = lin_mod.fit(df_x, df_y)
    print(model.coef_) # has to be print(model.intercetp_, model.coef_) for the Jupyter notebook

    print('----------------------------------------------------------------------')
    print('------------------ runtime vs revenue --------------------------------')
    print('----------------------------------------------------------------------')
    # for these, the WHERE and ORDER BY clauses must match exactly!!!
    sql_x = 'SELECT runtime                                   \
             FROM movies                                      \
             WHERE (runtime > 0) AND (revenue_adj_log > 0)    \
             ORDER BY id;'
    sql_y = 'SELECT revenue_adj_log                           \
             FROM movies                                      \
             WHERE (runtime > 0) AND (revenue_adj_log > 0)    \
             ORDER BY id;'

    df_x = sql_to_dataframe(moviecu, sql_x)
    df_y = sql_to_dataframe(moviecu, sql_y)

    df_x = sm.add_constant(df_x)
    model = sm.OLS(df_y, df_x).fit()
    print(model.summary())
    # fit_intercept has to be True in the Jupyter notebook
    lin_mod = linear_model.LinearRegression(fit_intercept=False)
    model = lin_mod.fit(df_x, df_y)
    print(model.coef_) # has to be print(model.intercetp_, model.coef_) for the Jupyter notebook

    # at this point, we dump our database because
    # we're going to start over for each of the following
    # this could be made more efficient by saving the database to disk at this point

    movieconn.close()

    fields_to_process = ['genres', 'keywords', 'director', 'performers', 'production_companies']
    # stuff to highlight was determined from previous runs of the program
    top_highlights = {
        'genres': ['genres_adventure', 'genres_documentary'],
        'keywords': ['keywords_animation', 'keywords_independent_film'],
        'director': ['director_steven_spielberg'],
        'performers': ['performers_tom_cruise'],
        'production_companies': [
            'production_companies_dreamworks_animation',
            'production_companies_wild_bunch'
        ]
    }
    for field_selected in fields_to_process:
        # re-establish database
        # this is inefficient and we can only get away with it because our dataset is small
        # smarter way is to re-load db from disk

        movieconn, moviecu = set_up_movie_db(csv_file)

        prefix = field_selected + '_'
        above_threshold_list = get_field_entries_above_threshold(moviecu, field_selected, 25)
        create_binomial_fields(moviecu, 'movies', prefix, above_threshold_list)
        movieconn.commit()
        zero_out_binomial_fields(moviecu, 'movies', prefix, above_threshold_list)
        movieconn.commit()
        fill_in_binomial_fields(moviecu, 'movies', 'id', field_selected, prefix,
                                above_threshold_list)
        movieconn.commit()

        fld_list = get_field_names(moviecu, 'movies')
        fld_lst_txt = ''
        len_x = len(prefix)
        for item in fld_list:
            if item[:len_x] == prefix:
                fld_lst_txt = fld_lst_txt + ', ' + item
        fld_lst_txt = fld_lst_txt[2:]

        targt_lst = ['revenue', 'popularity', 'votes']
        for target in targt_lst:
            print('----------------------------------------------------------------------')
            print('------------------ ' + field_selected + ' vs ' + target
                  + ' ----------------------------------')
            print('----------------------------------------------------------------------')

            # for these, the WHERE and ORDER BY must match exactly!!!
            if target == 'revenue':
                sql_x = 'SELECT ' + fld_lst_txt + '     \
                         FROM movies                    \
                         WHERE (revenue_adj_log > 0)    \
                         ORDER BY id;'
                sql_y = 'SELECT revenue_adj_log         \
                         FROM movies                    \
                         WHERE (revenue_adj_log > 0)    \
                         ORDER BY id;'
            elif target == 'popularity':
                # popularity is > 0 for all movies, so this filter is actually redundant
                sql_x = 'SELECT ' + fld_lst_txt + ' FROM movies WHERE (popularity > 0) ORDER BY id;'
                sql_y = 'SELECT popularity FROM movies WHERE (popularity > 0) ORDER BY id;'
            elif target == 'votes':
                # laplace_num is > 0 for all movies, so this filter is actually redundant
                sql_x = 'SELECT ' + fld_lst_txt + '    \
                         FROM movies                   \
                         WHERE (laplace_num > 0)       \
                         ORDER BY id;'
                sql_y = 'SELECT laplace_num FROM movies WHERE (laplace_num > 0) ORDER BY id;'
            else:
                print('Oh no, unknown target!')

            df_x = sql_to_dataframe(moviecu, sql_x)
            df_y = sql_to_dataframe(moviecu, sql_y)

            df_x = sm.add_constant(df_x)

            model = sm.OLS(df_y, df_x).fit()
            print(model.summary())
            lin_mod = linear_model.LinearRegression(fit_intercept=False)
            model = lin_mod.fit(df_x, df_y)
            print(model.coef_)

            # highlight things
            # we only do this with revenue for now, but in principle it can be done
            # with popularity and votes, too.

            if target == 'revenue':
                highlight_list = top_highlights[field_selected]
                for thing in highlight_list:
                    sql = 'SELECT id, original_title, revenue, revenue_adj_log    \
                           FROM movies                                            \
                           WHERE (revenue_adj_log > 0) AND (' + thing + ' = 1)    \
                           ORDER BY revenue_adj_log DESC                          \
                           LIMIT 20;'
                    df_top_in_category = sql_to_dataframe(moviecu, sql)
                    print('outputting to: ', path + target + '_' + thing + '.csv')
                    df_top_in_category.to_csv(path + target + '_' + thing + '.csv')
        movieconn.close()

    # ok, we're out of that loop!

    # now we just have to do the "by release_year" stuff!

    movieconn, moviecu = set_up_movie_db(csv_file)

    moviecu.execute('ALTER TABLE movies ADD COLUMN revenue_adj_mil REAL;')
    map_colum(moviecu, 'movies', 'id', '1', 'revenue_adj', 'revenue_adj_mil', mytomillions)

    sql = 'SELECT release_year, MAX(revenue_adj_mil) FROM movies GROUP BY release_year'
    df_max_rev_by_year = sql_to_dataframe(moviecu, sql)
    print(df_max_rev_by_year)
    print('outputting to: ' + path + 'max_rev_by_year.csv')
    df_max_rev_by_year.to_csv(path + 'max_rev_by_year.csv')

    # at this point, we resort to an inefficient loop.
    # it's POSSIBLE to do all this in one SQL statement, but
    # I'm too lazy at the moment to figure it out

    list_of_max_movies = []

    for _, row in df_max_rev_by_year.iterrows():
        sql = 'SELECT id                                                              \
               FROM movies                                                            \
               WHERE (                                                                \
                   release_year = ' + str(int(row['release_year'])) + ')              \
                   AND (revenue_adj_mil = ' + str(row['MAX(revenue_adj_mil)']) + '    \
               );'
        movieid = sql_to_scalar(moviecu, sql)
        list_of_max_movies.append(movieid)

    list_as_string = ''
    for movieid in list_of_max_movies:
        list_as_string = list_as_string + ', ' + str(movieid)
    list_as_string = list_as_string[2:]

    for field_selected in fields_to_process:
        sql = 'SELECT id, release_year, original_title, ' + field_selected + '    \
               FROM movies                                                        \
               WHERE id IN (' + list_as_string + ')                               \
               ORDER BY release_year;'
        df_max_movies = sql_to_dataframe(moviecu, sql)
        df_max_movies.to_csv(path + 'max_movies_by_year_for_' + field_selected + '.csv')

    movieconn.close()

    print('Done!')

main()
