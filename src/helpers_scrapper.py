"""Helper file for scrapping for instagram / facebook posts"""

############################### LIBRARIES #####################################

import json
import os
from datetime import date
import pandas as pd

# Instagram Scrapper
import instagrapi

# Web Scrapper - for facebook scrapping
from selenium import webdriver
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.firefox.firefox_binary import FirefoxBinary

# Facebook Scrapper
from io import StringIO
from facebook_page_scraper import Facebook_scraper

# Custom Libraries
import src.config as config

############################### INSTAGRAM #####################################

def login_instgram(username: str, password: str):
    """
    Initialise a instagram session for scrapping.
    Args:
        username (Str): Username login for instagram
        password (Str): Password login for instagram
    Returns:
        cl (Obj): Instance of instagrapi.Client()
    """
        
    # Either create or convert your personal insta account to a professional account
    cl = instagrapi.Client()
    # adds a random delay between 1 and 3 seconds after each request
    cl.delay_range = [1, 2]
    cl.login(username, password)
    cl.dump_settings("session.json")

    return cl

def check_session(client, username: str, password: str):
    """
    Check if the initialised session is still open.
    Args:
        client (Obj): Instance of instagrapi.Client()
        username (Str): Username login for instagram
        password (Str): Password login for instagram
    """

    # Initialises the session withon login
    session = client.get_timeline_feed()
    print(f"Session is: {session.status}")

def instagram_scrapper(client, user_list: list):
    """
    Extract the dataset from a instagram page, parse it from a json into a dataframe to be returned. 
    Args:
        client (Obj): Instance of instagrapi.Client()
        user_list (list): List of instagram handles to be extracting posts from
    Returns:
        userinfo_df (pandas.DataFrame): dataframe of user profile metadata
        allmedia_df (pandas.DataFrame): dataframe of instagram posts of user
    """

    userinfo_df = pd.DataFrame()
    allmedia_df = pd.DataFrame()

    for agent in user_list:
        print(f"fetching {agent}...")
        # here you can get the single user biography
        insta_user_id = client.user_id_from_username(agent) # get the user id
        user_info = pd.json_normalize(client.user_info(insta_user_id).dict())
        userinfo_df = pd.concat( [userinfo_df , user_info], ignore_index=True)

        # here to get all of single user media
        insta_medias = client.user_medias(user_id = insta_user_id, amount=int(user_info['media_count'].iloc[0]))
        
        counter = 0
        usermedia_df = pd.DataFrame()
        for i in range(len(insta_medias)):
            print(f"extract media no.: {counter}")
            json_media = pd.json_normalize(insta_medias[i].dict())
            usermedia_df = pd.concat( [usermedia_df , json_media], ignore_index=True)
            counter += 1

        allmedia_df = pd.concat( [allmedia_df , usermedia_df], ignore_index=True)

    return userinfo_df, allmedia_df

############################### FACEBOOK #####################################

def extract_json(page_name: str, posts_count: int, browser: str, timeout: int, headless: bool):
    """
    Extract the dataset from a facebook page, parse it from a json into a dataframe to be returned. 
    Args:
        page_name (Str): Name of the facebook page
        posts_count	(Int): Number of posts to scrap, if not passed default is 10
        browser	(Str): Which browser to use, either chrome or firefox. if not passed,default is chrome
        timeout	(Int): The maximum amount of time the bot should run for. If not passed, the default timeout is set to 10 minutes
        headless (Bool): Whether to run browser in headless mode?. Default is True
    Returns:
        df: dataframe of facebook page posts, returns an empty dataframe if posts cannot be scrapped
    """

    try:
        page_to_scrap = Facebook_scraper(page_name, posts_count, browser, timeout=timeout, headless=headless) #proxy=proxy
        json_data = page_to_scrap.scrap_to_json()
        df = pd.read_json(StringIO(json_data), orient='columns') #, lines=True, chunksize=1
        df = df.transpose()

        print(f"Extraction from {page_name} completed.")
        return df

    except Exception as e:
        print(f"{e}")
        return pd.DataFrame()
    

############################### DATA LOAD/EXPORT #####################################

def get_latest_csv(filepath: str, filename_filter: str):
    """
    Retrieve the current csv file as per the filter string from the specified filepath.
    Args:
        filepath (Str): The filepath to be imported from. Eg. "/path/to/filename.csv"
        filename_filter (Str): The file name filter eg. "full_features"
    Returns:
        dataset (pandas.DataFrame): The retrieved dataset

    """

    # Get list of all files
    all_files = [os.path.join(filepath, x) for x in os.listdir(filepath) if x.startswith(filename_filter) and x.endswith(".csv")]
    # Get latest file based on load date
    curr_filepath = max(all_files, key = os.path.getctime)
    dataset = pd.read_csv(curr_filepath, index_col= None)
    print(f"Dataset from {curr_filepath}")

    return dataset

def export_file_csv(dataframe, export_path: str, mode: str='x'):
    """
    Export the dataframe as a csv into specified filepath.
    Args:
        dataframe (pandas.DataFrame): The dataframe to be exported
        export_path	(Str): The filepath to be exported into. Eg. "/path/to/filename.csv"
        mode (Str): The mode of csv write
                    - 'x': default, no overwriting; exclusive file creation
                    - 'w+': overwrite mode
    """

    today = date.today().strftime("%y%m%d")
    
    try:
        dataframe.to_csv(export_path, index=False, mode=mode)
        print(f"Exported df {dataframe.shape} as {export_path}")
            
    except Exception as e:
        print(f"Could not export file as {e}")