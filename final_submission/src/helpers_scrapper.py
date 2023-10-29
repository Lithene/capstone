"""Helper file for scrapping for instagram / facebook posts"""

############################### LIBRARIES #####################################

from instagrapi import Client
import json
import pandas as pd

############################### INSTAGRAM #####################################

def login_instgram( username, password):
    # Either create or convert your personal insta account to a professional account
    cl = Client()
    # adds a random delay between 1 and 3 seconds after each request
    cl.delay_range = [1, 2]
    cl.login(username, password)
    cl.dump_settings("session.json")

    return cl

def instagram_scrapper(client, user_list):
    userinfo_df = pd.DataFrame()
    allmedia_df = pd.DataFrame()

    for agent in user_list:
        print(f"fetching {agent}...")
        # here you can get the single user biography
        insta_user_id = cl.user_id_from_username(agent) # get the user id
        user_info = pd.json_normalize(cl.user_info(insta_user_id).dict())
        userinfo_df = pd.concat( [userinfo_df , user_info], ignore_index=True)

        # here to get single user media
        insta_medias = cl.user_medias(user_id = insta_user_id, amount=int(user_info['media_count'].iloc[0]))
        
        counter = 0
        usermedia_df = pd.DataFrame()
        for i in range(len(insta_medias)):
            print(f"extract media no.: {counter}")
            json_media = pd.json_normalize(insta_medias[i].dict())
            usermedia_df = pd.concat( [usermedia_df , json_media], ignore_index=True)
            counter += 1

        allmedia_df = pd.concat( [allmedia_df , usermedia_df], ignore_index=True)

    return userinfo_df, allmedia_df


def export_instapost(userinfo_df, allmedia_df, export_path):

    print("userinfo_df", userinfo_df.shape)
    print("allmedia_df", allmedia_df.shape)

    if not os.path.exists(export_path):
        os.makedirs(export_path)

    today = date.today().strftime("%y%m%d")
    userinfo_df.to_csv(export_path + f'instagraphi_bio_{today}.csv', index=False)
    allmedia_df.to_csv(export_path + f'instagraphi_media_{today}.csv', index=False)