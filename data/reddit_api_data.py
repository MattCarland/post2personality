
import requests
import datetime
import os
from dotenv import load_dotenv

load_dotenv()

class RedditUser:
    """ A class to get the data from a Reddit user

    Attributes:
        client_id (str): The client_id of the Reddit app
        secret_key (str): The secret_key of the Reddit app
        login_user (str): Your Reddit username

        Client_id and secret_key can be obtained by creating a Reddit app at
        https://www.reddit.com/prefs/apps and can be stored in a .env file


    Methods:
        get_comments(self, target_username)
            Returns a string of the first 1000 words from a Reddit user's comments
        get_user_infos(self, target_username)
            Returns a dict of a Reddit user's username, account creation date, and karma

    """
    def __init__(self, client_id, secret_key, login_user):
        """ Initialize an instance of RedditUser

        Arguments:
            client_id (str): The client_id of the Reddit app
            secret_key (str): The secret_key of the Reddit app
            login_user (str): Your Reddit username
        """
        self.client_id = client_id
        self.secret_key = secret_key
        self.login_user = login_user

        self.auth = requests.auth.HTTPBasicAuth(client_id, secret_key)
        self.headers = {'User-Agent': 'Post2Personality/0.0.1'}

        data = {
            'grant_type': 'password',
            'username': self.login_user,
            'password': os.environ['REDDIT_PW'].strip()
        }

        res = requests.post('https://www.reddit.com/api/v1/access_token',
                    auth=self.auth, data=data, headers=self.headers)

        self.token = res.json()['access_token']

        self.headers['Authorization'] = f'bearer {self.token}'

    def get_comments(self, target_username):
        """ Return a string of the first 1000 words from a Reddit user's comments

        Arguments:
            target_username (str): The username of the Reddit user
        """
        comments = requests.get(f'https://oauth.reddit.com/user/{target_username}/comments',
                    headers = self.headers).json()['data']['children']

        usercomments = []
        for comment in comments:
            usercomments.append(comment['data']['body'])
            result_string = " ".join(usercomments).split(" ")[:1000]

        return result_string

    def get_user_infos(self, target_username):
        """
        This function returns a dict of a Reddit user's username, account creation date, and karma

        Arguments:
            target_username (str): The username of the Reddit user
        """
        username = requests.get(f'https://oauth.reddit.com/user/{target_username}/about',
                    headers = self.headers).json()['data']['name']
        member_since = requests.get(f'https://oauth.reddit.com/user/{target_username}/about', headers = self.headers).json()['data']['created_utc']
        member_since = datetime.datetime.fromtimestamp(member_since).strftime('%Y-%m-%d %H:%M:%S')
        karma = requests.get(f'https://oauth.reddit.com/user/{target_username}/about', headers = self.headers).json()['data']['total_karma']
        user_infos = {
            'Username': username,
            'Member since': member_since,
            'karma': karma
            }

        return user_infos
