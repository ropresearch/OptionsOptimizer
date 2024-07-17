import os
# import schwab
from dotenv import load_dotenv
import requests
import base64




def auth(): 
    load_dotenv()
    APP_KEY = os.getenv('APP_KEY')
    APP_SECRET = os.getenv('APP_SECRET')
    
    authUrl = f'https://api.schwabapi.com/v1/oauth/authorize?client_id={APP_KEY}&redirect_uri=https://127.0.0.1'
    
    print(f"Click to authenticate: {authUrl}")
    
    returnedLink = input("Paste the redirect URL here:")
    
    code = f"{returnedLink[returnedLink.index('code=')+5:returnedLink.index('%40')]}@"
    
    
    headers = {'Authorization': f'Basic {base64.b64encode(bytes(f"{APP_KEY}:{APP_SECRET}", "utf-8")).decode("utf-8")}', 'Content-Type': 'application/x-www-form-urlencoded'}
    data = {'grant_type': 'authorization_code', 'code': code, 'redirect_uri': 'https://127.0.0.1'}
    
    response = requests.post('https://api.schwabapi.com/v1/oauth/token', headers=headers, data=data)
    tD = response.json()
    
    access_token = tD['access_token']
    refresh_token = tD['refresh_token']
    
    print(access_token)
    print(refresh_token)


def get_access_token():
    load_dotenv()
    APP_KEY = os.getenv('APP_KEY')
    APP_SECRET = os.getenv('APP_SECRET')
    REFRESH_TOKEN = os.getenv('REFRESH_TOKEN')
    CALLBACK_URL = os.getenv('CALLBACK_URL')
    headers = {'Authorization': f'Basic {base64.b64encode(bytes(f"{APP_KEY}:{APP_SECRET}", "utf-8")).decode("utf-8")}', 'Content-Type': 'application/x-www-form-urlencoded'}
    data = {'grant_type': 'refresh_token', 'refresh_token': REFRESH_TOKEN , 'redirect_uri': CALLBACK_URL}

    response = requests.post('https://api.schwabapi.com/v1/oauth/token', headers=headers, data=data)
    tD = response.json()

    access_token = tD['access_token']
    refresh_token = tD['refresh_token']

    print(access_token)
    print(refresh_token)

    return access_token
