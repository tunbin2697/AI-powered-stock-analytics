from flask import jsonify
import requests

class WatsonService:
    def __init__(self, api_key, url):
        self.api_key = api_key
        self.url = url

    def send_message(self, message):
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {self.api_key}'
        }
        data = {
            'input': {
                'text': message
            }
        }
        response = requests.post(self.url, json=data, headers=headers)
        if response.status_code == 200:
            return jsonify(response.json())
        else:
            return jsonify({'error': 'Failed to get response from Watson API'}), response.status_code

    def get_response(self, user_message):
        return self.send_message(user_message)