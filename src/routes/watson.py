from flask import Blueprint, request, jsonify
from services.watson_service import WatsonService

watson_bp = Blueprint('watson', __name__)

@watson_bp.route('/chat', methods=['POST'])
def chat():
    user_input = request.json.get('message')
    if not user_input:
        return jsonify({'error': 'No message provided'}), 400

    response = WatsonService.send_message(user_input)
    return jsonify({'response': response})