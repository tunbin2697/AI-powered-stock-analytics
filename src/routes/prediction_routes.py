from flask import Blueprint, jsonify, request
from services.prediction_service import PredictionService

prediction_bp = Blueprint('prediction', __name__)

@prediction_bp.route('/api/stock/predict', methods=['GET'])
def predict_stock_price():
    """Simple stock prediction endpoint"""
    try:
        symbol = request.args.get('code')
        period = request.args.get('period', '1y')
        days = int(request.args.get('days', 7))
        model = request.args.get('model', 'linear_regression')
        
        if not symbol:
            return jsonify({"error": "Stock code parameter is required"}), 400
        
        prediction_service = PredictionService()
        result = prediction_service.predict(symbol, period, days, model)
        
        if "error" in result:
            return jsonify(result), 400
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({"error": f"Service error: {str(e)}"}), 500

@prediction_bp.route('/api/stock/predict/models/<symbol>', methods=['GET'])
def get_available_models(symbol):
    """Get available models for a symbol"""
    try:
        prediction_service = PredictionService()
        models = prediction_service.get_available_models(symbol)
        
        return jsonify({
            "symbol": symbol.upper(),
            "available_models": models,
            "total_models": len(models)
        })
        
    except Exception as e:
        return jsonify({"error": f"Failed to get models: {str(e)}"}), 500