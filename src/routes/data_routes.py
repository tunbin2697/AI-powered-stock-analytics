from flask import Blueprint, jsonify, request
from services.stock_data_service import fetch_stock_data, fetch_historical_data, StockDataService

data_bp = Blueprint('data', __name__)

VALID_PERIODS = ['1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max']

@data_bp.route('/api/stock/data', methods=['GET'])
def get_stock_data():
    """Get basic stock data"""
    stock_code = request.args.get('code', '').strip().upper()
    period = request.args.get('period', '1y')
    
    if not stock_code:
        return jsonify({"error": "Stock code is required"}), 400
    
    if period not in VALID_PERIODS:
        return jsonify({"error": f"Invalid period. Valid periods: {', '.join(VALID_PERIODS)}"}), 400
    
    try:
        service = StockDataService()
        
        # Get stock info and historical data
        stock_info = service.get_stock_info(stock_code)
        historical_data = service.get_historical_data_json(stock_code, period)
        
        if not historical_data:
            return jsonify({"error": f"No data found for {stock_code}"}), 404
        
        return jsonify({
            "stock_code": stock_code,
            "stock_info": stock_info,
            "period": period,
            "data": historical_data,
            "total_records": len(historical_data)
        })
        
    except Exception as e:
        return jsonify({"error": f"Failed to fetch data: {str(e)}"}), 500

@data_bp.route('/api/stock/analysis', methods=['GET'])
def get_technical_analysis():
    """Get technical analysis data (SMA, RSI)"""
    stock_code = request.args.get('code', '').strip().upper()
    period = request.args.get('period', '1y')
    
    if not stock_code:
        return jsonify({"error": "Stock code is required"}), 400
    
    if period not in VALID_PERIODS:
        return jsonify({"error": f"Invalid period. Valid periods: {', '.join(VALID_PERIODS)}"}), 400
    
    try:
        service = StockDataService()
        
        # Get historical data
        df = service.get_historical_data(stock_code, period)
        if df is None:
            return jsonify({"error": f"No data found for {stock_code}"}), 404
        
        # Calculate technical indicators
        indicators = service.calculate_technical_indicators(df)
        
        return jsonify({
            "stock_code": stock_code,
            "period": period,
            "indicators": indicators
        })
        
    except Exception as e:
        return jsonify({"error": f"Analysis failed: {str(e)}"}), 500

@data_bp.route('/api/stock/validate', methods=['GET'])
def validate_stock():
    """Validate if stock code exists"""
    stock_code = request.args.get('code', '').strip().upper()
    
    if not stock_code:
        return jsonify({"valid": False, "message": "Stock code is required"}), 400
    
    try:
        service = StockDataService()
        result = service.validate_stock_code(stock_code)
        
        if result["valid"]:
            return jsonify(result)
        else:
            return jsonify(result), 404
            
    except Exception as e:
        return jsonify({
            "valid": False,
            "message": f"Validation failed: {str(e)}"
        }), 500

# Legacy routes for backward compatibility
@data_bp.route('/api/stock/<symbol>', methods=['GET'])
def get_stock_data_legacy(symbol):
    try:
        stock_data = fetch_stock_data(symbol)
        return jsonify(stock_data), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@data_bp.route('/api/stock/<symbol>/historical', methods=['GET'])
def get_historical_data_legacy(symbol):
    start_date = request.args.get('start')
    end_date = request.args.get('end')
    try:
        historical_data = fetch_historical_data(symbol, start_date, end_date)
        return jsonify(historical_data), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500