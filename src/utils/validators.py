from flask import abort

def validate_stock_symbol(symbol):
    if not isinstance(symbol, str) or len(symbol) == 0:
        abort(400, description="Invalid stock symbol: must be a non-empty string.")
    
    if not symbol.isalpha():
        abort(400, description="Invalid stock symbol: must contain only alphabetic characters.")
    
    if len(symbol) > 5:
        abort(400, description="Invalid stock symbol: must not exceed 5 characters.")
    
    return True

def validate_date_format(date_str):
    from datetime import datetime
    
    try:
        datetime.strptime(date_str, '%Y-%m-%d')
    except ValueError:
        abort(400, description="Invalid date format: must be YYYY-MM-DD.")
    
    return True

def validate_request_data(data, required_fields):
    for field in required_fields:
        if field not in data:
            abort(400, description=f"Missing required field: {field}")
    
    return True