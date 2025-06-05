from flask import Blueprint, request, jsonify
from services.ml_service import ml_service
import traceback

ml_bp = Blueprint('ml', __name__, url_prefix='/api/ml')

VALID_PERIODS = ['1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max']

@ml_bp.route('/train-and-evaluate', methods=['POST'])
def train_and_evaluate_models():
    """Main endpoint for training and evaluation"""
    try:
        data = request.get_json() or {}
        
        # Extract parameters
        stock_code = data.get('stock_code', '').strip().upper()
        period = data.get('period', '1y')
        split_ratio = data.get('split_ratio', 0.8)
        epochs = data.get('epochs', 50)
        models_to_train = data.get('models_to_train', [])
        retrain = data.get('retrain', True)

        # Validation
        errors = []
        
        if not stock_code:
            errors.append("stock_code is required")
        
        if period not in VALID_PERIODS:
            errors.append(f"Invalid period. Valid options: {', '.join(VALID_PERIODS)}")
        
        if not isinstance(models_to_train, list) or not models_to_train:
            errors.append("models_to_train must be a non-empty list")
        
        # Validate model types
        supported_models = list(ml_service.supported_models.keys())
        invalid_models = [m for m in models_to_train if m not in supported_models]
        if invalid_models:
            errors.append(f"Unsupported models: {invalid_models}")
        
        # Validate numeric parameters
        try:
            split_ratio = float(split_ratio)
            if not (0.1 <= split_ratio <= 0.95):
                errors.append("split_ratio must be between 0.1 and 0.95")
        except (TypeError, ValueError):
            errors.append("split_ratio must be a valid number")
        
        try:
            epochs = int(epochs)
            if epochs < 1 or epochs > 1000:
                errors.append("epochs must be between 1 and 1000")
        except (TypeError, ValueError):
            errors.append("epochs must be a valid integer")

        if errors:
            return jsonify({"status": "error", "errors": errors}), 400

        # Run ML pipeline
        results = ml_service.train_and_evaluate_models(
            stock_code=stock_code,
            period=period,
            models_to_train=models_to_train,
            split_ratio=split_ratio,
            epochs=epochs,
            retrain=retrain
        )

        # Determine HTTP status code
        status_code = 200
        if results['status'] == 'error':
            status_code = 500
        elif results['status'] == 'partial_success':
            status_code = 206

        return jsonify(results), status_code

    except Exception as e:
        traceback.print_exc()
        return jsonify({
            "status": "error",
            "message": f"Server error: {str(e)}"
        }), 500

@ml_bp.route('/models/status/<stock_code>', methods=['GET'])
def get_models_status(stock_code):
    """Get status of available models for a stock"""
    try:
        stock_code = stock_code.strip().upper()
        status = ml_service.get_model_status(stock_code)
        return jsonify(status)
    except Exception as e:
        return jsonify({"error": f"Failed to get model status: {str(e)}"}), 500

@ml_bp.route('/models/supported', methods=['GET'])
def get_supported_models():
    """Get list of supported model types"""
    try:
        return jsonify({
            "supported_models": [
                {
                    "id": "linear_regression",
                    "label": "Linear Regression",
                    "supports_epochs": False
                },
                {
                    "id": "random_forest_regressor", 
                    "label": "Random Forest",
                    "supports_epochs": True
                },
                {
                    "id": "prophet", 
                    "label": "Prophet",
                    "supports_epochs": False
                },
                {
                    "id": "arima", 
                    "label": "ARIMA",
                    "supports_epochs": False
                },
                {
                    "id": "lstm", 
                    "label": "LSTM",
                    "supports_epochs": True
                },
                {
                    "id": "svm_regressor",
                    "label": "SVM Regressor", 
                    "supports_epochs": False
                }
            ]
        })
    except Exception as e:
        return jsonify({"error": f"Failed to get supported models: {str(e)}"}), 500

@ml_bp.route('/stock/validate-data', methods=['GET'])
def validate_stock_data():
    """Validate stock data availability for ML"""
    stock_code = request.args.get('code', '').strip().upper()
    period = request.args.get('period', '1y')
    
    if not stock_code:
        return jsonify({"error": "Stock code is required"}), 400
    
    if period not in VALID_PERIODS:
        return jsonify({"error": f"Invalid period"}), 400
    
    try:
        from services.stock_data_service import StockDataService
        service = StockDataService()
        
        validation_result = service.validate_stock_code(stock_code)
        
        if validation_result.get('valid', False):
            # Get sample data
            sample_data = service.get_historical_data_json(stock_code, "5d")
            
            return jsonify({
                "stock_code": stock_code,
                "period": period,
                "available": True,
                "message": f"Data available for {stock_code}",
                "company_name": validation_result.get('company_name', 'N/A'),
                "sample_data": sample_data[-3:] if sample_data else []
            })
        else:
            return jsonify({
                "stock_code": stock_code,
                "period": period,
                "available": False,
                "message": validation_result.get('message', 'Stock code not found')
            }), 404
            
    except Exception as e:
        return jsonify({
            "stock_code": stock_code,
            "period": period,
            "available": False,
            "message": f"Error validating data: {str(e)}"
        }), 500