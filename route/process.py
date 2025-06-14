from flask import Blueprint, jsonify, request
from service.process.yahoo_finance_process import YahooFinanceProcessor
from service.process.fred_process import FredProcessor
from service.process.newsapi_process import NewsApiProcessor

yahooFinanceProcess_service = YahooFinanceProcessor()
fredProcessor_service = FredProcessor()
newsapiProcessor_service = NewsApiProcessor()

process_bp = Blueprint("process", __name__, url_prefix="/lab2")

@process_bp.route("/process/yf", methods=['GET'])
def process_yf():
  
  ticker = request.args.get('ticker')
  start = request.args.get('start')
  end = request.args.get('end')
  interval = request.args.get('interval', "1d")
  
  data,_ = yahooFinanceProcess_service.process_raw_data(ticker=ticker, start=start, end=end, interval=interval)
  return jsonify(data.to_dict('records'))


@process_bp.route("/process/fred", methods=['GET'])
def process_fred():
  
  # ticker = request.args.get('ticker')
  start = request.args.get('start')
  end = request.args.get('end')
  
  data,_ = fredProcessor_service.process_raw_data(start=start, end=end)
  return jsonify(data.to_dict('records'))

@process_bp.route("/process/newsapi", methods=['GET'])
def process_newsapi():
  
  ticker = request.args.get('ticker')
  start = request.args.get('start')
  end = request.args.get('end')
  
  embeddings, dates, processed_articles = newsapiProcessor_service.process_raw_data(ticker=ticker, start=start, end=end)
  
  response_data = {
    'embeddings': embeddings.tolist() if len(embeddings) > 0 else [],
    'dates': dates,
    'article_count': len(processed_articles),
    'embedding_dimension': embeddings.shape[1] if len(embeddings) > 0 else 0,
    'articles_summary': [
      {
        'title': article.get('title', '')[:100],  
        'date': article.get('publishedAt', ''),
        'source': article.get('source', 'Unknown')
      } for article in processed_articles[:10]  
    ]
  }
  
  return jsonify(response_data)
