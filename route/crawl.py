from flask import Blueprint, jsonify, request
from service.crawl.yahoo_finance import YahooFinanceCrawl
from service.crawl.fred import FredCrawl
from service.crawl.newsapi import NewsapiCrawl

yahooFinanceCrawl_service = YahooFinanceCrawl()
fredCrawl_service = FredCrawl()
newsapi_service = NewsapiCrawl()

crawl_bp = Blueprint("crawl", __name__, url_prefix="/lab2")

@crawl_bp.route("/crawl/yf", methods=['GET'])
def craw_yf():
  
  ticker = request.args.get('ticker')
  start = request.args.get('start')
  end = request.args.get('end')
  interval = request.args.get('interval', "1d")
  
  data = yahooFinanceCrawl_service.load_data(ticker=ticker, start=start, end=end, interval=interval)
  return jsonify(data.to_dict('records'))

@crawl_bp.route("/crawl/fred", methods=['GET'])
def craw_fred():
  
  series_id = request.args.get('series_id')
  start = request.args.get('start')
  end = request.args.get('end')
  
  data = fredCrawl_service.load_data(series_id=series_id, start=start, end=end)
  return jsonify(data.to_dict('records'))

@crawl_bp.route("/crawl/newsapi", methods=['GET'])
def craw_newsapi():
  
  ticker = request.args.get('ticker')
  start = request.args.get('start')
  end = request.args.get('end')
  
  data = newsapi_service.load_data(ticker=ticker, start=start, end=end)
  return data

