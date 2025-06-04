from datetime import datetime

class Stock:
    def __init__(self, symbol, name, price, volume, market_cap):
        self.symbol = symbol
        self.name = name
        self.price = price
        self.volume = volume
        self.market_cap = market_cap
        self.history = []

    def add_history(self, date, price):
        self.history.append({'date': date, 'price': price})

    def get_latest_price(self):
        return self.price

    def __repr__(self):
        return f"Stock(symbol={self.symbol}, name={self.name}, price={self.price}, volume={self.volume}, market_cap={self.market_cap})"

class StockHistory:
    def __init__(self, stock_symbol):
        self.stock_symbol = stock_symbol
        self.prices = []

    def add_price(self, date, price):
        self.prices.append({'date': date, 'price': price})

    def get_prices(self):
        return self.prices

    def __repr__(self):
        return f"StockHistory(stock_symbol={self.stock_symbol}, prices={self.prices})"