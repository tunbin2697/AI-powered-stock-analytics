# 🚀 AI-Powered Stock Analytics

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-3.1.1-green.svg)](https://flask.palletsprojects.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)

A comprehensive **Flask-based web application** that leverages multiple AI/ML models for advanced stock price prediction and analysis. This application integrates cutting-edge machine learning algorithms, real-time data processing, and audio analysis capabilities to provide intelligent stock market insights.

## 🌟 Key Features

### 📈 **Advanced ML Models**
- **LSTM (Long Short-Term Memory)** networks for time series prediction
- **ARIMA** statistical modeling for trend analysis
- **Prophet** forecasting for seasonal patterns
- **Random Forest** and **SVM** for ensemble predictions
- **Linear Regression** with feature scaling

### 🎯 **Smart Analytics**
- Real-time stock data fetching and caching
- Technical indicators (RSI, MACD, Bollinger Bands, SMA)
- Volume analysis and price momentum
- Interactive web interface with dynamic charts

### 🔊 **Audio Processing**
- Watson AI integration for speech analysis
- Audio-based market sentiment analysis
- Voice command capabilities

### 💾 **Data Management**
- Intelligent caching system for performance optimization
- Multiple timeframe support (5d, 6mo, 1y, 2y, 5y)
- Automated model persistence and loading

---

## 🛠️ Technology Stack

| Component | Technology |
|-----------|------------|
| **Backend** | Flask, Python 3.9+ |
| **ML/AI** | TensorFlow, scikit-learn, Prophet |
| **Data Processing** | pandas, numpy, yfinance |
| **Frontend** | HTML5, JavaScript, Vite |
| **Audio Processing** | Watson AI Services |
| **Deployment** | Flask WSGI, CORS enabled |

---

## 📋 Prerequisites

- **Python 3.9 or 3.10** (recommended)
- **Windows PowerShell** or Command Prompt
- **Git** for version control
- **Visual Studio Code** (recommended IDE)

> **💡 Tip:** Check your Python version:
> ```powershell
> python --version
> ```

---

## 🚀 Quick Start

### 1. **Clone the Repository**

```powershell
git clone https://github.com/tunbin2697/AI-powered-stock-analytics.git
cd AI-powered-stock-analytics
```

### 2. **Set Up Virtual Environment**

```powershell
# Create virtual environment
python -m venv venv

# Activate virtual environment
.\venv\Scripts\Activate.ps1
```

> **⚠️ Windows Users:** If you encounter execution policy errors:
> ```powershell
> Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
> ```

### 3. **Install Dependencies**

```powershell
pip install -r requirements.txt
```

### 4. **Launch the Application**

```powershell
cd src
python app.py
```

🎉 **Success!** Open your browser and navigate to: `http://127.0.0.1:5000`

---

## 📁 Project Architecture

```
AI-powered-stock-analytics/
├── 📁 src/                          # Main application source
│   ├── 🐍 app.py                    # Flask application entry point
│   ├── 📁 config/                   # Configuration settings
│   │   ├── __init__.py
│   │   └── settings.py
│   ├── 📁 models/                   # Pre-trained ML models
│   │   ├── 🤖 *.pkl                 # Scikit-learn models
│   │   ├── 🧠 *.h5                  # TensorFlow/Keras models
│   │   └── stock_models.py          # Model definitions
│   ├── 📁 routes/                   # API endpoints
│   │   ├── audio.py                 # Audio processing routes
│   │   ├── data_routes.py           # Data fetching routes
│   │   ├── ml_routes.py             # Machine learning routes
│   │   ├── prediction_routes.py     # Prediction endpoints
│   │   └── watson.py                # Watson AI integration
│   ├── 📁 services/                 # Business logic
│   │   ├── ml_service.py            # ML model services
│   │   ├── prediction_service.py    # Prediction logic
│   │   ├── stock_data_service.py    # Data fetching service
│   │   └── watson_service.py        # Watson AI service
│   ├── 📁 utils/                    # Utility functions
│   │   ├── data_processor.py        # Data preprocessing
│   │   └── validators.py            # Input validation
│   ├── 📁 static/                   # Frontend assets
│   │   └── 📁 assets/               # CSS, JS files
│   ├── 📁 templates/                # HTML templates
│   └── 📁 stock_cache/              # Cached stock data
├── 📄 requirements.txt              # Python dependencies
└── 📖 README.md                     # Project documentation
```

---

## 🎯 Supported ML Models

| Model Type | Algorithm | Use Case | File Format |
|------------|-----------|----------|-------------|
| **Deep Learning** | LSTM | Time series prediction | `.h5` |
| **Statistical** | ARIMA | Trend analysis | `.pkl` |
| **Forecasting** | Prophet | Seasonal patterns | `.pkl` |
| **Ensemble** | Random Forest | Feature-based prediction | `.pkl` |
| **Regression** | Linear/SVM | Price correlation | `.pkl` |

---

## 🔧 Configuration

### Environment Setup for VS Code

1. **Open the project** in Visual Studio Code
2. **Select Python interpreter**: `Ctrl+Shift+P` → "Python: Select Interpreter"
3. **Choose your virtual environment**: `./venv/Scripts/python.exe`

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/data/*` | GET | Stock data retrieval |
| `/ml/*` | POST | Machine learning operations |
| `/predict/*` | POST | Price predictions |
| `/audio/*` | POST | Audio processing |
| `/watson/*` | POST | Watson AI services |

---

## 🐛 Troubleshooting

### Common Issues & Solutions

**🔴 PowerShell Execution Policy Error**
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

**🔴 Module Not Found Error**
```powershell
# Ensure virtual environment is activated
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

**🔴 Port Already in Use**
```powershell
# Check what's using port 5000
netstat -ano | findstr :5000
# Kill the process or change port in config
```

**🔴 TensorFlow/CUDA Issues**
```powershell
# For CPU-only TensorFlow
pip uninstall tensorflow
pip install tensorflow-cpu
```

---

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. **Fork** the repository
2. **Create** a feature branch: `git checkout -b feature/amazing-feature`
3. **Commit** your changes: `git commit -m 'Add amazing feature'`
4. **Push** to the branch: `git push origin feature/amazing-feature`
5. **Open** a Pull Request

---

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

---

## 📧 Contact & Support

- **GitHub**: [@tunbin2697](https://github.com/tunbin2697)
- **Project Link**: [AI-powered-stock-analytics](https://github.com/tunbin2697/AI-powered-stock-analytics)

---

## 🙏 Acknowledgments

- **TensorFlow** team for deep learning frameworks
- **Facebook Prophet** for time series forecasting
- **Yahoo Finance** for stock data API
- **IBM Watson** for AI services
- **Flask** community for web framework support

---

<div align="center">

**⭐ Star this repository if you found it helpful!**


</div>
