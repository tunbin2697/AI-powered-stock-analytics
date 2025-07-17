# 🚀 AI-Powered Stock Analytics Platform

[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.46.1-red.svg)](https://streamlit.io/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.19.0-orange.svg)](https://tensorflow.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.6.0-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

A comprehensive **Streamlit-based dashboard** that leverages advanced AI/ML models and cutting-edge technologies for intelligent stock price prediction and financial analysis. This platform integrates multiple machine learning algorithms, real-time data processing, and natural language processing to provide sophisticated financial market insights.

## 🌟 Key Features

### 🤖 **Advanced AI/ML Models**
- **T-GNN++ (Temporal Graph Neural Network)** - State-of-the-art multi-asset forecasting
- **LSTM (Long Short-Term Memory)** - Deep learning for time series prediction
- **ARIMA** - Statistical modeling for trend analysis  
- **Random Forest** - Ensemble learning for robust predictions
- **Linear Regression** - Baseline modeling with feature scaling

### 📊 **Comprehensive Analytics**
- Real-time stock data fetching with Yahoo Finance integration
- Technical indicators calculation (RSI, MACD, Bollinger Bands, SMA)
- News sentiment analysis with financial data correlation
- Macroeconomic indicators integration (FRED API)
- Interactive visualizations with Plotly

### � **AI-Powered Chatbot**
- LangChain integration with Google Gemini AI
- Natural language querying of financial data
- Contextual analysis of charts and trends
- Intelligent data interpretation and insights

### 🧠 **Multi-Modal Data Processing**
- Stock price data collection and preprocessing
- News sentiment analysis with FinBERT
- Macroeconomic indicators integration
- Technical analysis features engineering

---

## 🛠️ Technology Stack

| Component | Technology | Version |
|-----------|------------|---------|
| **Frontend** | Streamlit | 1.46.1 |
| **Backend** | Python | 3.10+ |
| **Deep Learning** | TensorFlow, PyTorch | 2.19.0, 2.6.0 |
| **Graph Neural Networks** | PyTorch Geometric | 2.6.1 |
| **NLP** | Transformers, LangChain | 4.53.0, 0.3.26 |
| **Data Processing** | pandas, numpy | 2.2.2, 2.0.2 |
| **Finance APIs** | yfinance, FRED | 0.2.64 |
| **Visualization** | Plotly, Matplotlib | 6.2.0, 3.10.3 |
| **AI Chat** | Google Gemini, LangChain | Latest |

---

## 📋 Prerequisites

- **Python 3.10+** (recommended)
- **CUDA compatible GPU** (optional, for faster training)
- **Google API Key** for Gemini AI integration
- **Internet connection** for real-time data fetching

> **💡 Check your Python version:**
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

# Activate virtual environment (Windows)
.\venv\Scripts\Activate.ps1

# For Linux/Mac
source venv/bin/activate
```

> **⚠️ Windows Users:** If you encounter execution policy errors:
> ```powershell
> Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
> ```

### 3. **Install Dependencies**

```powershell
pip install -r requirements.txt
```

### 4. **Environment Configuration**

Create a `.env` file in the root directory:

```env
GOOGLE_API_KEY=your_google_gemini_api_key
FRED_API_KEY=your_fred_api_key_optional
```

### 5. **Launch the Application**

```powershell
cd src
streamlit run app.py
```

🎉 **Success!** Your browser will automatically open to: `http://localhost:8501`

---

## 📁 Project Architecture

```
AI-powered-stock-analytics/
├── 📁 src/                          # Main application source
│   ├── � app.py                    # Streamlit application entry point
│   ├── 🤖 chatbot.py                # AI chatbot implementation
│   ├── � TGNN.md                   # T-GNN++ model documentation
│   ├── 🧪 test.py                   # Testing utilities
│   │
│   ├── �📁 config/                   # Configuration settings
│   │   ├── __init__.py
│   │   └── settings.py              # App configuration
│   │
│   ├── 📁 models/                   # Pre-trained ML models
│   │   ├── � *.h5                  # TensorFlow/Keras models
│   │   ├── � *.joblib              # Scikit-learn models
│   │   └── 🔥 *.pth                 # PyTorch models (T-GNN++)
│   │
│   ├── 📁 services/                 # Business logic services
│   │   ├── yf_service.py            # Yahoo Finance data service
│   │   ├── news_service.py          # News data collection
│   │   ├── macro_service.py         # Macroeconomic data
│   │   ├── data_prepare_service.py  # Data preprocessing
│   │   ├── visualize_service.py     # Chart generation
│   │   ├── lstm_service.py          # LSTM model service
│   │   ├── arima_service.py         # ARIMA model service
│   │   ├── random_forest_service.py # Random Forest service
│   │   ├── linear_regression_service.py # Linear regression
│   │   └── tgnn_service.py          # T-GNN++ model service
│   │
│   ├── 📁 data/                     # Data storage
│   │   ├── 📁 raw/                  # Raw data files
│   │   └── 📁 processed/            # Processed datasets
│   │
│   ├── 📁 utils/                    # Utility functions
│   │   └── quick_eda.py             # Exploratory data analysis
│   │
│   └── 📁 pygooglenews/             # Google News integration
│
├── � lab_1_2_pipeline_(2).ipynb    # Jupyter notebook for experiments
├── 📄 requirements.txt              # Python dependencies
├── 🖼️ tgnn_best_test.png           # Model performance visualization
└── 📖 README.md                     # Project documentation
```

---

## 🎯 Model Capabilities

### **T-GNN++ (Temporal Graph Neural Network)**
- **Multi-asset forecasting** with graph attention mechanisms
- **Spatial relationships** modeling between correlated stocks
- **Long-term and short-term** temporal dependencies
- **State-of-the-art performance** for financial prediction

### **Traditional ML Models**
| Model | Use Case | Strengths |
|-------|----------|-----------|
| **LSTM** | Time series prediction | Sequential pattern recognition |
| **ARIMA** | Trend analysis | Statistical foundation |
| **Random Forest** | Ensemble prediction | Robust to overfitting |
| **Linear Regression** | Baseline modeling | Interpretability |

---

## � Usage Guide

### **1. Stock Analysis Dashboard**
- Select stock symbols for analysis
- Choose time periods and data ranges
- View technical indicators and charts
- Get AI-powered insights via chatbot

### **2. Prediction Models**
- Train multiple ML models on historical data
- Compare model performances
- Generate future price predictions
- Visualize prediction confidence intervals

### **3. News Sentiment Analysis**
- Fetch recent financial news
- Analyze sentiment with FinBERT
- Correlate news sentiment with price movements
- Generate sentiment-based insights

### **4. Chatbot Interaction**
- Ask questions about displayed data
- Get explanations of chart patterns
- Request analysis of specific time periods
- Receive AI-generated market insights

---

## 🔧 Configuration

### **Model Settings**
- Adjust prediction horizons in `config/settings.py`
- Modify technical indicator parameters
- Configure data fetching intervals
- Set model hyperparameters

### **API Configuration**
- Google Gemini API for chatbot functionality
- FRED API for macroeconomic data (optional)
- Yahoo Finance for stock data (free)

---

## 📊 Model Performance

### **T-GNN++ Results**
- **Best Test Performance**: Visualized in `tgnn_best_test.png`
- **Multi-asset correlation**: Captures cross-stock dependencies
- **Temporal modeling**: Superior long-term prediction accuracy

### **Benchmark Comparison**
- LSTM: Strong sequential modeling
- ARIMA: Reliable trend analysis
- Random Forest: Robust ensemble performance
- Linear Regression: Fast baseline predictions

---

## 🐛 Troubleshooting

### **Common Issues & Solutions**

**🔴 CUDA/GPU Issues**
```powershell
# Install CPU-only versions
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install tensorflow-cpu
```

**🔴 API Key Issues**
- Ensure `.env` file is in the root directory
- Verify Google API key has Gemini access
- Check FRED API key permissions

**🔴 Memory Issues**
- Reduce batch sizes in model training
- Use smaller time windows for analysis
- Close unused browser tabs

**🔴 Port Already in Use**
```powershell
# Change Streamlit port
streamlit run app.py --server.port 8502
```

---

## 🧪 Development & Testing

### **Running Tests**
```powershell
cd src
python test.py
```

### **Model Training**
```powershell
# Use the Jupyter notebook for experiments
jupyter notebook lab_1_2_pipeline_\(2\).ipynb
```

### **Adding New Models**
1. Create service file in `services/`
2. Implement training and prediction methods
3. Add to main dashboard in `app.py`
4. Update configuration settings

---

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. **Fork** the repository
2. **Create** a feature branch: `git checkout -b feature/amazing-feature`
3. **Commit** your changes: `git commit -m 'Add: amazing feature'`
4. **Push** to the branch: `git push origin feature/amazing-feature`
5. **Open** a Pull Request

### **Development Guidelines**
- Follow PEP 8 coding standards
- Add docstrings to all functions
- Include unit tests for new features
- Update documentation as needed

---

## 📈 Roadmap

- [ ] **Real-time trading integration**
- [ ] **Portfolio optimization features**
- [ ] **Risk management tools**
- [ ] **Mobile-responsive design**
- [ ] **Advanced backtesting framework**
- [ ] **Multi-language support**
- [ ] **Cloud deployment options**

---

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

---

## 📧 Contact & Support

- **GitHub**: [@tunbin2697](https://github.com/tunbin2697)
- **Project Link**: [AI-powered-stock-analytics](https://github.com/tunbin2697/AI-powered-stock-analytics)

---

## 🙏 Acknowledgments

- **Google Gemini** for advanced AI capabilities
- **PyTorch Geometric** for graph neural network implementation
- **Streamlit** for the beautiful web interface
- **Yahoo Finance** for reliable financial data
- **FinBERT** for financial sentiment analysis
- **LangChain** for AI integration framework

---

## 📧 Contact & Support

- **GitHub Issues**: [Report bugs or request features](https://github.com/tunbin2697/AI-powered-stock-analytics/issues)
- **Documentation**: [Wiki pages](https://github.com/tunbin2697/AI-powered-stock-analytics/wiki)
- **Discussions**: [Community discussions](https://github.com/tunbin2697/AI-powered-stock-analytics/discussions)

---

<div align="center">

**⭐ Star this repository if you found it helpful!**
