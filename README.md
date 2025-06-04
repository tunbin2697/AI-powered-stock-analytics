# Stock Price AI Application

This is a Flask-based web application for stock price prediction using AI/ML models.

## Prerequisites

- Python 3.10 or higher
- Windows PowerShell or Command Prompt

## Setup Instructions

### 1. Create Virtual Environment

First, navigate to the project root directory and create a virtual environment with Python 3.10:

```bash
python -m venv venv
```

### 2. Activate Virtual Environment

Activate the virtual environment using PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

**Note:** If you encounter an execution policy error, run this command first:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### 3. Install Dependencies

Install the required packages from requirements.txt:

```bash
pip install -r requirements.txt
```

### 4. Run the Application

Navigate to the src directory and run the Flask application:

```bash
cd src
python app.py
```

The application will start running on `http://127.0.0.1:5000` by default.

## Project Structure

```
final project/
├── venv/                 # Virtual environment
├── src/                  # Source code
│   ├── app.py           # Main Flask application
│   ├── config/          # Configuration files
│   ├── routes/          # API routes
│   ├── static/          # Static files (CSS, JS, assets)
│   └── templates/       # HTML templates
├── requirements.txt     # Python dependencies
└── README.md           # This file
```

## Troubleshooting

- **PowerShell Execution Policy Error**: Run `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`
- **Module Not Found Error**: Ensure virtual environment is activated and dependencies are installed
- **Port Already in Use**: Change the port in config settings or kill the process using the port

## Features

- Stock price prediction using ML models
- Interactive web interface
- Real-time data processing
- Audio processing capabilities
- Watson AI integration