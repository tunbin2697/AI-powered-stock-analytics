# Stock Price AI Application

A Flask-based web application for stock price prediction using AI/ML models.

---

## Prerequisites

- Python 3.9 or 3.10 (you can chose which python to run by unsing cmp and run "where python", it will return the path to your python with its version ex C:\Users\ADMIN\AppData\Local\Programs\Python\Python310\python.exe
C:\Users\ADMIN\AppData\Local\Programs\Python\Python313\python.exe. Use that part instead of python ex: C:\Users\ADMIN\AppData\Local\Programs\Python\Python310\python.exe -m venv venv )
- Windows PowerShell or Command Prompt
- [Git](https://git-scm.com/)
- [Visual Studio Code](https://code.visualstudio.com/) (recommended)

---

## Setup Instructions

### 1. Create a Project Folder

Create a new folder for your project (e.g., `AI-powered-web` or a name of your choice):

```powershell
mkdir AI-powered-web
cd AI-powered-web
```

### 2. Set Up Python Virtual Environment

Create a virtual environment **outside** the cloned repository to avoid tracking `venv` with Git:

```powershell
python -m venv venv
```

Activate the virtual environment:

```powershell
.\venv\Scripts\Activate.ps1
```

> **Note:** If you encounter an execution policy error, run:
> ```powershell
> Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
> ```

### 3. Clone the Repository

Clone the project repository into your project folder:

```powershell
git clone <repo-url>
```

Replace `<repo-url>` with the actual repository URL.

### 4. Install Dependencies

Navigate into the cloned repository folder, then install dependencies:

```powershell
cd AI-powered-stock-analytics
pip install -r requirements.txt
```

### 5. Run the Application

Navigate to the `src` directory and start the Flask app:

```powershell
cd src
python app.py
```

The application will run at [http://127.0.0.1:5000](http://127.0.0.1:5000) by default.

---

## Visual Studio Code Setup

1. Open the project folder in VS Code.
2. Press `Ctrl+Shift+P` and select **Python: Select Interpreter**.
3. Choose the interpreter from your `venv` folder (e.g., `.venv\Scripts\python.exe`).

---

## Project Structure

```
AI-powered-web/
├── venv/                      # Virtual environment (outside repo)
├── AI-powered-stock-analytics/ # Cloned repository
│   ├── src/
│   │   ├── app.py
│   │   ├── config/
│   │   ├── routes/
│   │   ├── static/
│   │   └── templates/
│   ├── requirements.txt
│   └── README.md
```

---

## Troubleshooting

- **PowerShell Execution Policy Error:**  
  Run `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`
- **Module Not Found Error:**  
  Ensure the virtual environment is activated and dependencies are installed.
- **Port Already in Use:**  
  Change the port in config settings or stop the process using the port.

---

## Features

- Stock price prediction using ML models
- Interactive web interface
- Real-time data processing
- Audio processing capabilities
- Watson AI integration
