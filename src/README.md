# Project Structure and Development Guide

This document provides an overview of the `routes` and `services` folders in this project, along with guidance for custom frontend development and API testing.
The main idea of this project is using python to handle data fetch form yfinance and other source, process those data and then train models. Use those model for prediction purpose.

## Folder Structure

### `src/routes/`

The `src/routes/` directory contains the API endpoint definitions for the application. These files define the URLs that clients (like a web browser or mobile app) can use to interact with the backend. Each route typically handles incoming requests, validates parameters, calls the appropriate service to perform business logic, and then formats and returns a response (often in JSON format).

For example, `prediction_routes.py` defines endpoints related to stock predictions, such as `/api/stock/predict`.

### `src/services/`

The `src/services/` directory houses the core business logic of the application. Services are responsible for tasks like:

- Fetching data from external sources (e.g., stock data APIs).
- Interacting with the machine learning models (`ml_service.py`).
- Performing calculations and data transformations.
- Preparing data for predictions (`prediction_service.py`).

Routes call methods within these services to perform the actual work, keeping the route handlers clean and focused on request/response handling. This separation of concerns makes the codebase more modular, testable, and maintainable.

## Custom Frontend Development

The project come with a default `index.html`. For development purposes, you'll want to replace this with your own custom HTML, CSS, and JavaScript files to build your user interface.

**Steps to integrate your custom frontend:**

1.  **Locate the Static Files Directory:** Your Flask application will be configured to serve static files (HTML, CSS, JS) from a specific directory. This is  a folder named `static` contain JavaScripts and CSS; `template` contain main html file called `index.html` at the root of your `src` directory or alongside your main application file. I

2.  **Remove or Replace `index.html`:**

    - Navigate to the directory `templates`.
    - You can delete the existing `index.html`.
    - Place your main HTML file (e.g., `your_main_page.html`) in this directory. You should name it `index.html`, it will typically be served by default when accessing the root URL (e.g., `http://localhost:5000/`).
    - You also dont need to remove CSS and JavaScripts file, just leave it there and use your own in you html file (next step)

3.  **Add Your CSS and JS Files:**

    - Create your own CSS and JavaScripts files that will make you HTML file work.
    - Place your `.css` files and your `.js` files in somewhere. You should use the templates folder and copy the path to them in your html code as show in next step.
    - Link them in your HTML file:
      ```html
      <!-- In your HTML file -->
      <link rel="stylesheet" href="/your_styles.css" />
      <!-- ... -->
      <script src="/your_script.js"></script>
      ```
      _(Adjust paths if your `path` is different)_

4.  **Using an AI Assistant for Frontend Code:**
    - If you need help generating HTML, CSS, or JavaScript for your frontend, you can describe your desired layout, components, or functionality to an AI programming assistant (like GitHub Copilot).
    - For example, you could ask: "Generate HTML for a form with input fields for stock symbol, period, and days, and a button to submit." or "Write JavaScript to fetch data from `/api/stock/predict` and display the results."

## Testing API Endpoints with `curl`

`curl` is a command-line tool used for transferring data with URLs. It's very useful for testing your API endpoints directly.

**General Syntax:**

```bash
curl [options] "[URL]"
```

**Important:** Always enclose the URL in double quotes (`"`) if it contains special characters like `?` or `&` to prevent your shell from misinterpreting them.

**Examples based on `prediction_routes.py`:**

1.  **Predict Stock Price:**

    - Endpoint: `/api/stock/predict`
    - Method: `GET`
    - Parameters: `code` (symbol), `period` (optional), `days` (optional), `model` (optional)

    ```bash
    curl "http://localhost:5000/api/stock/predict?code=AAPL&period=1y&days=7&model=linear_regression"
    ```

    This command sends a GET request to predict the AAPL stock for the next 7 days using a 1-year historical period and the linear regression model.

2.  **Get Available Models for a Symbol:**

    - Endpoint: `/api/stock/predict/models/<symbol>`
    - Method: `GET`

    ```bash
    curl "http://localhost:5000/api/stock/predict/models/AAPL"
    ```

    This command fetches the available prediction models for the AAPL stock.

**Tips for using `curl`:**

- **Check Headers:** Use `-i` to include HTTP response headers:
  ```bash
  curl -i "http://localhost:5000/api/stock/predict?code=AAPL"
  ```
- **Verbose Output:** Use `-v` for more detailed information about the request and response:
  ```bash
  curl -v "http://localhost:5000/api/stock/predict?code=AAPL"
  ```
- **POST Requests (if you add them later):**
  ```bash
  curl -X POST -H "Content-Type: application/json" -d '{"key":"value"}' "http://localhost:5000/api/your_post_endpoint"
  ```

Remember to have your Flask application running when you are testing with `curl`. The default port is usually `5000`.
