# Website Test Runner

This repository contains `test_runner.py`, a Python script designed for automated website testing. It leverages the **Playwright** library to navigate websites, generate test flows, execute them, and produce detailed reports. The script is part of a larger system for automated testing, integrating with custom utilities and agent-based interactions to simulate user journeys on a website.

---

## Repository Structure

- **Dockerfile:** Defines the Docker container setup for running the test runner in a consistent environment.   
- **__init__.py:** Marks directories as Python package directories.  
- **__pycache__:** Directory for compiled Python bytecode.  
- **agent/:** Directory containing agent-related code for interacting with the test environment.  
- **computers/:** Directory with modules for browser control, including Computer and LocalPlaywrightComputer.  
- **examples/:** Directory with example scripts or configurations.  
- **utils.py:** Utility functions, including `create_response` and `check_blocklisted_url`.  
- **cli.py:** Command-line interface for the project (if applicable).  
- **main.py:** Entry point for running the application (if applicable).  
- **simple_cua_loop.py:** Simplified script for running a specific test loop (if applicable).  
- **test_runner.py:** The main script for generating and executing website test flows.  
- **requirements.txt:** Lists Python dependencies required for the project.

---

## Features

- **Website Navigation:** Automatically navigates to a specified website using Playwright.  
- **Test Flow Generation:** Discovers and generates user flows (e.g., login, checkout, search) based on website structure.  
- **Test Execution:** Executes generated test flows, handling actions like clicking, typing, and verifying content.  
- **Error Handling and Recovery:** Attempts to recover from errors during test execution and logs issues.  
- **Priority-Based Testing:** Prioritizes critical user flows (e.g., checkout, login) and supports custom priority categories.  
- **Reporting:** Generates detailed test reports, including success/failure status, errors, and agent assessments.  
- **Debug Mode:** Provides detailed logging and screenshot saving for debugging.  
- **User Personas:** Simulates different user behaviors (e.g., new visitor, returning customer) for varied testing scenarios.

---

## Prerequisites

- **Python:** Version 3.8 or higher.  
- **Playwright:** For browser automation.  
- **Dependencies:** Listed in `requirements.txt`.

---

## Installation

### Clone the Repository

```bash
git clone <repository-url>
cd <repository-directory>
```

### Set Up a Virtual Environment (optional but recommended)

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Install Playwright Browsers

```bash
playwright install
```

### (Optional) Build Docker Image

If using Docker, build the container:

```bash
docker build -t website-test-runner .
```

---

## Usage

Run `test_runner.py` from the command line to start the testing process. The script prompts for configuration details, such as the website URL, number of test flows, and debug options.

### Example Command

```bash
python test_runner.py
```

### Interactive Prompts

When you run the script, it will ask for:
- **Website URL:** The website to test (e.g., `https://example.com`).  
- **Maximum Test Flows:** Number of user flows to generate (default: 5).  
- **Output File:** Name of the file to save test flows (default: `test_flows.txt`).  
- **Wait Time:** Time to wait between actions in seconds (default: 0.5).  
- **Headless Mode:** Whether to run the browser in headless mode (y/n, default: n).  
- **Debug Mode:** Enable detailed logging and screenshots (y/n, default: n).  
- **Priority Category:** Optional category to prioritize (e.g., `checkout`).

### Example Interaction

```
Enter the website URL to test (e.g., https://example.com): https://example.com
Maximum number of test flows (default: 5): 3
Output text file name (default: test_flows.txt): test_flows.txt
Wait time between actions in seconds (default: 0.5): 0.5
Run in headless mode? (y/n, default: n): n
Enable debug mode? (y/n, default: n): y
Enter a flow category to prioritize (optional): login

Starting browser...
🔍 Do you want to execute the test flows? (y/n): y
Do you want to customize test execution parameters? (y/n): n
```

### Output

- **Test Flows:** Saved to the specified output file (e.g., `test_flows.txt`).  
- **Test Results:** Saved to `test_results_<domain>_<timestamp>.txt`.  
- **Final Report:** Saved to `final_test_report_<domain>_<timestamp>.txt`.  
- **Console Summary:** Displays a summary of test results, including success/failure rates and errors.

### Example Output Files

- **test_flows.txt**:
  ```
  Flow: User Journey: Login
  Description: Log in to access account
  Priority: High
  Steps:
    1. Click the 'Login/Signup' button
    2. Enter '+919876543210' in the mobile number field
    3. Enter '123456' in the OTP field
  ```

- **test_results_<domain>_<timestamp>.txt**:
  ```
  Website: https://example.com
  Timestamp: 2025-05-29 13:06:00
  Test Results:
  Flow: User Journey: Login
  Status: Passed
  Steps Completed: 3/3
  Duration: 5.23 seconds
  ```

---

## Key Components

- **WebsiteTestGenerator Class:** Core class for generating and executing test flows.  
  - Initializes with a browser instance, base URL, and configuration parameters.  
  - Methods include `discover_website_structure`, `execute_flow`, and `save_results`.  
- **handle_item Function:** Processes agent responses, executing browser actions and handling screenshots.  
- **send_message_to_agent Function:** Communicates with an agent to generate or refine test flows.  
- **Logging:** Uses Python's `logging` module for detailed logs, with debug mode for extra verbosity.

---

## Debugging

Enable debug mode (`debug=True`) to:
- Save screenshots for each browser action.  
- Log detailed information about agent interactions and errors.  
- Save raw agent responses for analysis.

Screenshots are saved as `screenshot_<action>_<timestamp>.png` in the working directory.

---

## Docker Usage

To run the script in a Docker container:
1. Build the image:
   ```bash
   docker build -t website-test-runner .
   ```
2. Run the container:
   ```bash
   docker run -it website-test-runner
   ```
3. Follow the interactive prompts as described above.

---

## License

See the `LICENSE` file for details.

---




