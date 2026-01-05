# EDA Notebooks - Complete Setup Guide

## Overview
This directory contains 7 Jupyter notebooks for exploratory data analysis of the OpenPowerlifting dataset. This guide will help you set up everything you need to run these notebooks, even if you're new to Python, Jupyter, or data analysis.

## Current Status

- **01_data_loading.ipynb** - ⚠️ **In Progress** - Still being worked on
- **02_overview.ipynb** - ⚠️ **In Progress** - Still being worked on
- **03_experience_analysis.ipynb** - ⚠️ **In Progress**
- **04_age_analysis.ipynb** - ⚠️ **In Progress**
- **05_meet_quality.ipynb** - ⚠️ **In Progress**
- **06_consistency_analysis.ipynb** - ⚠️ **In Progress**
- **07_insights_summary.ipynb** - ⚠️ **In Progress**
- **EDA_INSIGHTS.md** - ⚠️ **In Progress**

---

## What You'll Need

Before we start, you need three things:
1. **Python** - The programming language we'll use
2. **pip** - A tool to install Python packages (usually comes with Python)
3. **Jupyter** - An interactive notebook environment for running our analysis

---

## Step 1: Install Python and pip

### Check if Python is Already Installed

First, let's check if you already have Python installed. Open your terminal/command prompt and type:

**Windows (Command Prompt or PowerShell):**
```cmd
python --version
```

**Mac/Linux:**
```bash
python3 --version
```

If you see a version number (like `Python 3.11.5`), you're good! Skip to Step 2.

If you see an error like "command not found" or "python is not recognized", follow the installation instructions below for your operating system.

---

### Installing Python on Windows

1. **Download Python:**
   - Go to https://www.python.org/downloads/
   - Click the big yellow "Download Python" button
   - This will download the latest version (Python 3.x)

2. **Install Python:**
   - Run the downloaded installer (`.exe` file)
   - ⚠️ **IMPORTANT:** Check the box that says "Add Python to PATH" at the bottom of the installer window
   - Click "Install Now"
   - Wait for installation to complete

3. **Verify Installation:**
   - Open a **new** Command Prompt or PowerShell window (close and reopen if you had one open)
   - Type: `python --version`
   - You should see a version number

4. **Verify pip:**
   - Type: `pip --version`
   - You should see pip version information

**Note:** On Windows, you might need to use `py` instead of `python`. If `python --version` doesn't work, try `py --version`.

---

### Installing Python on Mac

**Option 1: Using Homebrew (Recommended)**

1. **Install Homebrew** (if you don't have it):
   - Open Terminal (Applications → Utilities → Terminal)
   - Paste this command and press Enter:
     ```bash
     /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
     ```
   - Follow the on-screen instructions

2. **Install Python:**
   ```bash
   brew install python3
   ```

3. **Verify Installation:**
   ```bash
   python3 --version
   pip3 --version
   ```

**Option 2: Using Official Installer**

1. **Download Python:**
   - Go to https://www.python.org/downloads/
   - Download the macOS installer

2. **Install Python:**
   - Open the downloaded `.pkg` file
   - Follow the installation wizard
   - Make sure to check "Add Python to PATH" if that option appears

3. **Verify Installation:**
   - Open Terminal
   - Type: `python3 --version`
   - Type: `pip3 --version`

**Note:** On Mac, you'll typically use `python3` and `pip3` instead of `python` and `pip`.

---

### Installing Python on Linux/Unix

Most Linux distributions come with Python pre-installed, but you may need to install it or update it.

**Ubuntu/Debian:**
```bash
sudo apt update
sudo apt install python3 python3-pip
```

**Fedora/RHEL/CentOS:**
```bash
sudo dnf install python3 python3-pip
```

**Arch Linux:**
```bash
sudo pacman -S python python-pip
```

**Verify Installation:**
```bash
python3 --version
pip3 --version
```

**Note:** On Linux, you'll typically use `python3` and `pip3` instead of `python` and `pip`.

---

## Step 2: Navigate to the Project Directory

You need to open your terminal/command prompt in the project directory.

### Windows

**Using File Explorer:**
1. Navigate to the project folder: `C:\Users\carte\OneDrive\Desktop\openpl`
2. In the address bar, type `cmd` and press Enter (this opens Command Prompt in that folder)

**Or using Command Prompt:**
```cmd
cd C:\Users\carte\OneDrive\Desktop\openpl
```

**Or using PowerShell:**
```powershell
cd C:\Users\carte\OneDrive\Desktop\openpl
```

### Mac/Linux

**Using Terminal:**
```bash
cd ~/OneDrive/Desktop/openpl
```

Or if your path is different:
```bash
cd /path/to/your/openpl/directory
```

**Tip:** You can also drag and drop the folder into the Terminal window (Mac) or right-click in the folder and select "Open Terminal Here" (Linux).

---

## Step 3: Install Required Packages

Now we'll install all the Python packages needed for this project. These packages include:
- **pandas** - For working with data tables
- **numpy** - For numerical calculations
- **matplotlib** - For creating graphs
- **seaborn** - For better-looking graphs
- **jupyter** - For running notebooks
- **pyarrow** - For reading/writing data files
- **tqdm** - For progress bars

### Windows

```cmd
pip install -r requirements.txt
```

If that doesn't work, try:
```cmd
python -m pip install -r requirements.txt
```

Or if you need to use `py`:
```cmd
py -m pip install -r requirements.txt
```

### Mac/Linux

```bash
pip3 install -r requirements.txt
```

If that doesn't work, try:
```bash
python3 -m pip install -r requirements.txt
```

**What to expect:** This will take a few minutes. You'll see a lot of text scrolling as packages are downloaded and installed. This is normal!

**If you get permission errors on Mac/Linux**, try:
```bash
pip3 install --user -r requirements.txt
```

---

## Step 4: Install Jupyter (if not already installed)

Jupyter should have been installed in Step 3, but let's verify:

### Windows
```cmd
python -m pip install jupyter
```

### Mac/Linux
```bash
pip3 install jupyter
```

---

## Step 5: Start Jupyter Notebook

Now you're ready to run the notebooks!

### Windows

**Option 1: Using Command Prompt**
```cmd
cd eda
python -m jupyter notebook
```

**Option 2: If Option 1 doesn't work**
```cmd
cd eda
py -m jupyter notebook
```

### Mac/Linux

```bash
cd eda
python3 -m jupyter notebook
```

**Or using jupyter directly (if it's in your PATH):**
```bash
cd eda
jupyter notebook
```

**What happens:** Your web browser should automatically open to a page showing the Jupyter notebook interface. If it doesn't open automatically, look for a URL in the terminal that looks like `http://localhost:8888` and copy it into your browser.

---

## Step 6: Running the Notebooks

Once Jupyter is open in your browser:

1. **Click on a notebook** (e.g., `01_data_loading.ipynb`) to open it
2. **Run cells** by clicking on a cell and pressing `Shift + Enter`
   - Or click the "Run" button in the toolbar
3. **Run all cells** by going to: Cell → Run All

### Notebook Execution Order

⚠️ **Note:** Files 1-2 are still in progress. Files 3-7 and insights are also in progress.

When ready, run the notebooks in this order:

1. **01_data_loading.ipynb** - Loads and aggregates all meet data
   - Creates `../data/processed/full_dataset.parquet`
   - ⚠️ This may take 10-20 minutes depending on dataset size
   - ⚠️ **Status: Still being worked on**

2. **02_overview.ipynb** - Global dataset overview
   - Basic statistics, distributions, missing data analysis
   - Comparison of existing scoring systems
   - ⚠️ **Status: Still being worked on**

3. **03_experience_analysis.ipynb** - Experience metrics and relationships
   - Calculates years competing, meet count
   - Analyzes experience vs performance
   - ⚠️ **Status: In progress**

4. **04_age_analysis.ipynb** - Age distributions and relationships
   - Age vs performance by sex and equipment
   - Identifies peak performance ages
   - ⚠️ **Status: In progress**

5. **05_meet_quality.ipynb** - Meet competitiveness analysis
   - Calculates meet quality scores
   - Identifies elite meets and federation prestige
   - ⚠️ **Status: In progress**

6. **06_consistency_analysis.ipynb** - Performance consistency metrics
   - Calculates CV (coefficient of variation)
   - Identifies consistent vs variable lifters
   - ⚠️ **Status: In progress**

7. **07_insights_summary.ipynb** - Synthesizes all findings
   - Loads all saved metrics
   - Generates summary statistics
   - ⚠️ **Status: In progress**

---

## Output Files

All notebooks save outputs to `../data/processed/`:
- `full_dataset.parquet` - Complete aggregated dataset
- `experience_metrics.parquet` - Experience calculations
- `age_metrics.parquet` - Age analysis results
- `meet_quality_metrics.parquet` - Meet quality scores
- `consistency_metrics.parquet` - Consistency metrics
- Various PNG plots for visualizations

---

## Final Deliverable

After running all notebooks, review and update:
- `EDA_INSIGHTS.md` - Comprehensive findings document
  - ⚠️ **Status: In progress**

---

## Common Issues and Troubleshooting

### "python: command not found" or "python is not recognized"

**Windows:**
- Make sure you checked "Add Python to PATH" during installation
- Try using `py` instead of `python`
- Restart your command prompt after installing Python

**Mac/Linux:**
- Use `python3` instead of `python`
- Make sure Python is installed (see installation steps above)

---

### "pip: command not found" or "pip is not recognized"

**Windows:**
- Try: `python -m pip` instead of just `pip`
- Or: `py -m pip`

**Mac/Linux:**
- Use `pip3` instead of `pip`
- Or: `python3 -m pip`

---

### "ModuleNotFoundError" or Import Errors

This means a package isn't installed. Try:
```bash
# Windows
pip install <package-name>
# or
python -m pip install <package-name>

# Mac/Linux
pip3 install <package-name>
# or
python3 -m pip install <package-name>
```

---

### Jupyter Notebook Won't Start

**Windows:**
- Make sure you're in the `eda` directory: `cd eda`
- Try: `python -m jupyter notebook`
- If that fails, try: `py -m jupyter notebook`

**Mac/Linux:**
- Make sure you're in the `eda` directory: `cd eda`
- Try: `python3 -m jupyter notebook`
- If you get a "command not found" error, make sure Jupyter is installed: `pip3 install jupyter`

---

### Browser Doesn't Open Automatically

Look in your terminal for a URL that looks like:
```
http://localhost:8888/?token=abc123...
```

Copy this entire URL and paste it into your web browser.

---

### Path Issues / "File not found" Errors

**Important:** Always run Jupyter from the `eda/` directory, or adjust paths in the notebooks.

**To fix:**
1. Make sure you're in the `eda` directory when starting Jupyter
2. All paths in notebooks are relative to where Jupyter is started

---

### Data Not Found

Make sure the `opl-data/meet-data/` directory exists in the project root. If it doesn't, you may need to download or clone the full dataset.

---

### Memory Issues

If you run out of memory:
- The data loading notebook can sample data if needed
- Close other applications to free up RAM
- Consider running on a subset of federations first

---

### Visualizations Not Showing

- Make sure matplotlib is installed: `pip install matplotlib` (or `pip3` on Mac/Linux)
- Try adding this to a notebook cell: `%matplotlib inline`

---

## Quick Reference Commands

### Windows
```cmd
# Check Python version
python --version
# or
py --version

# Install packages
pip install -r requirements.txt
# or
python -m pip install -r requirements.txt

# Start Jupyter
cd eda
python -m jupyter notebook
```

### Mac/Linux
```bash
# Check Python version
python3 --version

# Install packages
pip3 install -r requirements.txt
# or
python3 -m pip install -r requirements.txt

# Start Jupyter
cd eda
python3 -m jupyter notebook
```

---

## Getting Help

If you're stuck:
1. Check the error message carefully - it often tells you what's wrong
2. Make sure you followed all installation steps
3. Verify you're using the correct commands for your operating system (Windows vs Mac/Linux)
4. Check that you're in the correct directory
5. Try restarting your terminal/command prompt after installing Python

---

## Tips for Success

- **Read error messages carefully** - They usually tell you exactly what's wrong
- **One step at a time** - Don't skip installation steps
- **Use the right commands** - Windows uses `python`, Mac/Linux use `python3`
- **Keep your terminal open** - Don't close the terminal where Jupyter is running
- **Be patient** - Installing packages and running analysis can take time

Good luck! 🚀
