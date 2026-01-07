# OpenPowerlifting EDA

Exploratory data analysis of the OpenPowerlifting dataset (2015-2025).

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Start Jupyter Notebook
cd eda
jupyter notebook
# Or: python -m jupyter notebook
# Or: python3 -m jupyter notebook (Mac/Linux)
```

### Running the Notebooks

1. **Start Jupyter**: Run `jupyter notebook` from the `eda/` directory
2. **Open notebook**: Click on `01_data_loading.ipynb` in the browser
3. **Run cells**: 
   - Press `Shift + Enter` to run a cell
   - Or use `Cell → Run All` to run entire notebook
4. **Run in order**:
   - First: `01_data_loading.ipynb` (creates `full_dataset.parquet`)
   - Then: `02_overview.ipynb` (generates analysis and visualizations)

## Project Structure

- **01_data_loading.ipynb** - Loads and processes meet data from CSV files → `full_dataset.parquet`
- **02_overview.ipynb** - Dataset overview, distributions, quality analysis, visualizations
- **utils.py** - Shared utility functions (data loading, cleaning, categorization)

## What It Does



### Key Features

     
- **IPF Weight Classes**: Automatic categorization (47kg-84+kg women, 59kg-120+kg men)
- **Age Groups**: Maps Division to standard groups (Youth, Teen, Junior, Open, Masters I-IV)  -- should remove youth//masters 2-4 for better EDA
- **Federation Testing**: Categorizes as "Drug Tested" or "Untested"
   gender
   ruleset -- raw vs raww with wraps vs equipped vs multiply -- for now lets only do raw and equipped 
   type of meet- full SBD, single lift only, multiply bench -- remove all multiply bench its fake benching


Current structure(needs to be changed to only testing the tested column from dataset)
  - Always tested: IPL, EPF, EPA, THSWPA, THSPA, USAPL
  - Others: Uses `Tested` column value

- also needs changing **Failed Meet Detection**: Identifies bombed out lifters (all attempts missing in any lift group)\
needs to seperate the features -- full power meets with SBD vs squat only, bench only, deadlift only meets. 
bombed out vs one lift only meet needs distinction -- 




## Output Files

All saved to `../data/processed/`:


## Utility Functions (`utils.py`)  definetly not perfect should check before using 

- `load_all_meets()` - Recursive CSV loading with date filtering
- `merge_entries_meets()` - Merges entries with meet metadata
- `clean_data()` - Standardizes and cleans data
- `categorize_ipf_weightclass()` - Maps bodyweight to IPF classes
- `map_division_to_age_group()` - Converts Division to age groups
- `calculate_retirement_status()` - Identifies retired lifters
- `create_quality_filter()` - Creates quality filter mask
- `categorize_lifters()` - Classifies by experience (New/Intermediate/Advanced)
- `categorize_federation_testing_status()` - Categorizes tested/untested

## Design Decisions

- **Individual Visualizations**: Each combination gets its own plot (prevents data skewing)
- **2015-2025 Filter**: More recent, consistent data (adjustable via `date_range` parameter)
- **Multi-dimensional Separation**: All analysis split by Gender × Weight Class × Age Group × Testing Status

## Requirements

- Python 3.8+
- pandas, numpy, matplotlib, seaborn, jupyter, pyarrow, tqdm

See `requirements.txt` for versions.
