"""
Utility functions for EDA analysis
"""
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


def load_all_meets(base_path="opl-data/meet-data", max_federations=None, chunk_size=100):
    """
    Recursively load all entries.csv and meet.csv files from the meet-data directory.
    Uses chunked processing to avoid memory issues.
    
    Args:
        base_path: Path to meet-data directory
        max_federations: Limit number of federations to load (None = all)
        chunk_size: Number of dataframes to accumulate before concatenating
    
    Returns:
        entries_df: Combined entries dataframe
        meets_df: Combined meets dataframe
    """
    base_path = Path(base_path)
    entries_list = []
    meets_list = []
    
    # Get all federation directories
    fed_dirs = [d for d in base_path.iterdir() if d.is_dir() and not d.name.startswith('.')]
    
    # Limit federations if specified
    if max_federations:
        fed_dirs = fed_dirs[:max_federations]
        print(f"Loading {len(fed_dirs)} federations (limited from {len([d for d in base_path.iterdir() if d.is_dir()])})")
    else:
        print(f"Found {len(fed_dirs)} federation directories")
    
    total_entries = 0
    total_meets = 0
    
    for fed_dir in tqdm(fed_dirs, desc="Loading federations"):
        federation = fed_dir.name
        
        # Get all meet directories
        meet_dirs = [d for d in fed_dir.iterdir() if d.is_dir()]
        
        for meet_dir in meet_dirs:
            entries_file = meet_dir / "entries.csv"
            meet_file = meet_dir / "meet.csv"
            
            # Load entries if exists
            if entries_file.exists():
                try:
                    entries_df = pd.read_csv(entries_file, low_memory=False)
                    # Add metadata
                    entries_df['Federation'] = federation
                    entries_df['MeetPath'] = f"{federation}/{meet_dir.name}"
                    entries_list.append(entries_df)
                    total_entries += len(entries_df)
                    
                    # Chunked concatenation to save memory
                    if len(entries_list) >= chunk_size:
                        entries_list = [pd.concat(entries_list, ignore_index=True)]
                except Exception as e:
                    # Silently skip problematic files
                    continue
            
            # Load meet info if exists
            if meet_file.exists():
                try:
                    meet_df = pd.read_csv(meet_file, low_memory=False)
                    # Add metadata
                    meet_df['MeetPath'] = f"{federation}/{meet_dir.name}"
                    meets_list.append(meet_df)
                    total_meets += 1
                except Exception as e:
                    # Silently skip problematic files
                    continue
    
    # Final combination
    print(f"\nCombining {len(entries_list)} entry chunks...")
    if entries_list:
        entries_combined = pd.concat(entries_list, ignore_index=True)
        print(f"✓ Loaded {len(entries_combined):,} entries from {total_entries:,} total rows")
    else:
        entries_combined = pd.DataFrame()
        print("⚠ No entries found")
    
    print(f"\nCombining {len(meets_list)} meet records...")
    if meets_list:
        meets_combined = pd.concat(meets_list, ignore_index=True)
        print(f"✓ Loaded {len(meets_combined):,} meets")
    else:
        meets_combined = pd.DataFrame()
        print("⚠ No meets found")
    
    return entries_combined, meets_combined


def merge_entries_meets(entries_df, meets_df):
    """
    Merge entries with meet data on MeetPath.
    """
    # Handle empty dataframes
    if entries_df is None or len(entries_df) == 0:
        print("⚠ Warning: entries_df is empty, returning empty dataframe")
        return pd.DataFrame()
    
    if meets_df is None or len(meets_df) == 0:
        print("⚠ Warning: meets_df is empty, returning entries_df without merge")
        return entries_df.copy()
    
    # Standardize MeetPath column
    if 'MeetPath' not in entries_df.columns:
        if 'Federation' in entries_df.columns:
            entries_df['MeetPath'] = entries_df['Federation'] + '/' + entries_df.get('MeetID', '').astype(str)
        else:
            print("⚠ Warning: Cannot create MeetPath - missing Federation column")
            return entries_df.copy()
    
    # Merge
    try:
        merged = entries_df.merge(
            meets_df,
            on='MeetPath',
            how='left',
            suffixes=('', '_meet')
        )
        print(f"✓ Merged dataset: {len(merged):,} rows")
        return merged
    except Exception as e:
        print(f"⚠ Error during merge: {e}")
        print("  Returning entries_df without merge")
        return entries_df.copy()


def clean_data(df):
    """
    Basic data cleaning and type conversion.
    """
    df = df.copy()
    
    # Convert date columns
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    
    if 'Date_meet' in df.columns:
        df['Date_meet'] = pd.to_datetime(df['Date_meet'], errors='coerce')
    
    # Use meet date if available, otherwise try Date column
    if 'Date_meet' in df.columns:
        df['MeetDate'] = df['Date_meet'].fillna(df.get('Date', pd.NaT))
    else:
        df['MeetDate'] = df.get('Date', pd.NaT)
    
    # Convert numeric columns
    numeric_cols = ['TotalKg', 'BodyweightKg', 'Age', 'Wilks', 'Dots', 'Goodlift',
                   'Best3SquatKg', 'Best3BenchKg', 'Best3DeadliftKg']
    
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Handle weight class - create numeric version for analysis
    if 'WeightClassKg' in df.columns:
        # Create numeric version for analysis (keep original as object/string)
        df['WeightClassKg_numeric'] = df['WeightClassKg'].astype(str).str.replace('+', '').str.replace('kg', '').str.strip()
        df['WeightClassKg_numeric'] = pd.to_numeric(df['WeightClassKg_numeric'], errors='coerce')
    
    # Note: We don't convert all object columns to string here because:
    # 1. It's memory intensive for large datasets
    # 2. Some columns might be better kept as object for analysis
    # 3. We'll convert only when saving to parquet (in the notebook)
    
    return df


def categorize_ipf_weightclass(bodyweight_kg, sex):
    """
    Categorize bodyweight into IPF standard weight classes based on gender.
    
    IPF Weight Classes:
    - Women: 47kg, 52kg, 57kg, 63kg, 69kg, 76kg, 84kg, 84+kg
    - Men: 59kg, 66kg, 74kg, 83kg, 93kg, 105kg, 120kg, 120+kg
    
    Args:
        bodyweight_kg: Bodyweight in kilograms (float or NaN)
        sex: Gender ('M' for male, 'F' for female, or other)
    
    Returns:
        str: IPF weight class (e.g., "47kg", "84+kg", "59kg", "120+kg") or None if invalid
    """
    # Handle missing values
    if pd.isna(bodyweight_kg) or bodyweight_kg is None:
        return None
    
    # Convert to float if needed
    try:
        bw = float(bodyweight_kg)
    except (ValueError, TypeError):
        return None
    
    # IPF Women's weight classes: 47kg, 52kg, 57kg, 63kg, 69kg, 76kg, 84kg, 84+kg
    if sex == 'F':
        if bw < 47:
            return '47kg'  # Assign to lowest class if below minimum
        elif bw < 52:
            return '47kg'
        elif bw < 57:
            return '52kg'
        elif bw < 63:
            return '57kg'
        elif bw < 69:
            return '63kg'
        elif bw < 76:
            return '69kg'
        elif bw < 84:
            return '76kg'
        else:
            return '84+kg'
    
    # IPF Men's weight classes: 59kg, 66kg, 74kg, 83kg, 93kg, 105kg, 120kg, 120+kg
    elif sex == 'M':
        if bw < 59:
            return '59kg'  # Assign to lowest class if below minimum
        elif bw < 66:
            return '59kg'
        elif bw < 74:
            return '66kg'
        elif bw < 83:
            return '74kg'
        elif bw < 93:
            return '83kg'
        elif bw < 105:
            return '93kg'
        elif bw < 120:
            return '105kg'
        else:
            return '120+kg'
    
    # For other genders (Mx, etc.), return None
    else:
        return None


def calculate_retirement_status(df):
    """
    Calculate retirement status for each lifter based on last meet date.
    
    A lifter is considered "Retired" if they haven't competed in 4+ years.
    
    Args:
        df: DataFrame with Name and MeetDate columns
        
    Returns:
        DataFrame with 'Retired' column added
    """
    df = df.copy()
    
    # Calculate last meet date per lifter
    lifter_last_meet = df.groupby('Name')['MeetDate'].max().reset_index()
    lifter_last_meet.columns = ['Name', 'LastMeetDate']
    
    # Merge back
    df = df.merge(lifter_last_meet, on='Name', how='left')
    
    # Calculate years since last meet (using most recent date in dataset)
    most_recent_date = df['MeetDate'].max()
    df['YearsSinceLastMeet'] = (most_recent_date - df['LastMeetDate']).dt.days / 365.25
    
    # Flag retired (4+ years since last meet)
    df['Retired'] = (df['YearsSinceLastMeet'] >= 4) & (df['YearsSinceLastMeet'].notna())
    
    return df


def create_quality_filter(df):
    """
    Create a comprehensive quality filter mask to exclude outliers and invalid data.
    
    Filters out:
    - Failed meets (TotalKg < 50kg)
    - Extreme totals (>2000kg)
    - Extreme bodyweights (<30kg or >200kg)
    - Retired lifters (4+ years since last meet)
    - Missing critical data (Name, Sex, IPF_WeightClass, TotalKg, Division)
    
    NOTE: Age filtering has been removed since Age only has 24.29% coverage.
    We now use Division (100% coverage) for age-based analysis.
    
    Args:
        df: DataFrame with required columns
        
    Returns:
        Boolean mask (True = keep, False = filter out)
    """
    # Ensure retirement status is calculated
    if 'Retired' not in df.columns:
        df = calculate_retirement_status(df)
    
    # Ensure FailedMeet flag exists
    if 'FailedMeet' not in df.columns:
        df['FailedMeet'] = (df['TotalKg'] < 50) & (df['TotalKg'].notna())
    
    # Create quality mask (Age filtering removed - using Division instead)
    quality_mask = (
        ~df['FailedMeet'] &  # Not bombed out
        (df['TotalKg'] <= 2000) &  # Not extreme high total
        (df['TotalKg'] >= 50) &  # Not extreme low total (explicit check)
        (df['BodyweightKg'] >= 30) & (df['BodyweightKg'] <= 200) &  # Normal bodyweight
        ~df['Retired'] &  # Not retired
        df['Name'].notna() &  # Has name
        df['Sex'].isin(['M', 'F']) &  # Valid gender
        df['IPF_WeightClass'].notna() &  # Has weight class
        df['TotalKg'].notna() & (df['TotalKg'] > 0) &  # Has valid total
        df['Division'].notna()  # Has division (age class) - 100% coverage
    )
    
    return quality_mask


def map_division_to_age_group(division):
    """
    Map Division (age class) values to standard powerlifting age groups.
    
    Since Division has 100% coverage while Age only has 24.29%, we use Division
    to categorize lifters into standard age groups.
    
    Standard Age Groups:
    - Youth: <14
    - Teen: 14-16
    - Sub-Junior: 17-19 years old
    - Junior: 20-23 years old
    - Open: 24-39
    - Masters I: 40-49 years old
    - Masters II: 50-59 years old
    - Masters III: 60-69 years old
    - Masters IV: 70 years and above
    
    Args:
        division: Division string value (e.g., "Open", "Masters 1", "Juniors", etc.)
    
    Returns:
        str: Standardized age group category or "Open" as default
    """
    # Handle missing values
    if pd.isna(division) or division is None:
        return 'Open'  # Default to Open for missing divisions
    
    # Convert to string and normalize
    div_str = str(division).strip()
    div_lower = div_str.lower()
    
    # Masters IV (70+)
    if any(x in div_lower for x in ['masters 4', 'm4', 'masters 70', 'masters iv', 'over 70', '70+']):
        return 'Masters IV'
    
    # Masters III (60-69)
    if any(x in div_lower for x in ['masters 3', 'm3', 'masters 60', 'masters 65', 'masters iii', 
                                     'masters 60-64', 'masters 65-69', '60-69', '65-69']):
        return 'Masters III'
    
    # Masters II (50-59)
    if any(x in div_lower for x in ['masters 2', 'm2', 'masters 50', 'masters 55', 'masters ii',
                                     'masters 50-54', 'masters 55-59', 'masters 50-59', '50-59', 
                                     'over 50', '50+']):
        return 'Masters II'
    
    # Masters I (40-49)
    if any(x in div_lower for x in ['masters 1', 'm1', 'masters 40', 'masters 45', 'masters i',
                                     'masters 40-44', 'masters 45-49', 'masters 40-49', '40-49',
                                     'over 40', '40+', 'submasters 35-39', 'submasters 33-39',
                                     'submasters']):
        # Note: Submasters 33-39 and 35-39 are typically Masters I
        return 'Masters I'
    
    # Youth/Teen/Boys/Girls - Check FIRST to avoid false matches with Open/Junior
    if any(x in div_lower for x in ['youth', 'boys', 'girls', 'junior varsity']):
        # Boys and Girls are typically youth/teen - defaulting to Teen for now
        # Could be refined based on specific federation rules
        return 'Teen'
    
    # Teen (14-16) - Check BEFORE Sub-Junior
    if any(x in div_lower for x in ['teen', 'teen 14-18', 'teen 16-17', 'teen 18-19',
                                     'amateur teen 16-17', 'amateur teen 18-19']):
        return 'Teen'
    
    # Sub-Junior (17-19) - Check BEFORE Junior
    if any(x in div_lower for x in ['sub-juniors', 'sub-junior', 'submasters', 'juniors 18-19',
                                     'juniors 16-17', 'juniors 13-15', 'mr-sj', 't3', 't2', 't1', 'mr-t3', 'mr-t2',
                                     'mr-t1', 'fr-t3', 'm-t3', 'm-t2']):
        return 'Sub-Junior'
    
    # Junior (20-23) - Check BEFORE Open to avoid false matches
    if any(x in div_lower for x in ['juniors 20-23', 'juniors 19-23', 'j20-23', 'junior',
                                     'juniors', 'jr', 'mr-jr', 'fr-jr', 'm-jr', 'amateur juniors 20-23',
                                     'pro juniors 20-23']):
        return 'Junior'
    
    # Open (24-39) - Check AFTER Junior to avoid false matches
    if any(x in div_lower for x in ['open', 'o', 'mr-o', 'fr-o', 'm-o', 'amateur open', 'pro open',
                                     'f-o', 'm-or', 'm-or_apf', 'm_or_wpc', 'mor', 'm-c-open',
                                     'f-c-open', 'senior', 'seniors', 'seniorzy', 'snr', 'class 1',
                                     'novice', 'varsity', 'hs', 'under 23']):
        return 'Open'
    
    # Default to Open for unknown divisions
    return 'Open'


def categorize_age_group(division):
    """
    Categorize Division into standard powerlifting age groups.
    
    This function is now an alias for map_division_to_age_group to maintain
    backward compatibility while using Division instead of Age.
    
    Args:
        division: Division string value
    
    Returns:
        str: Standardized age group category
    """
    return map_division_to_age_group(division)


def categorize_lifters(df):
    """
    Categorize lifters into New, Intermediate, or Advanced based on experience and performance.
    
    Categorization is priority-based (no duplicates):
    - Advanced (highest priority): Top 20% of highest totals for their (Sex × IPF_WeightClass × AgeGroup)
    - New (second priority): 1 meet completed AND not Advanced
    - Intermediate (default): 2+ meets completed AND not Advanced
    
    IMPORTANT: Categorization is done PER-LIFTER, then merged back to all entries.
    This ensures all entries for the same lifter have the same category.
    
    NOTE: Now uses Division (100% coverage) instead of Age (24.29% coverage) for age group categorization.
    
    Args:
        df: DataFrame with Name, TotalKg, MeetDate, Sex, IPF_WeightClass, Division columns
        
    Returns:
        DataFrame with 'LifterCategory' and 'IsAdvanced' columns added
    """
    df = df.copy()
    
    # 1. Get lifter-level metrics (one row per lifter)
    lifter_metrics = df.groupby('Name').agg({
        'TotalKg': 'max',  # Best total
        'MeetDate': ['min', 'max', 'count'],  # First meet, last meet, total meets
        'Sex': 'first',  # Assume consistent
        'IPF_WeightClass': lambda x: x.mode()[0] if len(x.mode()) > 0 else None,  # Most common
        'Division': lambda x: x.mode()[0] if len(x.mode()) > 0 else None  # Most common division
    }).reset_index()
    
    # Flatten column names
    lifter_metrics.columns = ['Name', 'BestTotal', 'FirstMeetDate', 'LastMeetDate', 'TotalMeets', 
                             'Sex', 'IPF_WeightClass', 'MostCommonDivision']
    
    # 2. Determine age group for each lifter using Division (100% coverage)
    # Map most common Division to standard age group
    lifter_metrics['AgeGroup'] = lifter_metrics['MostCommonDivision'].apply(map_division_to_age_group)
    age_group_col = 'AgeGroup'
    
    # Filter out lifters with missing critical data for advanced classification
    valid_for_advanced = (
        lifter_metrics['BestTotal'].notna() &
        lifter_metrics['Sex'].isin(['M', 'F']) &
        lifter_metrics['IPF_WeightClass'].notna() &
        lifter_metrics[age_group_col].notna()
    )
    
    # 3. Calculate 80th percentile (top 20%) per (Sex × IPF_WeightClass × AgeGroup)
    # Only for valid lifters
    valid_metrics = lifter_metrics[valid_for_advanced].copy()
    
    if len(valid_metrics) > 0:
        top_20_thresholds = valid_metrics.groupby(['Sex', 'IPF_WeightClass', age_group_col])['BestTotal'].quantile(0.80).reset_index()
        top_20_thresholds.columns = ['Sex', 'IPF_WeightClass', age_group_col, 'Top20Threshold']
        
        # 4. Merge thresholds and flag advanced (per lifter)
        lifter_metrics = lifter_metrics.merge(
            top_20_thresholds, 
            on=['Sex', 'IPF_WeightClass', age_group_col], 
            how='left'
        )
        lifter_metrics['IsAdvanced'] = (
            lifter_metrics['BestTotal'] >= lifter_metrics['Top20Threshold']
        ) & lifter_metrics['BestTotal'].notna()
    else:
        lifter_metrics['IsAdvanced'] = False
    
    # 5. Categorize lifters (priority-based, per lifter)
    def categorize_lifter(row):
        # First priority: Advanced (regardless of meet count)
        if row.get('IsAdvanced', False):
            return 'Advanced'
        # Second priority: New (1 meet, but not Advanced)
        elif row['TotalMeets'] == 1:
            return 'New'
        # Third priority: Intermediate (2+ meets, but not Advanced)
        elif row['TotalMeets'] >= 2:
            return 'Intermediate'
        else:
            return 'Unknown'
    
    lifter_metrics['LifterCategory'] = lifter_metrics.apply(categorize_lifter, axis=1)
    
    # 6. Merge category back to all entries (all entries for a lifter get same category)
    df = df.merge(
        lifter_metrics[['Name', 'LifterCategory', 'IsAdvanced']], 
        on='Name', 
        how='left'
    )
    
    return df


def categorize_federation_testing_status(df):
    """
    Categorize federations as 'Drug Tested' (fully tested OR mixed) vs 'Untested'.
    
    Logic:
    1. Federations in the fully tested list (from is_fully_tested) are Drug Tested
    2. Federations with ANY entries where Tested='Yes' are considered Mixed -> Drug Tested
    3. All other federations are Untested
    
    Args:
        df: DataFrame with 'Federation', 'Tested', and 'Date' columns
        
    Returns:
        DataFrame with 'FederationTestingStatus' column added
    """
    df = df.copy()
    
    # Fully tested federations (from is_fully_tested function in Rust code)
    # Extracted from opl-data/crates/opltypes/src/federation.rs
    FULLY_TESTED_FEDERATIONS = {
        'AAPLF', 'ADAU', 'ADFPA', 'ADFPF', 'AEP', 'AfricanPF', 'AIWBPA', 'AMP',
        'APLA', 'APPortugal', 'APU', 'APUA', 'AsianPF', 'AusDFPF', 'BahamasPF',
        'BAWLA', 'BDFPA', 'BDFPF', 'BelPF', 'BP', 'BPA', 'BulgarianPF', 'BVDG',
        'BVDK', 'CBLB', 'CHNPL', 'CNFA', 'ColPF', 'CommonwealthPF', 'CPU',
        'CSST', 'CTPA', 'CyprusPF', 'DFPFNL', 'DSF', 'EgyptPF', 'EJTL', 'EPA',
        'EPF', 'FALPO', 'FAPL', 'FCLP', 'FCP', 'FDNLP', 'FECAPOLIF', 'FECHIPO',
        'Fedepotencia', 'FEFICULP', 'FELIPOME', 'FEMEPO', 'FESUPO', 'FEVEPO',
        'FFForce', 'FFHMFAC', 'FHSAA', 'FIAP', 'FIPL', 'FMPB', 'FPP', 'FPPR',
        'FPR', 'FRPL', 'FSFA', 'FULP', 'GAPLF', 'GDFPF', 'HKPF', 'HKWPA', 'HPF',
        'HPLS', 'HTPL', 'Hunpower', 'IBSA', 'IDFPA', 'IDFPF', 'IHSPLA', 'IPF',
        'IPFChina', 'IranBBF', 'IraqPF', 'IrishPF', 'IronBoy', 'JPA', 'KBGV',
        'KDKS', 'KNKFSP', 'KPF', 'KRAFT', 'KPC', 'LebanonPF', 'LFPH', 'LibyaPF',
        'LJTF', 'LPF', 'MaltaPA', 'ManxPL', 'MAP', 'MDFPA', 'MDFPF', 'MUPF',
        'NAPF', 'NASA', 'NaturalPA', 'NauruPF', 'NIPF', 'NORCAL', 'NordicPF',
        'NPAJ', 'NPB', 'NSF', 'NYFC', 'NZPF', 'NZAWLA', 'OceaniaPF', 'OCWP',
        'ORPF', 'OEVK', 'PA', 'PAP', 'PFBD', 'PI', 'PLRD', 'PLSS', 'PLZS',
        'PNGPF', 'POSK', 'PS', 'PWFL', 'PZKFiTS', 'QatarPL', 'RAW', 'RAWCAN',
        'RAWUKR', 'RDFPF', 'SADFPA', 'SAFKST', 'SAFP', 'SAPF', 'ScottishPL',
        'SDFPF', 'SLPF', 'SSAU', 'SSF', 'SSSC', 'SVNL', 'TAAP', 'ThaiPF',
        'THSPA', 'THSWPA', 'TPSSF', 'TTPF', 'UAEPL', 'UDFPF', 'UgandaPF',
        'UkrainePF', 'USABA', 'USAPL', 'USVIPF', 'VDFPA', 'VGPF', 'VPF',
        'WABDL', 'WDFPF', 'WelshPA', 'WNPF', 'WP', 'WPChina', 'WPIndia',
        'WPNauru', 'WPNiue', 'WPLanka', 'WPUSA', 'WPNZ', 'WPPO'
    }
    
    # Date-dependent federations
    # AAU: tested after 1995
    # ILPF: tested after 2023
    # RawIronPL: tested before 2020-06-20
    DATE_DEPENDENT_FULLY_TESTED = {
        'AAU': {'after_year': 1995},
        'ILPF': {'after_year': 2023},
        'RawIronPL': {'before_date': '2020-06-20'},
    }
    
    # Convert Tested to boolean
    df['IsTested'] = (df['Tested'] == 'Yes') | (df['Tested'] == 'yes')
    
    # Check if federation has any tested entries (mixed)
    fed_has_tested = df.groupby('Federation')['IsTested'].any()
    
    # Categorize each entry
    def get_testing_status(row):
        federation = row['Federation']
        if pd.isna(federation):
            return 'Unknown'
        
        # Check date-dependent cases
        if federation in DATE_DEPENDENT_FULLY_TESTED:
            rules = DATE_DEPENDENT_FULLY_TESTED[federation]
            # Try Date column first, then MeetDate
            date_col = 'Date' if 'Date' in row.index else 'MeetDate'
            if date_col in row.index and pd.notna(row[date_col]):
                try:
                    meet_date = pd.to_datetime(row[date_col])
                    
                    if 'after_year' in rules:
                        if meet_date.year >= rules['after_year']:
                            return 'Drug Tested'
                    elif 'before_date' in rules:
                        from datetime import datetime
                        cutoff = datetime.strptime(rules['before_date'], '%Y-%m-%d')
                        if meet_date < cutoff:
                            return 'Drug Tested'
                except:
                    pass
        
        # Check if in fully tested list
        if federation in FULLY_TESTED_FEDERATIONS:
            return 'Drug Tested'
        
        # Check if federation has any tested entries (mixed)
        if fed_has_tested.get(federation, False):
            return 'Drug Tested'
        
        # Otherwise untested
        return 'Untested'
    
    df['FederationTestingStatus'] = df.apply(get_testing_status, axis=1)
    
    return df


