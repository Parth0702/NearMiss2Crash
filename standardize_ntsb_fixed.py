#!/usr/bin/env python3
"""Read NTSB AviationData.csv with proper encoding handling and standardize schema."""

import pandas as pd
import sys
import os

# File path
filepath = r'D:\SC project\ntsb dataset\AviationData.csv'

# Read with cp1252 encoding (Windows ANSI) since the file has special chars like 0x92
print(f"Loading NTSB data from: {filepath}")
print("Using cp1252 encoding to handle Windows-1252 special characters...")

try:
    df = pd.read_csv(filepath, encoding='cp1252', low_memory=False, on_bad_lines='skip', nrows=10000)
    print(f"✓ Successfully loaded {len(df)} rows")
    print(f"Columns: {list(df.columns)[:20]}")
except Exception as e:
    print(f"✗ Failed with cp1252: {e}")
    # Try latin-1 as fallback
    try:
        df = pd.read_csv(filepath, encoding='latin-1', low_memory=False, on_bad_lines='skip')
        print(f"✓ Loaded with latin-1 fallback: {len(df)} rows")
    except Exception as e2:
        print(f"✗ latin-1 also failed: {e2}")
        sys.exit(1)

# Strip whitespace from column names
df.columns = df.columns.str.strip()

# Print available columns
print(f"\nAvailable columns ({len(df.columns)}):")
for i, col in enumerate(df.columns):
    print(f"  {i}: {col}")

# Create narrative text from available columns
# NTSB AviationData.csv doesn't have a dedicated 'text' or 'narrative' column
# Instead, combine descriptive fields
text_columns_to_try = [
    'Event.Id', 'Location', 'Country', 'Weather.Condition',
    'Broad.phase.of.flight', 'FAR.Description', 'Schedule',
    'Report.Status', 'Publication.Date', 'Accident.Number',
    'Investigation.Type', 'Aircraft.Category', 'Aircraft.Make',
    'Aircraft.Model', 'Injury.Severity', 'Fatalities',
    'Damage', 'Injuries', 'Sunrise.Sunset', 'Flight.Rules',
    'Airport.ID', 'Airport.Name', 'City', 'State'
]

text_parts = []
for col in text_columns_to_try:
    if col in df.columns:
        # Convert to string, replace NaN with empty, strip whitespace
        series = df[col].fillna('').astype(str).str.strip()
        # Only add if there's non-empty content
        non_empty = series[series != '']
        if len(non_empty) > 0:
            text_parts.append(series)
            print(f"  Added column '{col}': {len(non_empty)} non-empty values")

# Combine all text parts
if text_parts:
    combined_text = text_parts[0]
    for part in text_parts[1:]:
        combined_text = combined_text + ' ' + part
else:
    # Fallback: create text from Event.Id
    combined_text = df.get('Event.Id', pd.Series(['NTSB_event_' + str(i) for i in range(len(df))])).astype(str)

# Create standardized output DataFrame
result_df = pd.DataFrame({
    'report_id': df.get('Event.Id', [f"NTSB_{i}" for i in range(len(df))]).astype(str),
    'source': 'NTSB',
    'text': combined_text,
    'label': df.get('Investigation.Type', ['Unlabeled'] * len(df)).fillna('Unlabeled').astype(str)
})

# Drop rows with empty text
before = len(result_df)
result_df = result_df[result_df['text'].str.strip() != ''].reset_index(drop=True)
after = len(result_df)
print(f"\nDropped {before - after} rows with empty text")
print(f"Final dataset: {after} samples")

# Print label distribution
if 'label' in result_df.columns:
    print(f"\nLabel distribution:")
    print(result_df['label'].value_counts().to_string())

# Save standardized data
output_path = r'D:\SC project\ntsb_standardized.csv'
result_df.to_csv(output_path, index=False, encoding='utf-8')
print(f"\n✓ Saved standardized data to: {output_path}")

# Print first few rows
print(f"\nFirst 3 rows:")
for i in range(min(3, len(result_df))):
    print(f"  Row {i}: report_id={result_df['report_id'].iloc[i][:20]}...")
    print(f"           text={str(result_df['text'].iloc[i])[:150]}...")
    print(f"           label={result_df['label'].iloc[i]}")
    print()

print("\nDone! The NTSB data is now standardized and ready for model testing.")