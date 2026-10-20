#!/usr/bin/env python3
"""Detect and handle NTSB CSV encoding issues."""

import os

filepath = r'D:\SC project\ntsb dataset\AviationData.csv'

# Read the raw bytes
with open(filepath, 'rb') as f:
    raw = f.read()

print(f"File size: {len(raw)} bytes")
print(f"First 20 bytes: {raw[:20]}")

# Try different encodings
encodings_to_try = [
    ('utf-8', True),
    ('cp1252', False),
    ('latin-1', False),
    ('utf-16', False),
    ('utf-16-le', False),
    ('utf-16-be', False),
]

print("\nTrying encodings:")
for enc, remove_bom in encodings_to_try:
    try:
        if remove_bom and raw[:3] == b'\xef\xbb\xbf':
            raw_trimmed = raw[3:]
        else:
            raw_trimmed = raw
        text = raw_trimmed.decode(enc)
        print(f"  {enc}: SUCCESS - first 100 chars: {repr(text[:100])}")
        # Write test file
        with open(f'test_{enc}.csv', 'w', encoding=enc) as f:
            f.write(text)
        print(f"    -> Saved to test_{enc}.csv")
    except Exception as e:
        print(f"  {enc}: FAIL - {str(e)[:50]}")

# Also try with pandas error handling
print("\nTrying pandas with error handling:")
import pandas as pd
for encoding in ['utf-8', 'cp1252', 'latin1']:
    try:
        df = pd.read_csv(filepath, encoding=encoding, on_bad_lines='skip', engine='python')
        print(f"  {encoding}: SUCCESS - {len(df)} rows, columns: {list(df.columns)[:10]}")
        df.to_csv(f'test_pandas_{encoding}.csv', index=False)
        print(f"    -> Saved to test_pandas_{encoding}.csv")
        break
    except Exception as e:
        print(f"  {encoding}: {str(e)[:60]}")

print("\nDone! Check the test_*.csv files for which encoding works best.")