"""
NEMOSIS Data Availability Debug Script
Standalone script to test data download and availability
"""

import pandas as pd
import nemosis
from datetime import datetime, timedelta
import os


def test_nemosis_download():
    """Test NEMOSIS data download with various date ranges."""

    # Configuration - match your project settings
    raw_data_cache = r"C:\Users\Anderson\Machine Learning\NEM\NEMOSIS_raw_data_cache"
    region_id = "NSW1"
    table_name = "DISPATCHPRICE"
    columns = ['REGIONID', 'SETTLEMENTDATE', 'RRP', 'INTERVENTION']

    print("=" * 60)
    print("NEMOSIS DATA AVAILABILITY DEBUG")
    print("=" * 60)
    print(f"Target region: {region_id}")
    print(f"Cache directory: {raw_data_cache}")
    print(f"Table: {table_name}")
    print()

    # Ensure cache directory exists
    os.makedirs(raw_data_cache, exist_ok=True)

    # Test different date ranges
    test_cases = [
        {
            'name': 'Current Week',
            'start': '2025/06/23 00:00:00',
            'end': '2025/06/29 23:55:00'
        },
        {
            'name': 'Last Week of June 2025',
            'start': '2025/06/24 00:00:00',
            'end': '2025/06/30 23:55:00'
        },
        {
            'name': 'Single Day - June 30',
            'start': '2025/06/30 00:00:00',
            'end': '2025/06/30 23:55:00'
        },
        {
            'name': 'Single Day - June 28',
            'start': '2025/06/28 00:00:00',
            'end': '2025/06/28 23:55:00'
        },
        {
            'name': 'Early June 2025',
            'start': '2025/06/01 00:00:00',
            'end': '2025/06/07 23:55:00'
        },
        {
            'name': 'May 2025 Sample',
            'start': '2025/05/01 00:00:00',
            'end': '2025/05/07 23:55:00'
        },
        {
            'name': 'April 2025 Sample',
            'start': '2025/04/01 00:00:00',
            'end': '2025/04/07 23:55:00'
        }
    ]

    results = []

    for test_case in test_cases:
        print(f"Testing: {test_case['name']}")
        print(f"  Date range: {test_case['start']} to {test_case['end']}")

        try:
            # Download data using NEMOSIS
            data = nemosis.dynamic_data_compiler(
                test_case['start'],
                test_case['end'],
                table_name,
                raw_data_cache,
                filter_cols=['REGIONID', 'INTERVENTION'],
                filter_values=([region_id], [0]),  # Non-intervention only
                select_columns=columns,
                fformat='parquet',
                keep_csv=False
            )

            if data is not None and not data.empty:
                # Remove INTERVENTION column
                data = data.drop(columns=['INTERVENTION'])
                data['SETTLEMENTDATE'] = pd.to_datetime(data['SETTLEMENTDATE'])
                data = data.sort_values('SETTLEMENTDATE')

                actual_start = data['SETTLEMENTDATE'].min()
                actual_end = data['SETTLEMENTDATE'].max()
                unique_dates = sorted(data['SETTLEMENTDATE'].dt.date.unique())

                result = {
                    'test_name': test_case['name'],
                    'requested_start': test_case['start'],
                    'requested_end': test_case['end'],
                    'success': True,
                    'record_count': len(data),
                    'actual_start': actual_start,
                    'actual_end': actual_end,
                    'unique_dates_count': len(unique_dates),
                    'date_range_days': (actual_end.date() - actual_start.date()).days + 1,
                    'first_date': unique_dates[0] if unique_dates else None,
                    'last_date': unique_dates[-1] if unique_dates else None,
                    'sample_rrp_stats': {
                        'min': data['RRP'].min(),
                        'max': data['RRP'].max(),
                        'mean': data['RRP'].mean(),
                        'count': len(data)
                    }
                }

                print(f"  ✅ SUCCESS: {len(data):,} records")
                print(f"     Actual range: {actual_start.date()} to {actual_end.date()}")
                print(f"     Unique dates: {len(unique_dates)}")
                print(f"     RRP range: ${data['RRP'].min():.2f} to ${data['RRP'].max():.2f}")

                # Show first few records
                print(f"     First few records:")
                for i, row in data.head(3).iterrows():
                    print(f"       {row['SETTLEMENTDATE']} | RRP: ${row['RRP']:.2f}")

            else:
                result = {
                    'test_name': test_case['name'],
                    'requested_start': test_case['start'],
                    'requested_end': test_case['end'],
                    'success': False,
                    'error': 'No data returned',
                    'record_count': 0
                }
                print(f"  ❌ FAILED: No data returned")

        except Exception as e:
            result = {
                'test_name': test_case['name'],
                'requested_start': test_case['start'],
                'requested_end': test_case['end'],
                'success': False,
                'error': str(e),
                'record_count': 0
            }
            print(f"  ❌ FAILED: {str(e)}")

        results.append(result)
        print()

    return results


def test_specific_project_dates():
    """Test the exact dates used in your project."""

    print("=" * 60)
    print("TESTING PROJECT-SPECIFIC DATES")
    print("=" * 60)

    # Your project dates
    raw_data_cache = r"C:\Users\Anderson\Machine Learning\NEM\NEMOSIS_raw_data_cache"

    project_config = {
        'start_date': "2025/04/01 00:00:00",
        'end_date': "2025/06/30 23:55:00",
        'analysis_start_date': "2025/05/01 00:00:00"
    }

    print(f"Project configuration:")
    for key, value in project_config.items():
        print(f"  {key}: {value}")
    print()

    try:
        print("Attempting full project date range download...")

        data = nemosis.dynamic_data_compiler(
            project_config['start_date'],
            project_config['end_date'],
            "DISPATCHPRICE",
            raw_data_cache,
            filter_cols=['REGIONID', 'INTERVENTION'],
            filter_values=(['NSW1'], [0]),
            select_columns=['REGIONID', 'SETTLEMENTDATE', 'RRP', 'INTERVENTION'],
            fformat='parquet',
            keep_csv=False
        )

        if data is not None and not data.empty:
            data = data.drop(columns=['INTERVENTION'])
            data['SETTLEMENTDATE'] = pd.to_datetime(data['SETTLEMENTDATE'])
            data = data.sort_values('SETTLEMENTDATE')

            print(f"✅ SUCCESS: Downloaded {len(data):,} records")
            print(f"   Date range: {data['SETTLEMENTDATE'].min()} to {data['SETTLEMENTDATE'].max()}")

            # Check data for specific problem dates
            problem_dates = ['2025-06-28', '2025-06-29', '2025-06-30']
            print(f"\n   Checking problem dates:")

            for date_str in problem_dates:
                date_data = data[data['SETTLEMENTDATE'].dt.date == pd.to_datetime(date_str).date()]
                print(f"     {date_str}: {len(date_data)} records")

                if len(date_data) > 0:
                    print(
                        f"       Time range: {date_data['SETTLEMENTDATE'].min().time()} to {date_data['SETTLEMENTDATE'].max().time()}")
                    print(f"       RRP range: ${date_data['RRP'].min():.2f} to ${date_data['RRP'].max():.2f}")

            # Check analysis period specifically
            analysis_start = pd.to_datetime(project_config['analysis_start_date'])
            analysis_end = pd.to_datetime(project_config['end_date'])

            analysis_data = data[
                (data['SETTLEMENTDATE'] >= analysis_start) &
                (data['SETTLEMENTDATE'] <= analysis_end)
                ]

            print(f"\n   Analysis period data (May 1 - June 30):")
            print(f"     Records: {len(analysis_data):,}")
            print(f"     Unique dates: {analysis_data['SETTLEMENTDATE'].dt.date.nunique()}")

            return data

        else:
            print("❌ FAILED: No data returned for project date range")
            return None

    except Exception as e:
        print(f"❌ FAILED: {str(e)}")
        return None


def generate_summary_report(results, project_data):
    """Generate a summary report of findings."""

    print("=" * 60)
    print("SUMMARY REPORT")
    print("=" * 60)

    successful_tests = [r for r in results if r['success']]
    failed_tests = [r for r in results if not r['success']]

    print(f"Test Results:")
    print(f"  Total tests: {len(results)}")
    print(f"  Successful: {len(successful_tests)}")
    print(f"  Failed: {len(failed_tests)}")
    print()

    if successful_tests:
        print("Successful Downloads:")
        for test in successful_tests:
            print(f"  ✅ {test['test_name']}: {test['record_count']:,} records")
        print()

    if failed_tests:
        print("Failed Downloads:")
        for test in failed_tests:
            print(f"  ❌ {test['test_name']}: {test.get('error', 'Unknown error')}")
        print()

    if project_data is not None:
        print("✅ PROJECT DATA IS AVAILABLE")
        print("   The issue is likely in your project code, not data availability.")
        print("   Suggested areas to check:")
        print("   1. Date filtering logic in get_current_day_data()")
        print("   2. Peak period filtering")
        print("   3. Data loading sequence")
        print()
    else:
        print("❌ PROJECT DATA NOT AVAILABLE")
        print("   The issue appears to be with data availability.")
        print("   Consider using earlier dates or checking NEMOSIS status.")


if __name__ == "__main__":
    print("Starting NEMOSIS data availability debug...")
    print()

    # Test various date ranges
    test_results = test_nemosis_download()

    # Test specific project dates
    project_data = test_specific_project_dates()

    # Generate summary
    generate_summary_report(test_results, project_data)

    print("\nDebug complete!")