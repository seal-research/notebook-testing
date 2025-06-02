from time import sleep
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from argparse import ArgumentParser
import os
import pandas as pd
import re
import statistics
import json
import datetime

baseURL = "https://www.kaggle.com"
driver = None
CACHE_FILE = "notebook_cache.json"
CHECKPOINT_INTERVAL = 5  # Save checkpoint after processing this many notebooks

def init_driver():
    """Initialize the Chrome webdriver with appropriate options"""
    global driver 
    options = Options()
    options.add_argument("start-maximized")
    options.add_argument("--headless")  # Run in headless mode
    prefs = {"download.default_directory": os.getcwd()}  # Set default download directory
    options.add_experimental_option("prefs", prefs)
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    sleep(2)

def load_driver(url, timeout=3):
    """Load a URL in the webdriver"""
    global driver
    driver.get(url)
    driver.implicitly_wait(timeout)
    driver.fullscreen_window()

def go_to_version(notebook_path):
    """Navigate to the version history page for a notebook"""
    notebook_type = ''
    load_driver(notebook_path)
    try:
        # Competition button                                                  
        btn_dot = driver.find_element(by=By.XPATH, value="/html/body/div[2]/div[3]/div[2]/div/div/div[5]/div[1]/div/div[2]/div[1]/div/div/div[1]/div[2]/button")
        btn_dot.click()
        sleep(2)
        notebook_type = 'competition'
    except:
        # Regular notebook button
            btn_dot = driver.find_element(by=By.XPATH, value="/html/body/div[2]/div[3]/div[2]/div/div/div[5]/div[1]/div[1]/button[1]")
            btn_dot.click() 
            sleep(2)
            notebook_type = 'regular'
        # try:    
        #     # Regular notebook button
        #     btn_dot = driver.find_element(by=By.XPATH, value="/html/body/div[2]/div[3]/div[2]/div/div/div[5]/div[1]/div[1]/button[1]")
        #     btn_dot.click() 
        #     sleep(2)
        #     notebook_type = 'regular'
        #     print('regular')
        # except:
        #     # Alternate competition button
        #     try:
        #         btn_dot = driver.find_element(by=By.XPATH, value="/html/body/main/div[1]/div/div[5]/div[2]/div/div/div[5]/div[1]/div/div/div/button")
        #         btn_dot.click()
        #         sleep(2)
        #         notebook_type = 'competition'
        #     except Exception as e:
        #         print(f"Failed to find and click the button: {e}")
    print(f"Notebook type: {notebook_type}")
    return notebook_type

def get_versions_with_errors():
    """Get a dictionary of version numbers and their error status from the version history panel"""
    error_dict = {}
    try:
        # Get the version history list container
        version_list = driver.find_element(By.XPATH, 
            "/html/body/div[3]/div[3]/div/div/div[3]/div/div/div[2]/div/div[2]/ul")
        
        # Get all version items in the list
        version_items = version_list.find_elements(By.CSS_SELECTOR, 'div[role="button"]')

        version_items = version_items[::-1]  # Reverse to start from the oldest version
        
        for idx, item in enumerate(version_items, 1):
            try:
                # Check if there's an error icon in this version
                error_icon = item.find_element(By.CSS_SELECTOR, 'img[src*="error-icon.svg"]')
                error_dict[idx] = True
            except:
                error_dict[idx] = False
                
    except Exception as e:
        print(f"Error while checking for version errors: {e}")
    
    return error_dict

def parse_diff(diff_text):
    """Extract addition and deletion numbers from diff text"""
    additions, deletions = 0, 0
    
    if not diff_text or diff_text == "No diff found" or diff_text == "Error" or "N/A" in diff_text:
        return additions, deletions
    
    # Extract additions
    additions_match = re.search(r'\+(\d+)', diff_text)
    if additions_match:
        additions = int(additions_match.group(1))
    
    # Extract deletions
    deletions_match = re.search(r'-(\d+)', diff_text)
    if deletions_match:
        deletions = int(deletions_match.group(1))
    
    return additions, deletions

def calculate_percentile(data, percentile):
    """
    Calculate the percentile value of a dataset
    """
    if not data:
        return 0
    
    sorted_data = sorted(data)
    
    index = (len(sorted_data) - 1) * (percentile / 100)
    
    if index.is_integer():
        return sorted_data[int(index)]
    
    # Otherwise, interpolate between the two nearest indices
    lower_index = int(index)
    upper_index = lower_index + 1
    
    lower_value = sorted_data[lower_index]
    upper_value = sorted_data[upper_index] if upper_index < len(sorted_data) else sorted_data[-1]
    
    # Linear interpolation
    fraction = index - lower_index
    return lower_value + (upper_value - lower_value) * fraction

def filter_versions(all_versions_data, all_diffs, max_diff_absolute=500, percentile_threshold=75):
    """
    Filter versions based on:
    1. No errors
    2. Reasonable diff size, using statistical methods
    
    Parameters:
    - all_versions_data: List of version data dictionaries
    - all_diffs: List of diff strings
    - max_diff_absolute: Absolute maximum diff size regardless of statistics
    - percentile_threshold: Percentile to use as cutoff (e.g., 75 means keep only versions below 75th percentile)
    """
    filtered_versions = []
    
    # Parse diffs to get numerical values
    parsed_diffs = []
    for diff in all_diffs:
        additions, deletions = parse_diff(diff)
        total_diff = additions + deletions
        parsed_diffs.append(total_diff)
    
    # Calculate statistics for non-zero diffs
    valid_diffs = [d for d in parsed_diffs if d > 0]
    
    if not valid_diffs:
        print("No valid diffs found")
        return []
    
    try:
        median_diff = statistics.median(valid_diffs)
        
        # Calculate percentile as threshold
        percentile_value = calculate_percentile(valid_diffs, percentile_threshold)
        threshold = min(percentile_value, max_diff_absolute)
        
        print(f"Diff statistics - Median: {median_diff}, {percentile_threshold}th percentile: {percentile_value}")
        print(f"Using threshold: {threshold}")
    except Exception as e:
        # Fallback if statistical methods fail
        print(f"Error calculating statistics: {e}")
        threshold = min(sum(valid_diffs) / len(valid_diffs), max_diff_absolute)
        print(f"Fallback to simple average threshold: {threshold}")
    
    # Filter versions
    for i, version_data in enumerate(all_versions_data):
        if i < len(parsed_diffs):
            diff_size = parsed_diffs[i]
            if not version_data['has_error'] and (diff_size <= threshold):
                filtered_versions.append(version_data)
                
    # If we filtered too much and have no versions, take the smallest diff version
    if not filtered_versions and all_versions_data:
        non_error_versions = [v for v in all_versions_data if not v['has_error']]
        if non_error_versions:
            min_diff_idx = 0
            min_diff_value = float('inf')
            
            for i, version in enumerate(non_error_versions):
                idx = all_versions_data.index(version)
                if idx < len(parsed_diffs) and parsed_diffs[idx] < min_diff_value and parsed_diffs[idx] > 0:
                    min_diff_value = parsed_diffs[idx]
                    min_diff_idx = i
            
            filtered_versions.append(non_error_versions[min_diff_idx])
            print(f"No versions met criteria. Selected version with smallest non-zero diff: {min_diff_value}")
    
    return filtered_versions

def load_cache():
    """Load the cache from file if it exists"""
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, 'r') as f:
                cache = json.load(f)
            print(f"Loaded cache with {len(cache['processed_notebooks'])} processed notebooks")
            return cache
        except Exception as e:
            print(f"Error loading cache: {e}")
            return {'processed_notebooks': [], 'all_filtered_versions': []}
    else:
        return {'processed_notebooks': [], 'all_filtered_versions': []}
    
def save_cache(cache):
    """Save the cache to file"""
    try:
        # Add timestamp
        cache['last_updated'] = datetime.datetime.now().isoformat()
        
        with open(CACHE_FILE, 'w') as f:
            json.dump(cache, f)
        print(f"Cache saved with {len(cache['processed_notebooks'])} processed notebooks")
    except Exception as e:
        print(f"Error saving cache: {e}")
        
def save_checkpoint(all_filtered_versions, processed_notebooks):
    """Save intermediate results to CSV files"""
    # Save all filtered versions to CSV
    if all_filtered_versions:
        filtered_df = pd.DataFrame(all_filtered_versions)
        
        # Save all the data including diffs
        filtered_df.to_csv('filtered_notebook_versions.csv', index=False)
        
        # Create a simplified version with just the essential columns
        simplified_df = filtered_df[['notebook_link', 'version_link', 'notebook_type']]
        simplified_df.to_csv('simplified_notebook_versions.csv', index=False)
        
        print(f"Checkpoint saved: {len(all_filtered_versions)} versions")

def main(args):
    init_driver()
    
    # Load notebook links
    notebook_links_df = pd.read_csv(args.csv_file)
    
    # Load cache
    cache = load_cache()
    all_filtered_versions = cache['all_filtered_versions']
    processed_notebooks = cache['processed_notebooks']
    
    # Determine which notebooks still need processing
    pending_notebooks = []
    for index, row in notebook_links_df.iterrows():
        if row['url'] not in processed_notebooks:
            pending_notebooks.append((index, row))
    
    print(f"Loaded {len(notebook_links_df)} notebooks from CSV")
    print(f"Already processed {len(processed_notebooks)} notebooks")
    print(f"Need to process {len(pending_notebooks)} notebooks")
    
    # Process pending notebooks
    notebooks_since_checkpoint = 0
    for index, row in pending_notebooks:
        notebook_path = row['url']
        print(f"\nProcessing notebook {index+1}/{len(notebook_links_df)}: {notebook_path}")
        notebook_type = go_to_version(notebook_path)
        print(f"Notebook type: {notebook_type}")
        # Get all the current versions  
        # try:
        #     version_container = driver.find_element(By.XPATH, "/html/body/div[3]/div[3]/div/div/div[3]/div/div/div[2]/div/div[2]/ul")
        #     ls_version = version_container.find_elements(By.CSS_SELECTOR, "div[role='button']")
        #     print(f"Found {len(ls_version)} versions")
            
        #     if len(ls_version) <= 1:
        #         print(f"Skipping notebook with only 1 version")
        #         processed_notebooks.append(notebook_path)
        #         notebooks_since_checkpoint += 1
        #         if notebooks_since_checkpoint >= CHECKPOINT_INTERVAL:
        #             save_cache({'processed_notebooks': processed_notebooks, 'all_filtered_versions': all_filtered_versions})
        #             save_checkpoint(all_filtered_versions, processed_notebooks)
        #             notebooks_since_checkpoint = 0
        #         continue
                
        #     version_errors = get_versions_with_errors()
        #     print(f"Found {sum(1 for v in version_errors.values() if not v)} non-error versions")
                                                    
        #     all_versions_data = []
        #     version_number = 1
        #     all_diffs = []
            
        #     for idx, version in enumerate(ls_version[::-1]):  # Reverse to start from oldest version
        #         has_error = version_errors.get(version_number, False)
        #         # Get the div_id for this version
        #         div_id = len(ls_version) - idx
                
        #         # Get diff information
        #         if not has_error:
        #             try:
        #                 # Get the diff element
        #                 diff_element_xpath = f"/html/body/div[3]/div[3]/div/div/div[3]/div/div/div[2]/div/div[2]/ul/div[{div_id}]/div[3]/div[2]"
        #                 diff_element = driver.find_element(By.XPATH, diff_element_xpath)
        #                 diff_text = diff_element.text
                        
        #                 # Extract the diff numbers
        #                 if "Diff:" in diff_text:
        #                     diff_part = diff_text.split("Diff:")[1].strip()
        #                     version_diff = diff_part
        #                 else:
        #                     version_diff = "No diff found"
        #             except Exception as e:
        #                 print(f"Error extracting diff: {e}")
        #                 version_diff = "Error"
        #         else:
        #             version_diff = "N/A (Error version)"
                
        #         all_diffs.append(version_diff)
                
        #         # Construct the appropriate XPath based on whether there's an error
        #         if has_error:
        #             button_xpath = f"/html/body/div[3]/div[3]/div/div/div[3]/div/div/div[2]/div/div[2]/ul/div[{div_id}]/div[3]/button"
        #         else:
        #             button_xpath = f"/html/body/div[3]/div[3]/div/div/div[3]/div/div/div[2]/div/div[2]/ul/div[{div_id}]/div[4]/button"

        #         try:
        #             # Click the button
        #             button = driver.find_element(By.XPATH, button_xpath)
        #             button.click()
        #             sleep(2)
                    
        #             # Switch to the new dialog window and get the URL
        #             dialog_element = driver.find_element(By.XPATH, "/html/body/div[4]/div[3]")
        #             url = dialog_element.find_element(By.TAG_NAME, "a").get_attribute("href")
        #             sleep(2)
                    
        #             # Close dialog by clicking outside of it
        #             driver.find_element(By.XPATH, "//body").click()
        #             sleep(1)
                    
        #             if 'scriptVersionId' not in url:
        #                 version_number += 1
        #                 continue
        #             print(f"Notebook type: {notebook_type}")    
        #             all_versions_data.append({
        #                 'notebook_link': notebook_path,
        #                 'version_link': url,
        #                 'version_number': version_number,
        #                 'notebook_type': notebook_type,
        #                 'has_error': has_error,
        #                 'diff': version_diff
        #             })
                    
        #             version_number += 1
        #         except Exception as e:
        #             print(f"Error processing version {version_number}: {e}")
        #             version_number += 1
                    
        #     # Parse and print all diffs for this notebook
        #     numerical_diffs = []
        #     for i, diff in enumerate(all_diffs):
        #         adds, dels = parse_diff(diff)
        #         total = adds + dels
        #         numerical_diffs.append(total)
        #         print(f"Version {i+1}: Diff {diff} (Total: {total})")
                
        #     # Filter versions based on diff size and errors
        #     if all_versions_data:
        #         filtered_versions = filter_versions(
        #             all_versions_data, 
        #             all_diffs,
        #             max_diff_absolute=args.max_diff,
        #             percentile_threshold=args.percentile
        #         )
        #         print(f"Selected {len(filtered_versions)} out of {len(all_versions_data)} versions")
        #         all_filtered_versions.extend(filtered_versions)
                
        #         # Display summary for debugging
        #         for v in filtered_versions:
        #             print(f"  Selected version {v['version_number']} with diff: {v['diff']}")
        
        # except Exception as e:
        #     print(f"Error processing notebook {notebook_path}: {e}")
            
        # Mark this notebook as processed regardless of outcome
        processed_notebooks.append(notebook_path)
        notebooks_since_checkpoint += 1
        
        # Save checkpoint periodically
        if notebooks_since_checkpoint >= CHECKPOINT_INTERVAL:
            save_cache({'processed_notebooks': processed_notebooks, 'all_filtered_versions': all_filtered_versions})
            save_checkpoint(all_filtered_versions, processed_notebooks)
            notebooks_since_checkpoint = 0

    # Save final results
    save_cache({'processed_notebooks': processed_notebooks, 'all_filtered_versions': all_filtered_versions})
    
    # Save results to CSV
    if all_filtered_versions:
        filtered_df = pd.DataFrame(all_filtered_versions)
        
        # Save all the data including diffs
        filtered_df.to_csv('filtered_notebook_versions1.csv', index=False)
        
        # Create a simplified version with just the essential columns
        simplified_df = filtered_df[['notebook_link', 'version_link', 'notebook_type']]
        simplified_df.to_csv('simplified_notebook_versions1.csv', index=False)
        
        print(f"\nProcessed {len(notebook_links_df)} notebooks total")
        print(f"Selected a total of {len(all_filtered_versions)} versions")
        print(f"Data saved to filtered_notebook_versions.csv and simplified_notebook_versions.csv")
    else:
        print("No versions were selected based on criteria")

    driver.quit()

if __name__ == "__main__":
    parser = ArgumentParser(description="Filter and store notebook version links from a CSV file with caching support.")
    parser.add_argument("-csv", "--csv_file", type=str, required=True, help="Path to the CSV file containing notebook links")
    parser.add_argument("--max_diff", type=int, default=500, help="Maximum acceptable diff size (default: 500)")
    parser.add_argument("--percentile", type=float, default=75, help="Percentile threshold for diffs (default: 75)")
    parser.add_argument("--reset_cache", action="store_true", help="Reset cache and start from scratch")
    args = parser.parse_args()
    
    # Check if we should clear the cache
    if args.reset_cache and os.path.exists(CACHE_FILE):
        os.remove(CACHE_FILE)
        print("Cache reset as requested")
        
    main(args)