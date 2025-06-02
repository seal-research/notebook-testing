from time import sleep
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from argparse import ArgumentParser
import os
import shutil
import pandas as pd
import re

baseURL = "https://www.kaggle.com"
driver = None

MASTER_CSV = "notebook_versions_local_download.csv"
MAIN_FOLDER = "kaggle_notebooks_collection_part" 

def initDriver():
    global driver 
    options = Options()
    options.add_argument("start-maximized")
    options.add_argument("--headless")
    prefs = {"download.default_directory": os.getcwd()}
    options.add_experimental_option("prefs", prefs)
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    sleep(2)

def loadDriver(url, timeout=3):
    global driver
    driver.get(url)
    driver.implicitly_wait(timeout)

def extract_user_project(notebook_link):
    # Expected format: https://www.kaggle.com/username/projectname
    pattern = r"kaggle\.com/([^/]+)/([^/]+)"
    match = re.search(pattern, notebook_link)
    if match:
        return match.group(1), match.group(2)
    return "unknown", "unknown"

def downloadNotebook(url, currVer, username, projectname):
    loadDriver(url)
    
    try:
        btnDot = driver.find_element(by=By.XPATH, value='//*[@id="site-content"]/div[2]/div/div/div[2]/div[1]/div/div[3]/button')
    except: 
        btnDot = driver.find_element(by=By.XPATH, value='//*[@id="site-content"]/div[2]/div/div/div[4]/div[1]/div/div[3]/button')
    
    btnDot.click()
    sleep(2)
    popupBox = driver.find_element(by=By.XPATH, value='/html/body/div[3]/div[3]/ul')
    btnDownload = popupBox.find_element(by=By.XPATH, value="/html/body/div[3]/div[3]/ul/li[5]")
    btnDownload.click()

    download_dir = os.getcwd()
    before_download = set(os.listdir(download_dir))
    sleep(5)
    after_download = set(os.listdir(download_dir))
    new_files = after_download - before_download
    downloaded_files = [f for f in new_files if f.endswith('.ipynb')]

    if downloaded_files:
        latest_file = os.path.join(download_dir, downloaded_files[0])
        
        base_folder = os.path.join(MAIN_FOLDER, f"kaggle__{username}__{projectname}")
        versions_folder = os.path.join(base_folder, "kaggle_other_versions")
        
        os.makedirs(versions_folder, exist_ok=True)

        # Move the file to the versions folder with a versioned filename
        filename_without_ext = os.path.splitext(downloaded_files[0])[0]
        new_filename = f"{filename_without_ext}_version_{currVer}.ipynb"
        local_path = os.path.join(versions_folder, new_filename)
        shutil.move(latest_file, local_path)
        print(f"Downloaded version {currVer} to {local_path}")
        return local_path
    else:
        print("No new .ipynb file found.")
        return None

def main(args):
    csv_path = args.csv_file
    os.makedirs(MAIN_FOLDER, exist_ok=True)
    
    print(f"Loading notebooks from CSV: {csv_path}")
    versions_df = pd.read_csv(csv_path)
    
    if versions_df.empty:
        print("No notebook versions found in the CSV file.")
        return
        
    initDriver()
    local_paths = []

    for idx, row in versions_df.iterrows():
        version_link = row['version_link']
        notebook_link = row['notebook_link']
        version_number = row['version_number']
        username, projectname = extract_user_project(notebook_link)
        
        print(f"Processing {idx+1}/{len(versions_df)}: Version {version_number} for {username}/{projectname}")
        local_path = downloadNotebook(version_link, version_number, username, projectname)
        local_paths.append(local_path if local_path else "Download failed")

        sleep(1)

    versions_df['local_path'] = local_paths

    # Save updated CSV with local paths
    output_csv_path = os.path.join(MAIN_FOLDER, MASTER_CSV)
    versions_df.to_csv(output_csv_path, index=False)
    print(f"CSV updated with download paths at {output_csv_path}")
    
    driver.quit()

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('-csv', '--csv_file', required=True, help='Path to the CSV with version links')
    args = parser.parse_args()
    main(args)