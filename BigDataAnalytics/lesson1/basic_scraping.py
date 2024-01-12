import time
from bs4 import BeautifulSoup
import re

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By

def get_browser():
    options = Options()

    try:
        # initializing webdriver for Chrome with our options
        browser = webdriver.Chrome(options=options)
        print("Success.")
    except:
        print("It failed.")
    return browser
browser = get_browser()


URL = "https://www.rottentomatoes.com/browse/movies_in_theaters/sort:newest"
browser.get(URL)

# Give the browser time to load all content.
time.sleep(3)

data = browser.find_elements(By.CSS_SELECTOR, ".p--small")


def get_text(content):
    inner_html = content.get_attribute('innerHTML')

    # Beautiful soup allows us to remove HTML tags from our content.
    soup = BeautifulSoup(inner_html, features="lxml")
    raw_string = soup.get_text()

    # Remove hidden carriage returns and tabs.
    text_only = re.sub(r"[\n\t]*", "", raw_string)
    # Replace two or more consecutive empty spaces with '*'
    text_only = re.sub('[ ]{2,}', ' ', text_only)

    return text_only


for i in range(0, len(data)):
    text = get_text(data[i])
    # date = getText(dates[i])
    print(str(i) + " " + text)
    print("***")  # Go to new line.
