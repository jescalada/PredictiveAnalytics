import time
from bs4 import BeautifulSoup
import re

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By

def get_browser():
    options = Options()

    # this parameter tells Chrome that
    # it should be run without UI (Headless)
    # Uncommment this line if you want to hide the browser.
    # options.add_argument('--headless=new')

    try:
        # initializing webdriver for Chrome with our options
        browser = webdriver.Chrome(options=options)
        print("Success.")
    except:
        print("It failed.")
    return browser
browser = get_browser()

URL = "https://www.bcit.ca/study/programs/5512cert#courses"
browser.get(URL)

# Give the browser time to load all content.
time.sleep(3)

course_names = browser.find_elements(By.CLASS_NAME, "course_name")
course_numbers = browser.find_elements(By.CLASS_NAME, "course_number")


def getText(content):
    innerHtml = content.get_attribute('innerHTML')

    # Beautiful soup allows us to remove HTML tags from our content.
    soup = BeautifulSoup(innerHtml, features="lxml")
    rawString = soup.get_text()

    # Remove hidden carriage returns and tabs.
    textOnly = re.sub(r"[\n\t]*", "", rawString)
    # Replace two or more consecutive empty spaces with '*'
    textOnly = re.sub('[ ]{2,}', ' ', textOnly)

    return textOnly


for i in range(0, len(course_names)):
    text = getText(course_numbers[i]) + " - " + getText(course_names[i])

    print(str(i) + " " + text)
    print("Juan ***")  # Go to new line.
