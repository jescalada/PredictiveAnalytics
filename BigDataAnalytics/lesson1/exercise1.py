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

URL = "https://www.bcit.ca/study/programs/5512cert#courses"
browser.get(URL)

# Give the browser time to load all content.
time.sleep(3)

course_names = browser.find_elements(By.CLASS_NAME, "course_name")
course_numbers = browser.find_elements(By.CLASS_NAME, "course_number")


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


for i in range(0, len(course_names)):
    text = get_text(course_numbers[i]) + " - " + get_text(course_names[i])

    print(str(i) + " " + text)
    print("Juan ***")  # Go to new line.
