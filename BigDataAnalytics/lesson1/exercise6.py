import time
from bs4 import BeautifulSoup
import re

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By


def get_browser():
    options = Options()
    try:
        browser = webdriver.Chrome(options=options)
        print("Success.")
    except:
        print("It failed.")
    return browser


browser = get_browser()

URL = "https://shop.canon.ca/en_ca/cameras/all-cameras"
browser.get(URL)

# Give the browser time to load all content.
time.sleep(3)

products = {}


def get_text(content):
    inner_html = content.get_attribute('innerHTML')

    # Beautiful soup allows us to remove HTML tags from our content.
    soup = BeautifulSoup(inner_html, features="lxml")
    raw_string = soup.get_text()

    # Remove hidden carriage returns and tabs.
    text_only = re.sub(r"[\n\t]", " ", raw_string)
    # Replace two or more consecutive empty spaces with '*'
    text_only = re.sub('[ ]{2,}', ',', text_only)

    return text_only


def get_camera_info(text):
    text_array = text.split(",")
    name = text_array[6]

    # Extract price by finding the $ sign and getting all the characters up until 2 characters after the dot
    price_unparsed = text.split("$")[1]
    dollars = price_unparsed.split(".")[0]
    cents = price_unparsed.split(".")[1][:2]
    price = f"${dollars}.{cents}"

    return name, price


while True:
    try:
        data = browser.find_elements(By.CLASS_NAME, "product-item")

        for element in data:
            text = get_text(element)
            title, price = get_camera_info(text)
            if title in products:
                continue
            products[title] = price
            print("Product Name: " + title + " Price: " + price)
            print("***")  # Go to new line

        # Get the LAST load more button.
        load_more_button = browser.find_elements(By.CLASS_NAME, "amscroll-load-button")[-1]
        load_more_button.click()
        time.sleep(3)
        print("Loading more products...")
    except:
        print("No more products to load.")
        break
