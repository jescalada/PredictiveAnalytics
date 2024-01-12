import time
import re
from bs4 import BeautifulSoup
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
URL = "https://bpl.bc.ca/events/"
browser.get(URL)

# Give the browser time to load all content.
time.sleep(1)

SEARCH_TERM = "Barf"
search = browser.find_element(By.CSS_SELECTOR, "input")
search.send_keys(SEARCH_TERM)

# Find the search button - this is only enabled when a search query is entered
button = browser.find_element(By.CSS_SELECTOR, "button")
button.click()  # Click the button.
time.sleep(3)


def get_content(content):
    text_content = content.get_attribute('innerHTML')

    # Beautiful soup removes HTML tags from our content if it exists.
    soup = BeautifulSoup(text_content, features="lxml")
    raw_string = soup.get_text().strip()

    # Remove hidden characters for tabs and new lines.
    raw_string = re.sub(r"[\n\t]*", "", raw_string)

    # Replace two or more consecutive empty spaces with '*'
    raw_string = re.sub('[ ]{2,}', '*', raw_string)
    return raw_string


# content = browser.find_elements_by_css_selector(".cp-search-result-item-content")
page_num = 1

for i in range(0, 10):
    try:
        titles = browser.find_elements(By.CSS_SELECTOR, ".title-content")
        formats = browser.find_elements(By.CSS_SELECTOR, ".manifestation-item-format-info-wrap")

        NUM_ITEMS = len(titles)

        # This technique works only if counts of all scraped items match.
        if (len(titles) != NUM_ITEMS or len(formats) != NUM_ITEMS):
            print("**WARNING: Items scraped are misaligned because their counts differ")

        for i in range(0, NUM_ITEMS):
            title = get_content(titles[i])
            mediaFormat = get_content(formats[i])
            print("Title: " + title)
            print("Media: " + mediaFormat)
            print("********")

        # Go to a new page.
        page_num += 1

        URL_NEXT = "https://burnaby.bibliocommons.com/v2/search?query=" \
                   + SEARCH_TERM + "&searchType=smart&pagination_page="

        URL_NEXT = URL_NEXT + str(page_num)
        browser.get(URL_NEXT)
        print("Count: ", str(i))
        time.sleep(3)
    except:
        print("No more pages")
        break

browser.quit()