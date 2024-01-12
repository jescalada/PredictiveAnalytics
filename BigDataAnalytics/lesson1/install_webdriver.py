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
my_browser = get_browser()
