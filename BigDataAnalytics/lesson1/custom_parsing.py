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

URL = "https://vpl.bibliocommons.com/events/search/index"
browser.get(URL)

# Give the browser time to load all content.
time.sleep(3)

data = browser.find_elements(By.CSS_SELECTOR, ".cp-events-search-item")
location = browser.find_elements(By.CSS_SELECTOR, ".cp-event-location")

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


def getEndTime(content):
    amIdx = content.find('am') # Get index of 1st 'am' occurence in string.
    pmIdx = content.find('pm')

    if(amIdx>=0 and (amIdx<pmIdx or pmIdx==-1)):
        endTime = content[0:amIdx] + "am" # add 'am' to substring
        return endTime
    startTime = content[0:pmIdx] + "pm"
    return startTime


def getEventTitle(dayNumOfMonth, text):
    daysOfWeek = ['Sunday', 'Monday', 'Tuesday',
                  'Wednesday', 'Thursday', 'Friday', 'Saturday']
    dayIndexes = []
    for day in daysOfWeek:
        dayIdx = text.find(day)
        if(dayIdx >=0):
            dayIndexes.append(dayIdx)
    dayIndexes.sort()
    startIndex = text.find(dayNumOfMonth) + len(dayNumOfMonth)
    title = text[startIndex:dayIndexes[0]]
    return title


for i in range(0, len(data)):
    text      = getText(data[i])
    textArray = text.split(',')

    DATE_IDX  = 1
    YEAR_IDX  = 2
    INFO_IDX  = 3
    date      = textArray[DATE_IDX].split("on ")[0]
    dayOfMonth = date.strip().split(" ")[1]
    year      = textArray[YEAR_IDX].strip() # strip() removes extra characters.
    startTime = textArray[INFO_IDX].split("–")[0]
    endTime   = getEndTime(textArray[INFO_IDX].split("–")[1])
    title     = getEventTitle(dayOfMonth, text)
    location_split = textArray[INFO_IDX].split("Branch")
    location = textArray[INFO_IDX].split("Branch")[1] if len(location_split) > 1 else "Event location: Online Event"

    print("\n" + title)
    print("Date: " + date + "   Year: " + year + " Start time: " + startTime +\
          "  End time: " + endTime + " " + location)
    print("***") # Go to new line.
