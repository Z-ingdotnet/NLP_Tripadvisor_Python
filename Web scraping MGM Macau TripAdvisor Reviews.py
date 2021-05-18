# -*- coding: utf-8 -*-
"""
Created on Mon May 17 15:25:35 2021

@author: 119987
"""


import sys
import csv
from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
import time
import numpy
numpy.set_printoptions(threshold=sys.maxsize)

import requests 
from bs4 import BeautifulSoup
import argparse




#url="https://www.tripadvisor.com/Hotel_Review-g664891-d845057-Reviews-MGM_Macau-Macau.html#REVIEWS"

#driver = webdriver.Chrome(executable_path='X:/Program Files (x86)/Google/Chrome/Application/chrome.exe')

#driver_url = r"C:\Users\Anaconda3\msedgedriver\msedgedriver.exe"


#driver = webdriver.Edge('X:/ProgramData/Anaconda3/msedgedriver.exe')
#driver =webdriver.Chrome(ChromeDriverManager().install()) # diver
#driver.get(url)





pathToReviews = "Z:/MGM/My work/Python/TripReviews.csv"
pathToStoreInfo = "Z:/MGM/My work/Python/TripHoteInfo.csv"



#def scrapeRestaurantsUrls(tripURLs):
#    urls =[]
#    for url in tripURLs:
#        page = requests.get(url)
#        soup = BeautifulSoup(page.text, 'html.parser')
#        results = soup.find('div', class_='_1l3JzGX1')
#        stores = results.find_all('div', class_='wQjYiB7z')
#        for store in stores:
#            unModifiedUrl = str(store.find('a', href=True)['href'])
#            urls.append('https://www.tripadvisor.com'+unModifiedUrl)            
#    return urls


#<span class="_33O9dg0j">2,171 reviews</span>

def splitString(str):
 
    alpha = ""
    num = ""
    special = ""
    for i in range(len(str)):
        if (str[i].isdigit()):
            num = num+ str[i]
        elif((str[i] >= 'A' and str[i] <= 'Z') or
             (str[i] >= 'a' and str[i] <= 'z')):
            alpha += str[i]
        else:
            special += str[i]
 
    print(alpha)
    print(num )
    print(special)
 

def scrapeHotelInfo(url):
    print(url)
    page = requests.get(url)
    soup = BeautifulSoup(page.text, 'html.parser')
    hotelname = soup.find('h1', class_='_1mTlpMC3').text.strip()
    #avgRating = soup.find('span', class_='_3fVK8yi6').text.strip()
    avgRating = soup.find_all('span', class_='_3fVK8yi6')
    storeAddress = soup.find('div', class_= '_1sPw_t0w _3sCS_WGO').find('span', class_='_3ErVArsu jke2_wbp').text.strip()
    #noReviews = soup.find('a', class_='_2F5IkNIg').text.strip().split()[0]
    #noReviews = soup.find('a', class_='_2F5IkNIg').text.strip().split()[0]
    #noReviews = soup.find('a', class_= '_2F5IkNIg').find('span', class_='_33O9dg0j').text.strip()
    noReviews = soup.find('a', class_= '_15eFvJyR _3nlVsadw').find('span', class_='_33O9dg0j').text.strip()
    with open(pathToStoreInfo, mode='a', encoding="utf-8") as trip:
        data_writer = csv.writer(trip, delimiter = ',', quotechar = '"', quoting = csv.QUOTE_MINIMAL)
        data_writer.writerow([hotelname, storeAddress, avgRating, noReviews])
    #print(hotelname) 
    #print(avgRating) 
    #print(storeAddress) 
    #print(noReviews) 



#parser = argparse.ArgumentParser()
#parser.add_argument('https://www.tripadvisor.com/Hotel_Review-g664891-d845057-Reviews-MGM_Macau-Macau.html#REVIEWS', required=True, help ='need starting url')
#parser.add_argument('-i', '--info', action='store_true', help="Collects restaurant's info")
#parser.add_argument('-m', '--many', action='store_true', help="Collects whole area info")
#args = parser.parse_args()
startingUrl = 'https://www.tripadvisor.com/Hotel_Review-g664891-d845057-Reviews-MGM_Macau-Macau.html#REVIEWS'


driver = webdriver.Edge('X:/ProgramData/Anaconda3/msedgedriver.exe')
#driver = webdriver.Chrome("C:/Users/Zing/OneDrive/GitHub/Python/Iverson's/chromedriver.exe"
#                          #,chrome_options=chrome_options
#                         )
#driver.get(startingUrl)



#if args.info:
#    info = True
#else:
#    info = False
#if args.many:
    #urls = scrapeRestaurantsUrls([startingUrl])
#else:
urls = [startingUrl]

info = True    
#urls = 'https://www.tripadvisor.com/Hotel_Review-g664891-d845057-Reviews-MGM_Macau-Macau.html#REVIEWS'

for url in urls:
    print(url)
    driver.get(url)
    #if you want to scrape restaurants info
    if info == True:
        #scrapeHotelInfo(url) 
        soup = BeautifulSoup(driver.page_source, 'html.parser')
        #results = soup.find('div', class_='oETBfkHU')
        results = soup('div', class_='was-ssr-only')
        reviews = soup.find_all('div', class_='_2wrUUKlw _3hFEdNs8')
        #reviews2 = soup.find('div', class_= '_2wrUUKlw _3hFEdNs8').find('span', class_='_3ErVArsu jke2_wbp').text.strip()
        print(reviews)






#####################testing 
def scrapeUrls(url):
    url =[startingUrl]
    for url in url:
        page = requests.get(url)
        soup = BeautifulSoup(driver.page_source, 'html.parser')
        results = soup('div', class_='was-ssr-only')
        try:
            reviews = soup.find_all('div', class_='_2wrUUKlw _3hFEdNs8')
        except Exception:
            continue
        try:
           
                for review in reviews:
                    ratingDate = review.find('div', class_='_2fxQ4TOx').text.strip()
                    #ratingDate = ratingDate.select('div > div')[0].get_text(strip=True)
                    #ratingDate=ratingDate[ratingDate.index('a'):].split('wrote a review')
                    #ratingDate=ratingDate[ratingDate.index('a'):]
                    ratingDate=ratingDate.split('wrote a review ')[1]
                    stayDate = review.find('span', class_='_34Xs-BQm').text.strip()
                    text_review = review.find('q', class_='IRsGHoPm')
                    if len(text_review.contents) > 2:
                        #reviewText = str(text_review.contents[0][:-3]) + ' ' + str(text_review.contents[1].text)
                        #reviewText= str(text_review.contents[0][:-3]) 
                        #reviewText = str(text_review.contents[1].text)
                        reviewText = text_review.text
                    else:
                        reviewText = text_review.text
                                      
                    reviewerUsername = review.find('a', class_='ui_header_link _1r_My98y').text
                   # reviewerUsername = reviewerUsername.select('div > div')[0].get_text(strip=True)
                    rating = review.find('div', class_='nf9vGX55').findChildren('span')
                    rating = str(rating[0]).split('_')[3].split('0')[0]
                    #data_writer.writerow([hotelname, reviewerUsername, ratingDate, reviewText, rating])
                    print(reviewerUsername)
                    print(rating)
                    print(reviewText)
                    print(stayDate)
                    print(ratingDate)
        except:
            pass


scrapeUrls(url)

#ratingDate = 'Campsay upon Crooked wrote a review Feb 2021'

##first_a = ratingDate.index('a')
#a_split = ratingDate[ratingDate.index('a'):].split('wrote a review')
##a_split[0] = a_string[:first_a] + a_split[0]
##a_split = [x.strip() for x in a_split]
#print(a_split[1])

#####################testing 



#####################testing2 
def scrapeUrls(url):
    url =[startingUrl]
    for url in url:
        #page = requests.get(url)
        soup = BeautifulSoup(driver.page_source, 'html.parser')
        results = soup('div', class_='was-ssr-only')
        try:
            reviews = soup.find_all('div', class_='_2wrUUKlw _3hFEdNs8')
        except Exception:
            continue
        try:
            with open(pathToReviews, mode='a', encoding="utf-8") as trip_data:
                for review in reviews:
                    ratingDate = review.find('div', class_='_2fxQ4TOx').text.strip()
                    #ratingDate = ratingDate.select('div > div')[0].get_text(strip=True)
                    #ratingDate=ratingDate[ratingDate.index('a'):].split('wrote a review')
                    #ratingDate=ratingDate[ratingDate.index('a'):]
                    ratingDate=ratingDate.split('wrote a review ')[1]
                    stayDate = review.find('span', class_='_34Xs-BQm').text.strip()
                    text_review = review.find('q', class_='IRsGHoPm')
                    if len(text_review.contents) > 2:
                       # reviewText = str(text_review.contents[0][:-3]) + ' ' + str(text_review.contents[1].text)
                        #reviewText= str(text_review.contents[0][:-3]) 
                        #reviewText = str(text_review.contents[1].text)
                        reviewText = text_review.text
                    else:
                        reviewText = text_review.text
                                      
                    reviewerUsername = review.find('a', class_='ui_header_link _1r_My98y').text
                   # reviewerUsername = reviewerUsername.select('div > div')[0].get_text(strip=True)
                    rating = review.find('div', class_='nf9vGX55').findChildren('span')
                    rating = str(rating[0]).split('_')[3].split('0')[0]
                    data_writer = csv.writer(trip_data, delimiter = ',', quotechar = '"', quoting = csv.QUOTE_MINIMAL)
                    data_writer.writerow([reviewerUsername, rating, reviewText, ratingDate,stayDate])
        except:
            pass


scrapeUrls(url)
#####################testing2






startingUrl = 'https://www.tripadvisor.com/Hotel_Review-g664891-d845057-Reviews-MGM_Macau-Macau.html'
driver = webdriver.Edge('X:/ProgramData/Anaconda3/msedgedriver.exe')
urls = [startingUrl]

def scrapeUrls(url):
    for url in urls:
        print(url)
        nextPage = True
        while nextPage:
            time.sleep(10)
            page = requests.get(url)
            soup = BeautifulSoup(driver.page_source, 'html.parser')
            results = soup('div', class_='was-ssr-only')
            try:
                reviews = soup.find_all('div', class_='_2wrUUKlw _3hFEdNs8')
            except Exception:
                continue
            try:
                    with open(pathToReviews, mode='a', encoding="utf-8") as trip_data:
                        for review in reviews:
                            ratingDate = review.find('div', class_='_2fxQ4TOx').text.strip()
                            #ratingDate = ratingDate.select('div > div')[0].get_text(strip=True)
                            #ratingDate=ratingDate[ratingDate.index('a'):].split('wrote a review')
                            #ratingDate=ratingDate[ratingDate.index('a'):]
                            ratingDate=ratingDate.split('wrote a review ')[1]
                            stayDate = review.find('span', class_='_34Xs-BQm').text.strip()
                            text_review = review.find('q', class_='IRsGHoPm')
                            if len(text_review.contents) > 2:
                                reviewText = str(text_review.contents[0][:-3]) + ' ' + str(text_review.contents[1].text)
                                #reviewText= str(text_review.contents[0][:-3])
                                #reviewText = str(text_review.contents[1].text)
                            else:
                                    reviewText = text_review.text
                            reviewerUsername = review.find('a', class_='ui_header_link _1r_My98y').text
                            # reviewerUsername = reviewerUsername.select('div > div')[0].get_text(strip=True)
                            rating = review.find('div', class_='nf9vGX55').findChildren('span')
                            rating = str(rating[0]).split('_')[3].split('0')[0]
                            data_writer = csv.writer(trip_data, delimiter = ',', quotechar = '"', quoting = csv.QUOTE_MINIMAL)
                            data_writer.writerow([reviewerUsername, rating, reviewText, ratingDate,stayDate])
            except:
                pass
                            #Go to next page if exists
                try:
                    unModifiedUrl = str(soup.find('a', class_ = 'ui_button nav next primary ',href=True)['href'])
                    url = 'https://www.tripadvisor.com/Hotel_Review-g664891-d845057-Reviews-MGM_Macau-Macau.html' + unModifiedUrl
                except:
                    nextPage = False


scrapeUrls(url)













#########################working


def scrapeUrls(url):
        nextPage = True
        while nextPage:
            driver.get(url)
            time.sleep(1)
            #more = driver.find_elements_by_xpath("//span[contains(text(),'Read more')]")
            #for x in range(0,len(more)):
                #try:
                    #driver.execute_script("arguments[0].click();", more[x])
                    #time.sleep(3)
                #except:
                    #pass
            page = requests.get(url)
            soup = BeautifulSoup(driver.page_source, 'html.parser')
            #results = soup('div', class_='was-ssr-only')
            try:
                reviews = soup.find_all('div', class_='_2wrUUKlw _3hFEdNs8')
            except Exception:
                continue
            try:
                    with open(pathToReviews, mode='a', encoding="utf-8") as trip_data:
                        for review in reviews:
                            ratingDate = review.find('div', class_='_2fxQ4TOx').text.strip()
                            #ratingDate = ratingDate.select('div > div')[0].get_text(strip=True)
                            #ratingDate=ratingDate[ratingDate.index('a'):].split('wrote a review')
                            #ratingDate=ratingDate[ratingDate.index('a'):]
                            ratingDate=ratingDate.split('wrote a review ')[1]
                            stayDate = review.find('span', class_='_34Xs-BQm').text.strip()
                            text_review = review.find('q', class_='IRsGHoPm')
                            if len(text_review.contents) > 2:
                                #reviewText = str(text_review.contents[0][:-3]) + ' ' + str(text_review.contents[1].text)
                                #reviewText= str(text_review.contents[0][:-3])
                                #reviewText = str(text_review.contents[1].text)
                                reviewText = text_review.text
                            else:
                                reviewText = text_review.text
                            reviewerUsername = review.find('a', class_='ui_header_link _1r_My98y').text
                            # reviewerUsername = reviewerUsername.select('div > div')[0].get_text(strip=True)
                            rating = review.find('div', class_='nf9vGX55').findChildren('span')
                            rating = str(rating[0]).split('_')[3].split('0')[0]
                            data_writer = csv.writer(trip_data, delimiter = ',', quotechar = '"', quoting = csv.QUOTE_MINIMAL)
                            data_writer.writerow([reviewerUsername, rating, reviewText, ratingDate,stayDate])
                            #print(reviewerUsername)
                            #print(rating)
                            #print(reviewText)
                            #print(stayDate)
                            #print(ratingDate)
            except:
                pass
                            #Go to next page if exists
                try:
                    unModifiedUrl = str(soup.find('a', class_ = 'ui_button nav next primary',href=True)['href'])
                    url = 'https://www.tripadvisor.com' + unModifiedUrl
                except:
                    nextPage = False

scrapeUrls(url)
































#####################example
#https://github.com/LaskasP/TripAdvisor-Python-Scraper-Restaurants-2021/blob/main/Scraper.py

#if args.info:
#    info = True
#else:
#    info = False
#if args.many:
    #urls = scrapeRestaurantsUrls([startingUrl])
#else:
urls = [startingUrl]

info = True    
#urls = 'https://www.tripadvisor.com/Hotel_Review-g664891-d845057-Reviews-MGM_Macau-Macau.html#REVIEWS'

driver = webdriver.Edge('X:/ProgramData/Anaconda3/msedgedriver.exe')
for url in urls:
    print(url)
    driver.get(url)
    #if you want to scrape restaurants info
    if info == True:
        scrapeHotelInfo(url)
        
    nextPage = True
    while nextPage:
        #Requests
        driver.get(url)
        time.sleep(10)
        #Click More button
        more = driver.find_elements_by_xpath("//span[contains(text(),'More')]")
        for x in range(0,len(more)):
            try:
                driver.execute_script("arguments[0].click();", more[x])
                time.sleep(3)
            except:
                pass
        soup = BeautifulSoup(driver.page_source, 'html.parser')
        #Store name
        hotelname = soup.find('h1', class_='_1mTlpMC3').text.strip()
        #Reviews
        results = soup.find('div', class_='listContainer hide-more-mobile')
        try:
            reviews = results.find_all('div', class_='prw_rup prw_reviews_review_resp')
        except Exception:
            continue
        #Export to csv
        try:
            with open(pathToReviews, mode='a', encoding="utf-8") as trip_data:
                data_writer = csv.writer(trip_data, delimiter = ',', quotechar = '"', quoting = csv.QUOTE_MINIMAL)
                for review in reviews:
                    ratingDate = review.find('span', class_='ratingDate').get('title')
                    text_review = review.find('p', class_='partial_entry')
                    if len(text_review.contents) > 2:
                        reviewText = str(text_review.contents[0][:-3]) + ' ' + str(text_review.contents[1].text)
                    else:
                        reviewText = text_review.text
                    reviewerUsername = review.find('div', class_='info_text pointer_cursor')
                    reviewerUsername = reviewerUsername.select('div > div')[0].get_text(strip=True)
                    rating = review.find('div', class_='ui_column is-9').findChildren('span')
                    rating = str(rating[0]).split('_')[3].split('0')[0]
                    data_writer.writerow([hotelname, reviewerUsername, ratingDate, reviewText, rating])
        except:
            pass
        #Go to next page if exists
        try:
            unModifiedUrl = str(soup.find('a', class_ = 'ui_button nav next primary ',href=True)['href'])
            url = 'https://www.tripadvisor.com/Hotel_Review-g664891-d845057-Reviews-MGM_Macau-Macau.html#REVIEWS' + unModifiedUrl
        except:
            nextPage = False
