
# coding: utf-8

# In[3]:


#####Assignment-3
##### Web Scraper
####Monica Vijaywargi
#####10432110

####Importing Libraries
import requests
from requests import get
from requests import post
from bs4 import BeautifulSoup

####Writing a python python function for web scraper

def web_scraper(url):
    Male= open("Male.txt","w+")                           ### Defining a Male text file
    Female=open("Female.txt","w+")                        ### Defining a Female text file
    for i in range(0,50):
        get_req=get(url)                                  ####Making a request to the url
        soup = BeautifulSoup(get_req.text, "html.parser")  
        links = soup.select("center p")                   ###Extracting the html tags Center,p
        texts=links[0].get_text()                         ####Getting the texts from the tag
        list_split=texts.split("She")                     ####Splitting the
        Male.write(list_split[0]+"\n")                    ####Extracting the first part and writing to male text file
        Female_split=('She'+list_split[1]).split("They")     ###Spliting the file on . 
        Female.write(Female_split[0]+"\n")                ###Extracting the first part and writing to the female text file

    Male.close()  
    Female.close()                                        ###Closing the file

###calling the function

if __name__=="__main__":
    web_scraper("https://theyfightcrime.org/")

