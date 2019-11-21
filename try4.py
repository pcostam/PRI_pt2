# -*- coding: utf-8 -*-
"""
Exercise-4
"""

from bs4 import BeautifulSoup
import re
from urllib.request import urlopen
from xml.etree.ElementTree import parse

def main():
    doc = urlopen("https://archive.nytimes.com/www.nytimes.com/services/xml/rss/index.html").read()
    links_xml = extract(doc)
    files = [urlopen(el) for el in links_xml]
    docs = list()
    for f in files:
        xmldoc = parse(f)
        
        for item in xmldoc.iterfind('channel/item'):
            title = item.findtext('title')
            desc = item.findtext('description')
            doc = [title + " " + desc]
            docs.append(doc)
       

       
def extract(content):
    links = []
    soup = BeautifulSoup(content, 'lxml')
    for tag in soup.find_all():
        if tag.name == 'a' and 'href' in tag.attrs:
            link = tag.attrs['href']
            if(re.match(".+\.xml{1}", link)): 
                    links.append(link)
    return links