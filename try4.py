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
    links_xml_dict = extract(doc)
    files = dict()
    docs = dict()
    for key, value in links_xml_dict.items():
        if key not in files.keys():
            files[key] = [urlopen(el) for el in value]
        else:
            files[key] += [urlopen(el) for el in value]
   

     
    for key, value in files.items():
        for f in value:
            xmldoc = parse(f)
            
            for item in xmldoc.iterfind('channel/item'):
                title = item.findtext('title')
                desc = item.findtext('description')
                doc = [title + " " + desc]
                docs[key] = doc
    print(docs)
       

       
def extract(content):
    links = list()
    cat_dict = dict()
    soup = BeautifulSoup(content, 'lxml')
    for tag in soup.find_all():
        if tag.name == 'div' and 'class' in tag.attrs.keys() and tag.attrs['class'] == ['columnGroup', 'singleRule']:
            category = ''
                
        if tag.name == 'span' and 'class' in tag.attrs.keys() and tag.attrs['class']==['rssRow' ]:
                category = tag.get_text()
                print("category", category)
                
        if tag.name == 'a' and 'href' in tag.attrs:
                link = tag.attrs['href']
                
                if(re.match(".+\.xml{1}", link)): 
                    if category not in cat_dict.keys():
                        cat_dict[category] = [link]
                    else:
                        cat_dict[category] += [link]
                        links.append(link)
                        
    return cat_dict