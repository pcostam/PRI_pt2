# -*- coding: utf-8 -*-
"""
Exercise-4
"""

from bs4 import BeautifulSoup 
import re
import matplotlib.pyplot as plt
from urllib.request import urlopen
from xml.etree.ElementTree import parse
import itertools
from wordcloud import WordCloud, ImageColorGenerator
try2 = __import__('try2')
try3 = __import__('try3')

def main():
    files = dict()
    docs = dict()
    
    train_set, test_set = try2.get_dataset("test", t="word", stem_or_not_stem = "not stem")
    doc = urlopen("https://archive.nytimes.com/www.nytimes.com/services/xml/rss/index.html").read()
    links_xml_dict = extract(doc)
    
    for key, value in links_xml_dict.items():
        if key not in files.keys():
            files[key] = [urlopen(el) for el in value]
        else:
            files[key] += [urlopen(el) for el in value]


    for key, value in files.items():
        for f in value:
            xmldoc = parse(f)
            #print(xmldoc)
            for item in xmldoc.iterfind('channel/item'):
                title = item.findtext('title')
                desc = item.findtext('description')
                doc = title + " " + desc
                
                if key not in docs:
                    docs[key] = [[doc]]
                   
                else: 
                    docs[key].append([doc])
                  
                  
    #print(docs['Business'])
    RRF_dict_results = dict()
    
    f = open('keyphrasesResults.html','w')
    vectorizer, vectorizer_tf, bm25 = try3.do_train(train_set)
    idx = 0
    
    for cat, cat_docs in docs.items():
        
        text = ' '.join(list(itertools.chain.from_iterable(cat_docs)))
        RRF_sorted, CombSum_sorted, CombMNZ_sorted = try3.do_score([text], vectorizer, vectorizer_tf, bm25)
        if idx == 5:
            break
        idx += 1
        
        res = dict(zip([i[0] for i in RRF_sorted], [i[1] for i in RRF_sorted[:10]]))
        
        #BAR CHART
        plt.figure()
        plt.bar(range(len(res)), list(res.values()), align='center')
        plt.xticks(range(len(res)), list(res.keys()))
        plt.savefig('fig'+ str(idx) + '.png',format='png')
        #plt.show()
        
        #WORD CLOUD
        generated_text = ""
        for key, val in res.items():
            occur = int(val*100)
            generated_text += " " +  key + " " * occur

        wordcloud = WordCloud().generate(generated_text)
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        #plt.show()
        
        #LOLLIPOP CHART
        #plt.stem(list(res.keys())[:5], list(res.values())[:5])
        #plt.ylim(0, 0.8)
        #plt.show()

#    message = """<html>
#    <head>Exercise 4 - </head>
#    <img src="fig.png">
#    </html>"""
#    
#    f.write(message)
#    f.close()   

       
def extract(content):
    links = list()
    cat_dict = dict()
    soup = BeautifulSoup(content, 'lxml')
    for tag in soup.find_all():
        if tag.name == 'div' and 'class' in tag.attrs.keys() and tag.attrs['class'] == ['columnGroup', 'singleRule']:
            category = ''
                
        if tag.name == 'span' and 'class' in tag.attrs.keys() and tag.attrs['class']==['rssRow' ]:
                category = tag.get_text()
                #print("category", category)
                
        if tag.name == 'a' and 'href' in tag.attrs:
                link = tag.attrs['href']
                
                if(re.match(".+\.xml{1}", link)): 
                    if category not in cat_dict.keys():
                        cat_dict[category] = [link]
                    else:
                        cat_dict[category] += [link]
                        links.append(link)
    return cat_dict