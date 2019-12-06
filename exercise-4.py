# -*- coding: utf-8 -*-
"""Exercise-4""" 
from bs4 import BeautifulSoup
from urllib.request import urlopen
from xml.etree.ElementTree import parse
from wordcloud import WordCloud
from math import pi
from sklearn.datasets import fetch_20newsgroups
import re
import matplotlib.pyplot as plt
import itertools
import pandas as pd

try2 = __import__('try2')
try3 = __import__('try3')

def main():
    files = dict()
    docs = dict()
    
    #train_set, test_set = try2.get_dataset("test", t="word", stem_or_not_stem = "not stem")
    train_set_aux = get_20_news_group(600)
    dictOfWords = { i : train_set_aux[i] for i in range(0, len(train_set_aux) ) }
    train_set = dictOfWords.values()
    #print(type(train_set))
    #print(train_set)

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
    
    f = open('keyphrasesResults.html','w')
    message = """<html>
    <style>
    html, body, h1, h2, h3, h4, h5, h6 {
            font-family: "Lucida Console", cursive, sans-serif;
    }
    </style>
    <h1>Exercise 4 - A practical application</h1>
    <body>
    """
    vectorizer, vectorizer_tf, bm25 = try3.do_train(train_set)
    idx = 0

    for cat, cat_docs in docs.items():
        text = ' '.join(list(itertools.chain.from_iterable(cat_docs)))
        RRF_sorted, CombSum_sorted, CombMNZ_sorted = try3.do_score(train_set, [text], vectorizer, vectorizer_tf, bm25, combination_features = ('tf','tfidf', 'idf', 'bm25'))
        
        res = dict(zip([i[0] for i in RRF_sorted], [i[1] for i in RRF_sorted[:5]]))
    
        print("res: ", res)
        
        #BAR CHART -- SO RRF
        plt.figure()
        plt.bar(range(len(res)), list(res.values()), align='center')
        plt.xticks(range(len(res)), list(res.keys()))
        plt.savefig('bar_chart'+ 'cat' + str(cat) + '.png',format='png')
        message += """<h3>"""+ str(cat) + """</h3>"""
        message += """<img src=bar_chart"""  + """cat""" + str(cat) +""".png>""" 
        plt.show()
        
        #WORD CLOUD -- SO RRF
        generated_text = ""
        for key, val in res.items():
            occur = int(val*100)
            generated_text += " " +  key + " " * occur
        
        wordcloud = WordCloud().generate(generated_text)
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        plt.savefig('word_cloud'+ 'cat' + str(cat) + '.png',format='png')
        #message += """<h3>"""+ str(cat) + """</h3>"""
        message += """<img src=word_cloud"""  + """cat""" + str(cat) +""".png>"""
        plt.show()
        
        #SPYDER CHART
        keyphrases = [i[0] for i in CombSum_sorted[:5]]
        
        RRF_df = list()
        CombSum_df = list()
        CombMNZ_df = list()
        
        RRF_dict =  dict(zip([i[0] for i in RRF_sorted], [i[1] for i in RRF_sorted]))
        CombSum_dict = dict(zip([i[0] for i in CombSum_sorted], [i[1] for i in CombSum_sorted]))
        CombMNZ_dict = dict(zip([i[0] for i in CombMNZ_sorted], [i[1] for i in CombMNZ_sorted]))
        
        for kp in keyphrases:
            RRF_df.append(RRF_dict[kp])
            CombSum_df.append(CombSum_dict[kp])
            CombMNZ_df.append(CombMNZ_dict[kp])
        
        df = pd.DataFrame({
        'group': keyphrases,
        'RRF': RRF_df,
        'CombSum': CombSum_df,
        'CombMNZ': CombMNZ_df
       
        })
        
        # number of variable
        categories=list(df)[1:]
        N = len(categories)
        # We are going to plot the first line of the data frame.
        # But we need to repeat the first value to close the circular graph:
        values=df.loc[0].drop('group').values.flatten().tolist()
        #print("values, ", values)
        values += values[:1]
        #Sprint("values, ", values)
        
        # What will be the angle of each axis in the plot? (we divide the plot / number of variable)
        angles = [n / float(N) * 2 * pi for n in range(N)]
        angles += angles[:1]
        # Initialise the spider plot
        ax = plt.subplot(111, polar=True)
        # Draw one axe per variable + add labels labels yet
        plt.xticks(angles[:-1], categories, color='grey', size=8)
        # Draw ylabels
        ax.set_rlabel_position(0)
        plt.yticks([10,20,30], ["0.01","0.05","0.1"], color="grey", size=7)
        plt.ylim(0,35)
        # Plot data
        ax.plot(angles, values, linewidth=1, linestyle='solid')
        # Fill area
        ax.fill(angles, values, 'b', alpha=0.1)
        plt.savefig('spyder_chart_'+ 'cat' + str(cat) + '.png',format='png')
        #message += """<h3>"""+ str(cat) + """</h3>"""
        message += """<img src=spyder_chart_"""  + """cat""" + str(cat) +""".png>
        <br>"""

        idx+=1
    message += """</body>
    </html>"""
    f.write(message)
    f.close()   

       
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

def get_20_news_group(size):
    docs = fetch_20newsgroups(subset = 'train') 
    
    return docs.data[:size + 1]
