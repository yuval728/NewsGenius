from bs4 import BeautifulSoup
import requests

def crawler(url):
    if url is None:
        return None
    try:
        soup = BeautifulSoup(requests.get(url).text, 'html.parser')
    except:
        return None
    # main=soup.find('main')
    news_title=soup.find('h1').text
    # news_subtitle=soup.find('h2').text
    news_content_box=soup.find('article')
    
    news_content=' '.join([p.text for p in news_content_box.find_all('p')])
    
    # print(news_title)
    # print(news_subtitle)
    # print()
    # print(news_content)
    return {
        'title': news_title,
        # 'subtitle': news_subtitle,
        'content': news_content
    }