from bs4 import BeautifulSoup
import markdown
import re
import urllib.parse
n_found = 0
known_subreddits = set()
def find_links(md):
    global n_found
    html = markdown.markdown(md)
    ast = BeautifulSoup(html)
    anchors = ast.find_all('a')

    
    referenced_subreddits = set()
    raw_text = ast.get_text(separator=' ')
    redditlinks = re.finditer('/(r|u)/([A-Za-z0-9_]{3,})',raw_text, re.I)
    redditgroups = [r.groups() for r in redditlinks]
    for (entity, name) in redditgroups:
        name = name.replace('/submit','').lower()
        if entity in {'r'}:
            known_subreddits.add(name)
            #referenced_subreddits.add(name)

    weblinks = re.finditer('https?://((?:[A-Za-z0-9\-]+\.)+)([A-Za-z0-9\-]+)/(\S*)', raw_text, re.I)
    webgroups = [r.groups() for r in weblinks]
    referenced_domains = set()
    for (domain, tld, path) in webgroups:
        referenced_domains.add(domain.replace('www.',''))

    for anchor in anchors:
        url = urllib.parse.urlparse(anchor['href'])
        domain = '.'.join(url.netloc.split('.')[:-1]).lower().replace('www.','')
        referenced_domains.add(domain)
        #print(domain)

    shittylinks = re.finditer('(?:\s+|^)((?:[A-Za-z0-9\-]+\.)+)(com|net|org|us|ca|de|fr|biz|info|name|co|io|eu|uk|ru|cn|app|in|es|mx|pt|br|ly|eu|ro|it|is|be|au|nz|hu|fi|se|za|nl|dk|pl|tk|kr|jp|tv)(?:/(\S*)|\s+|$)', raw_text, re.I)
    shittygroups = [r.groups() for r in shittylinks]

    if len(anchors) or len(redditgroups) or len(webgroups) or len(shittygroups):
        n_found += 1

    return md, referenced_subreddits, referenced_domains

