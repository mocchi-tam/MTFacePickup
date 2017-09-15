import time
import hashlib
import urllib
import os
from urllib.parse import urlparse
from urllib.request import urlopen
from urllib.error import URLError, HTTPError

NAMES=[
'浅井七海',
'稲垣香織',
'梅本和泉',
'黒須遥香',
'佐藤美波',
'庄司なぎさ',
'鈴木くるみ',
'田口愛佳',
'田屋美咲',
'長友彩海',
'野口菜々美',
'播磨七海',
'本間麻衣',
'前田彩佳',
'道枝咲',
'武藤小麟',
'安田叶',
'山内瑞葵',
'山根涼羽'
]

########### 設定パラメータ ###########

#検索ワード
#word = '志田未来'
#ダウンロード数
imageNum = 100 #最大値100
USER_AGENT = 'Mozilla/5.0 (X11; Linux i686) AppleWebKit/537.17 (KHTML, like Gecko) Chrome/24.0.1312.27 Safari/537.17'
########### End ###########

#ダウンロード
def download_page(url):
    import urllib.request
    try:
        headers = {}
        headers['User-Agent'] = "Mozilla/5.0 (Windows NT 6.1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/41.0.2228.0 Safari/537.36"
        req = urllib.request.Request(url, headers=headers)
        resp = urllib.request.urlopen(req)
        respData = str(resp.read())
        return respData
    except Exception as e:
        print(str(e))

#検索
def _images_get_next_item(s):
    start_line = s.find('rg_di')
    if start_line == -1:
        end_quote = 0
        link = "no_links"
        return link, end_quote
    else:
        start_line = s.find('"class="rg_meta"')
        start_content = s.find('"ou"', start_line+1)
        end_content = s.find(',"ow"', start_content+1)
        content_raw = str(s[start_content+6:end_content-1])
        return content_raw, end_content

#リンク取得
def _images_get_all_items(page):
    _items = []
    while True:
        item, end_content = _images_get_next_item(page)
        if item == "no_links":
            break
        else:
            _items.append(item)
            time.sleep(0.05)
            page = page[end_content:]
    return _items

def main():
    ############## Main Program ############
    T0 = time.time()   #開始時間
    
    for word in NAMES:
        #画像リンク取得
        temp = word
        items = []
        
        search = temp.replace(" ", "%20")
        
        print("検索ワード:" + search)
        URL = 'https://www.google.com/search?q=' + search + '&espv=2&biw=1366&bih=667&site=webhp&source=lnms&tbm=isch&sa=X&ei=XosDVaCXD8TasATItgE&ved=0CAcQ_AUoAg'
        p = urlparse(URL)
        query = urllib.parse.quote_plus(p.query, safe='=&')
        URL = '{}://{}{}{}{}{}{}{}{}'.format(p.scheme, p.netloc, p.path,';' if p.params else '', p.params,'?' if p.query else '', query,'#' if p.fragment else '', p.fragment)
        RAW_HTML = (download_page(URL))
        time.sleep(0.05)
        items.extend(_images_get_all_items(RAW_HTML))
        
        print("ダウンロード開始")
        
        errorCount = 0
        Cnt = 0
        folderName = os.getcwd() + '/data/' + word
        if os.path.exists(folderName)==False:
            os.mkdir(folderName)
        for item in items:
            if Cnt == imageNum:
                break
            try:
                outputPath = folderName + '/' + hashlib.md5(item.encode('utf-8')).hexdigest() + ".jpg"
                print('-------------------------------------------------------------------')
                if os.path.isfile(outputPath):
                    print(outputPath + " ダウンロード済み画像のためスキップ")
                else:
                    REQ = urllib.request.Request(item, headers={"User-Agent": USER_AGENT})
                    RESPONSE = urlopen(REQ)
                    DATA = RESPONSE.read()
                    open(outputPath, 'wb').write(DATA)
                    RESPONSE.close()
                    pass
        
                print("ダウンロード完了 ====> "+str(item))
        
            except IOError:
                errorCount += 1
                Cnt -= 1
                print("IOError"+str(item))
            except HTTPError as e:
                errorCount += 1
                Cnt -= 1
                print("HTTPError"+str(item))
            except URLError as e:
                errorCount += 1
                Cnt -= 1
                print("URLError "+str(item))
            except UnicodeEncodeError as e:
                errorCount += 1
                Cnt -= 1
                print("UnicodeEncodeError "+str(item))
            print('-------------------------------------------------------------------')
            Cnt += 1
        
        print("\n")
        print("ダウンロード完了")
        print("\n"+str(errorCount)+" ----> 合計エラー数")
    
if __name__ == '__main__':
    main()