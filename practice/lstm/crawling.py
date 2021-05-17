from bs4 import BeautifulSoup
import requests


def week_parse(code, count):
    url = "https://fchart.stock.naver.com/sise.nhn?symbol={0}&timeframe=day&count={1}&requestType=0".format(code, count)
    res = requests.get(url)
    charset = res.encoding
    decode = res.content.decode(charset)
    soup = BeautifulSoup(decode, 'html.parser')
    data = soup.find_all('item')
    result = {}
    Date = []
    Opne = []
    Close = []
    High = []
    Low = []
    Volume = []
    for i in range(int(count)):
        data_list = data[i]['data'].split('|')
        Date.append(data_list[0])
        Opne.append(data_list[1])
        High.append(data_list[2])
        Low.append(data_list[3])
        Close.append(data_list[4])
        Volume.append(data_list[5])
    result['Date'] = Date
    result['Open'] = Opne
    result['High'] = High
    result['Low'] = Low
    result['Close'] = Close
    result['Volume'] = Volume
    return result


def day_parse(code):
    stock = {}
    url = 'https://finance.naver.com/item/main.nhn?code={0}'.format(code)
    res = requests.get(url)
    charset = res.encoding
    decode = res.content.decode(charset)
    soup = BeautifulSoup(decode, 'html.parser')
    result = soup.select(".today")[0].get_text()
    result = result.replace('\n', '')
    result2 = soup.select(".no_info")[0].get_text()
    result2 = result2.replace('\n', '')
    result3 = soup.select(".wrap_company")[0].get_text()
    result3 = result3.replace('\n', '')
    current_end_idx = int(result.find('전일')/2)
    current = result[:current_end_idx]
    prev_start_idx = result2.find('전일') + 2
    prev_end_idx = int((result2.find('고가') - prev_start_idx)/2 + prev_start_idx)
    prev = result2[prev_start_idx:prev_end_idx]
    high_start_idx = result2.find('고가') + 2
    high_end_idx = int((result2.find('(상') - high_start_idx)/2 + high_start_idx)
    high = result2[high_start_idx:high_end_idx]
    volume_start_idx = result2.find('거래량') + 3
    volume_end_idx = int((result2.find('시가') - volume_start_idx)/2 + volume_start_idx)
    volume = result2[volume_start_idx:volume_end_idx]
    start_start_idx = result2.find('시가') + 2
    start_end_idx = int((result2.find('저가') - start_start_idx)/2 + start_start_idx)
    start = result2[start_start_idx:start_end_idx]
    low_start_idx = result2.find('저가') + 2
    low_end_idx = int((result2.find('(하') - low_start_idx)/2 + low_start_idx)
    low = result2[low_start_idx:low_end_idx]
    stock['current'] = current
    stock['prev'] = prev
    stock['high'] = high
    stock['low'] = low
    stock['start'] = start
    stock['volume'] = volume
    if '하락' in result:
        stock['status'] = '하락'
        percent_start_idx = result.find('-') + 1
        percent_end_idx = int((result.find('%')-percent_start_idx)/2 +percent_start_idx)
        percent = result[percent_start_idx:percent_end_idx]
        stock['percent'] = percent
    else:
        stock['status'] = '상승'
        percent_start_idx = result.find('+') + 1
        percent_end_idx = int((result.find('%')-percent_start_idx)/2 +percent_start_idx)
        percent = result[percent_start_idx:percent_end_idx]
        stock['percent'] = percent
    name_end_idx = result3.find(code)
    stock['name'] = result3[:name_end_idx]
    return stock