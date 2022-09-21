"""快代理代理IP使用教程"""

import requests
import random

# 要访问的目标网页
page_url = "http://dev.kdlapi.com/testproxy"

# 隧道的host与端口
proxy = "tps363.kdlapi.com:15818"

# 用户名和密码(隧道代理分配的)
username = ""
password = ""

# 代理IP的格式
proxies = {
     "http": "http://%(user)s:%(pwd)s@%(proxy)s/" % {'user': username, 'pwd': password, 'proxy': proxy},
     "https": "https://%(user)s:%(pwd)s@%(proxy)s/" % {'user': username, 'pwd': password, 'proxy': proxy
     }}

# 添加header，模拟用户请求
headers = {
    "Accept-Encoding": "Gzip",  # 使用gzip压缩传输数据让访问更快
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/79.0.3945.117 Safari/537.36"
}

# 发送request请求,打印响应code与body内容
r = requests.get(url=page_url, proxies=proxies, headers=headers)
print("response code",r.status_code)
print("response body",r.text)