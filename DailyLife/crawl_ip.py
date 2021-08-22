
#!/usr/bin/env python
# encoding: utf-8
"""
@author: yousheng
@contact: 1197993367@qq.com
@site: http://youyuge.cn

@version: 1.0
@license: Apache Licence
@file: crawl_ip.py
@time: 17/9/27 下午3:06

"""

import requests #用requests库来做简单的网络请求
from scrapy.selector import Selector
#从scrapy的settings中导入数据库配置



def crawl_xici_ip(pages):
    '''
    爬取一定页数上的所有代理ip,每爬完一页，就存入数据库
    :return:
    '''
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 6.1; WOW64; rv:52.0) Gecko/20100101 Firefox/52.0"}
    for i in range(1, pages):
        response = requests.get(url='http://www.xicidaili.com/nn/{0}'.format(i), headers=headers)

        all_trs = Selector(text=response.text).css('#ip_list tr')

        ip_list = []
        for tr in all_trs[1:]:
            ip = tr.xpath('td[2]/text()').extract_first().encode('utf8')
            port = tr.xpath('td[3]/text()').extract_first().encode('utf8')
            ip_type = tr.xpath('td[6]/text()').extract_first().encode('utf8')
            ip_speed = tr.xpath('td[7]/div/@title').extract_first()
            if ip_speed:
                ip_speed = float(ip_speed.split(u'秒')[0])
            ip_alive = tr.xpath('td[9]/text()').extract_first().encode('utf8')

            ip_list.append((ip, port, ip_type, ip_speed, ip_alive))

        # 每页提取完后就存入数据库
        print(ip_list)

'''
# ip的管理类
class IPUtil(object):
    # noinspection SqlDialectInspection
    def get_random_ip(self):
        # 从数据库中随机获取一个可用的ip
        random_sql = """
              SELECT ip, port, type FROM proxy_ip
            ORDER BY RAND()
            LIMIT 1
            """

        result = cursor.execute(random_sql)
        for ip_info in cursor.fetchall():
            ip = ip_info[0]
            port = ip_info[1]
            ip_type = ip_info[2]

            judge_re = self.judge_ip(ip, port, ip_type)
            if judge_re:
                return "{2}://{0}:{1}".format(ip, port, str(ip_type).lower())
            else:
                return self.get_random_ip()

    def judge_ip(self, ip, port, ip_type):
        # 判断ip是否可用，如果通过代理ip访问百度，返回code200则说明可用
        # 若不可用则从数据库中删除
        print('begin judging ---->', ip, port, ip_type)
        http_url = "https://www.baidu.com"
        proxy_url = "{2}://{0}:{1}".format(ip, port, str(ip_type).lower())
        try:
            proxy_dict = {
                "http": proxy_url,
            }
            response = requests.get(http_url, proxies=proxy_dict)
        except Exception as e:
            print(ip,"invalid ip and port,cannot connect baidu")
            self.delete_ip(ip)
            return False
        else:
            code = response.status_code
            if code >= 200 and code < 300:
                print(ip,"effective ip")
                return True
            else:
                print(ip,"invalid ip and port,code is " + code)
                self.delete_ip(ip)
                return False

    # noinspection SqlDialectInspection
    def delete_ip(self, ip):
        # 从数据库中删除无效的ip
        delete_sql = """
            delete from proxy_ip where ip='{0}'
        """.format(ip)
        cursor.execute(delete_sql)
        conn.commit()
        return True
'''

if __name__ == '__main__':
    crawl_xici_ip(pages=3)
    # ip = IPUtil()
    # for i in range(20):
    #     print ip.get_random_ip()