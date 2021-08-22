import re
import json
import rich
import requests
import pandas as pd
from tqdm import tqdm
from retry import retry
from functools import wraps
from jsonpath import jsonpath
from collections import namedtuple
from typing import Dict, List, Union


def to_numeric(func):
    """
    将 DataFrame 或者 Series 尽可能地转为数字的装饰器
    Parameters
    ----------
    func : Callable
        返回结果为 DataFrame 或者 Series 的函数
    Returns
    -------
    Union[DataFrame, Series]
    """

    ignore = ['股票代码', '基金代码', '代码', '市场类型', '市场编号', '债券代码']

    @wraps(func)
    def run(*args, **kwargs):
        values = func(*args, **kwargs)
        if isinstance(values, pd.DataFrame):
            for column in values.columns:
                if column not in ignore:

                    values[column] = values[column].apply(convert)
        elif isinstance(values, pd.Series):
            for index in values.index:
                if index not in ignore:

                    values[index] = convert(values[index])
        return values

    def convert(o: Union[str, int, float]) -> Union[str, float, int]:
        if not re.findall('\d', str(o)):
            return o
        try:
            if str(o).isalnum():
                o = int(o)
            else:
                o = float(o)
        except:
            pass
        return o
    return run



class BasicSpider:
    def __init__(self) -> None:

        self.session = requests.Session()
        # 存储证券代码的实体
        self.Quote = namedtuple(
            'Quote', [
                'code', 'name', 'pinyin', 'id', 'jys', 'classify', 'market_type','security_typeName', 
                'security_type', 'mkt_num', 'type_us', 'quote_id', 'unified_code', 'inner_code'
                ])

        # 请求头
        self.EASTMONEY_REQUEST_HEADERS = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 6.3; WOW64; Trident/7.0; Touch; rv:11.0) like Gecko',
            'Accept': '*/*',
            'Accept-Language': 'zh-CN,zh;q=0.8,zh-TW;q=0.7,zh-HK;q=0.5,en-US;q=0.3,en;q=0.2',
            'Referer': 'http://quote.eastmoney.com/center/gridlist.html',
        }
        # 股票、ETF、债券 K 线表头
        self.EASTMONEY_KLINE_FIELDS = {
            'f51': '日期',
            'f52': '开盘',
            'f53': '收盘',
            'f54': '最高',
            'f55': '最低',
            'f56': '成交量',
            'f57': '成交额',
            'f58': '振幅',
            'f59': '涨跌幅',
            'f60': '涨跌额',
            'f61': '换手率',


        }



    @retry(tries=3, delay=1)
    def get_quote_id(self,stock_code: str) -> str:
        """
        生成东方财富股票专用的行情ID
        Parameters
        ----------
        stock_code : str
            证券代码或者证券名称
        Returns
        -------
        str
            东方财富股票专用的 secid
        """
        if len(str(stock_code).strip()) == 0:
            raise Exception('证券代码应为长度不应为 0')
        quote = self.search_quote(stock_code)
        if isinstance(quote, self.Quote):
            return quote.quote_id
        if quote is None:
            rich.print(f'证券代码 "{stock_code}" 可能有误')
            return ''

    ## TODO 返回值需要控制一下
    # def search_quote(self,keyword: str,count: int = 1) -> Union[self.Quote, None, List[self.Quote]]:
    def search_quote(self,keyword: str,count: int = 1):
        """
        根据关键词搜索以获取证券信息
        Parameters
        ----------
        keyword : str
            搜索词(股票代码、债券代码甚至证券名称都可以)
        count : int, optional
            最多搜索结果数, 默认为 `1`
        Returns
        -------
        Union[Quote, None, List[Quote]]
        """
        # quote = search_quote_locally(keyword)
        # if count == 1 and quote:
        #     return quote
        url = 'https://searchapi.eastmoney.com/api/suggest/get'
        params = (
            ('input', f'{keyword}'),
            ('type', '14'),
            ('token', 'D43BF722C8E33BDC906FB84D85E326E8'),
            ('count', f'{count}'))
        json_response = self.session.get(url, params=params).json()
        items = json_response['QuotationCodeTable']['Data']
        if items is not None:
            quotes = [self.Quote(*item.values()) for item in items]
            # save_search_result(keyword, quotes)
            if count == 1:
                return quotes[0]
            return quotes
        return None



    @to_numeric
    def get_quote_history_single(self,code: str,beg: str = '19000101',end: str = '20500101',klt: int = 101,fqt: int = 1) -> pd.DataFrame:
        """
        获取单只股票、债券 K 线数据

        """

        fields = list(self.EASTMONEY_KLINE_FIELDS.keys())
        columns = list(self.EASTMONEY_KLINE_FIELDS.values())
        fields2 = ",".join(fields)
        quote_id = self.get_quote_id(code)
        params = (
            ('fields1', 'f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11,f12,f13'),
            ('fields2', fields2),
            ('beg', beg),
            ('end', end),
            ('rtntype', '6'),
            ('secid', quote_id),
            ('klt', f'{klt}'),
            ('fqt', f'{fqt}'),
        )

        url = 'https://push2his.eastmoney.com/api/qt/stock/kline/get'

        json_response = self.session.get(url, headers=self.EASTMONEY_REQUEST_HEADERS, params=params).json()
        klines: List[str] = jsonpath(json_response, '$..klines[:]')
        if not klines:
            columns.insert(0, '代码')
            columns.insert(0, '名称')
            return pd.DataFrame(columns=columns)

        rows = [kline.split(',') for kline in klines]
        name = json_response['data']['name']
        code = quote_id.split('.')[-1]
        df = pd.DataFrame(rows, columns=columns)
        df.insert(0, '代码', code)
        df.insert(0, '名称', name)

        return df
if __name__ == "__main__":
    a = BasicSpider()
    # 股票代码列表
    # stock_codes = ['600519', '300750']
    stock_code = '600519'
    # 开始日期
    beg = '20210101'
    # 结束日期
    end = '20210708'
    # 获取股票日 K 数据

    result = a.get_quote_history_single(stock_code, beg=beg, end=end)
    print(result)