from Spider.basic_spider import to_numeric,process_dataframe_and_series,BasicSpider
from datetime import date
import re
import json
import rich
import requests
import multitasking
import pandas as pd
from tqdm import tqdm
from retry import retry
from functools import wraps
from jsonpath import jsonpath
from collections import namedtuple
from collections import defaultdict
from typing import Dict, List, Union, Callable
class QuotesSpider(BasicSpider):
    def __init__(self):
        super(QuotesSpider, self).__init__()



    @to_numeric
    def get_quote_history_single(self, code: str,
                                 beg: str = '19000101',
                                 end: str = '20500101',
                                 klt: int = 101,
                                 fqt: int = 1,
                                 **kwargs) -> pd.DataFrame:
        """
        获取单只股票、债券 K 线数据
        """

        fields = list(self.EASTMONEY_KLINE_FIELDS.keys())
        columns = list(self.EASTMONEY_KLINE_FIELDS.values())
        fields2 = ",".join(fields)
        if kwargs.get('quote_id_mode'):
            quote_id = code
        else:
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

        json_response = self.session.get(
            url, headers=self.EASTMONEY_REQUEST_HEADERS, params=params).json()
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

    def get_quote_history_multi(self, codes: List[str],
                                beg: str = '19000101',
                                end: str = '20500101',
                                klt: int = 101,
                                fqt: int = 1,
                                tries: int = 3,
                                **kwargs
                                ) -> Dict[str, pd.DataFrame]:
        """
        获取多只股票、债券历史行情信息
        """

        dfs: Dict[str, pd.DataFrame] = {}
        total = len(codes)

        @multitasking.task
        @retry(tries=tries, delay=1)
        def start(code: str):
            _df = self.get_quote_history_single(
                code,
                beg=beg,
                end=end,
                klt=klt,
                fqt=fqt,
                **kwargs)
            dfs[code] = _df
            pbar.update(1)
            pbar.set_description_str(f'Processing => {code}')

        pbar = tqdm(total=total)
        for code in codes:
            start(code)
        multitasking.wait_for_tasks()
        pbar.close()
        return dfs

    def get_quote_history(self, codes: Union[str, List[str]],
                          beg: str = '19000101',
                          end: str = '20500101',
                          klt: int = 101,
                          fqt: int = 1,
                          **kwargs) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
        """
        获取股票、ETF、债券的 K 线数据
        Parameters
        ----------
        codes : Union[str,List[str]]
            股票、债券代码 或者 代码构成的列表
        beg : str, optional
            开始日期，默认为 ``'19000101'`` ，表示 1900年1月1日
        end : str, optional
            结束日期，默认为 ``'20500101'`` ，表示 2050年1月1日
        klt : int, optional
            行情之间的时间间隔，默认为 ``101`` ，可选示例如下
            - ``1`` : 分钟
            - ``5`` : 5 分钟
            - ``15`` : 15 分钟
            - ``30`` : 30 分钟
            - ``60`` : 60 分钟
            - ``101`` : 日
            - ``102`` : 周
            - ``103`` : 月
        fqt : int, optional
            复权方式，默认为 ``1`` ，可选示例如下
            - ``0`` : 不复权
            - ``1`` : 前复权
            - ``2`` : 后复权
        Returns
        -------
        Union[DataFrame, Dict[str, DataFrame]]
            股票、债券的 K 线数据
            - ``DataFrame`` : 当 ``codes`` 是 ``str`` 时
            - ``Dict[str, DataFrame]`` : 当 ``codes`` 是 ``List[str]`` 时
        """

        if isinstance(codes, str):
            return self.get_quote_history_single(codes,
                                                 beg=beg,
                                                 end=end,
                                                 klt=klt,
                                                 fqt=fqt,
                                                 **kwargs)

        elif hasattr(codes, '__iter__'):
            codes = list(codes)
            return self.get_quote_history_multi(codes,
                                                beg=beg,
                                                end=end,
                                                klt=klt,
                                                fqt=fqt,
                                                **kwargs)
        raise TypeError(
            '代码数据类型输入不正确！'
        )

    @to_numeric
    def get_base_info_stock_single(self, stock_code: str) -> pd.Series:
        """
        获取单股票基本信息
        Parameters
        ----------
        stock_code : str
            股票代码
        Returns
        -------
        Series
            单只股票基本信息
        """
        fields = ",".join(self.EASTMONEY_STOCK_BASE_INFO_FIELDS.keys())
        params = (
            ('ut', 'fa5fd1943c7b386f172d6893dbfba10b'),
            ('invt', '2'),
            ('fltt', '2'),
            ('fields', fields),
            ('secid', self.get_quote_id(stock_code)),

        )
        url = 'http://push2.eastmoney.com/api/qt/stock/get'
        json_response = self.session.get(url,
                                         headers=self.EASTMONEY_REQUEST_HEADERS,
                                         params=params).json()

        s = pd.Series(json_response['data']).rename(
            index=self.EASTMONEY_STOCK_BASE_INFO_FIELDS)
        return s[self.EASTMONEY_STOCK_BASE_INFO_FIELDS.values()]

    def get_base_info_stock_muliti(self, stock_codes: List[str]) -> pd.DataFrame:
        """
        获取股票多只基本信息
        Parameters
        ----------
        stock_codes : List[str]
            股票代码列表
        Returns
        -------
        DataFrame
            多只股票基本信息
        """

        @multitasking.task
        @retry(tries=3, delay=1)
        def start(stock_code: str):
            s = self.get_base_info_stock_single(stock_code)
            dfs.append(s)
            pbar.update()
            pbar.set_description(f'Processing => {stock_code}')

        dfs: List[pd.DataFrame] = []
        pbar = tqdm(total=len(stock_codes))
        for stock_code in stock_codes:
            start(stock_code)
        multitasking.wait_for_tasks()
        df = pd.DataFrame(dfs)
        return df

    @to_numeric
    def get_base_info_stock(self, stock_codes: Union[str, List[str]]) -> Union[pd.Series, pd.DataFrame]:
        """
        Parameters
        ----------
        stock_codes : Union[str, List[str]]
            股票代码或股票代码构成的列表
        Returns
        -------
        Union[Series, DataFrame]
            - ``Series`` : 包含单只股票基本信息(当 ``stock_codes`` 是字符串时)
            - ``DataFrane`` : 包含多只股票基本信息(当 ``stock_codes`` 是字符串列表时)
        Raises
        ------
        TypeError
            当 ``stock_codes`` 类型不符合要求时
        Examples
        --------
        
        >>> # 获取单只股票信息
        >>> self.get_base_info('600519')
        股票代码                  600519
        股票名称                    贵州茅台
        市盈率(动)                 39.38
        市净率                    12.54
        所处行业                    酿酒行业
        总市值          2198082348462.0
        流通市值         2198082348462.0
        板块编号                  BK0477
        ROE                     8.29
        净利率                  54.1678
        净利润       13954462085.610001
        毛利率                  91.6763
        dtype: object
        >>> # 获取多只股票信息
        >>> self.get_base_info(['600519','300715'])
            股票代码  股票名称  市盈率(动)    市净率  所处行业           总市值          流通市值    板块编号   ROE      净利率           净利润      毛利率
        0  300715  凯伦股份   42.29   3.12  水泥建材  9.160864e+09  6.397043e+09  BK0424  3.97  12.1659  5.415488e+07  32.8765
        1  600519  贵州茅台   39.38  12.54  酿酒行业  2.198082e+12  2.198082e+12  BK0477  8.29  54.1678  1.395446e+10  91.6763
        """

        if isinstance(stock_codes, str):
            return self.get_base_info_stock_single(stock_codes)
        elif hasattr(stock_codes, '__iter__'):
            return self.get_base_info_stock_muliti(stock_codes)

        raise TypeError(f'所给的 {stock_codes} 不符合参数要求')

    @to_numeric
    def get_realtime_quotes_by_fs(self, fs: str) -> pd.DataFrame:
        """
        获取沪深市场最新行情总体情况
        Returns
        -------
        DataFrame
            沪深市场最新行情信息（涨跌幅、换手率等信息）
        """

        fields = ",".join(self.EASTMONEY_QUOTE_FIELDS.keys())
        params = (
            ('pn', '1'),
            ('pz', '1000000'),
            ('po', '1'),
            ('np', '1'),
            ('fltt', '2'),
            ('invt', '2'),
            ('fid', 'f3'),
            ('fs', fs),
            ('fields', fields)
        )
        url = 'http://push2.eastmoney.com/api/qt/clist/get'
        json_response = self.session.get(url, headers=self.EASTMONEY_REQUEST_HEADERS, params=params).json()
        df = pd.DataFrame(json_response['data']['diff'])
        df = df.rename(columns=self.EASTMONEY_QUOTE_FIELDS)
        df = df[self.EASTMONEY_QUOTE_FIELDS.values()]
        df['行情ID'] = df['市场编号'].astype(str) + '.' + df['代码'].astype('str')
        df['市场类型'] = df['市场编号'].astype(str).apply(lambda x: self.MARKET_NUMBER_DICT.get(x))

        return df

    @process_dataframe_and_series(remove_columns_and_indexes=['市场编号'])
    @to_numeric
    def get_realtime_quotes(self) -> pd.DataFrame:
        """
        获取沪深市场最新行情总体情况
        Returns
        -------
        DataFrame
            沪深全市场A股上市公司的最新行情信息（涨跌幅、换手率等信息）
        Examples
        --------
        >>> self.get_realtime_quotes()
                股票代码   股票名称     涨跌幅     最新价      最高      最低      今开     涨跌额    换手率    量比    动态市盈率     成交量           成交额   昨日收盘           总市值         流通市值      行情ID 市场类型
        0     688787    N海天  277.59  139.48  172.39  139.25  171.66  102.54  85.62     -    78.93   74519  1110318832.0  36.94    5969744000   1213908667  1.688787   沪A
        1     301045    N天禄  149.34   39.42   48.95    39.2   48.95   23.61  66.66     -    37.81  163061   683878656.0  15.81    4066344240    964237089  0.301045   深A
        2     300532   今天国际   20.04   12.16   12.16   10.69   10.69    2.03   8.85  3.02   -22.72  144795   171535181.0  10.13    3322510580   1989333440  0.300532   深A
        3     300600   国瑞科技   20.02   13.19   13.19   11.11   11.41     2.2  18.61  2.82   218.75  423779   541164432.0  10.99    3915421427   3003665117  0.300600   深A
        4     300985   致远新能   20.01   47.08   47.08    36.8    39.4    7.85  66.65  2.17    58.37  210697   897370992.0  39.23    6277336472   1488300116  0.300985   深A
        ...      ...    ...     ...     ...     ...     ...     ...     ...    ...   ...      ...     ...           ...    ...           ...          ...       ...  ...
        4598  603186   华正新材   -10.0   43.27   44.09   43.27   43.99   -4.81   1.98  0.48    25.24   27697   120486294.0  48.08    6146300650   6063519472  1.603186   沪A
        4599  688185  康希诺-U  -10.11   476.4  534.94  460.13   530.0   -53.6   6.02  2.74 -2088.07   40239  1960540832.0  530.0  117885131884  31831479215  1.688185   沪A
        4600  688148   芳源股份  -10.57    31.3   34.39    31.3    33.9    -3.7  26.07  0.56   220.01  188415   620632512.0   35.0   15923562000   2261706043  1.688148   沪A
        4601  300034   钢研高纳  -10.96   43.12   46.81   42.88    46.5   -5.31   7.45  1.77    59.49  323226  1441101824.0  48.43   20959281094  18706911861  0.300034   深A
        4602  300712   永福股份  -13.71    96.9  110.94    95.4   109.0   -15.4   6.96  1.26   511.21  126705  1265152928.0  112.3   17645877600  17645877600  0.300712   深A
        """
        print(1)
        fs = self.FS_DICT['stock']
        df = self.get_realtime_quotes_by_fs(fs)
        df.rename(columns={
            '代码': '股票代码',
            '名称': '股票名称'}, inplace=True)

        return df




if __name__ == "__main__":
    a = FundingsSpider()
    # ******************测试基金***********************
    print('获取股票K线')
    result = a.get_quote_history('000799')
    print(result)
    print('获取股票基本信息')
    result = a.get_base_info_stock(['000799','300715'])
    print(result)
    print('获取沪深市场最新行情总体情况')
    result = a.get_realtime_quotes()
    print(result)