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


def process_dataframe_and_series(function_fields: Dict[str, Callable] = dict(),
                                 remove_columns_and_indexes: List[str] = list()):
    """
    对 DataFrame 和 Series 进一步操作
    Parameters
    ----------
    function_fields : Dict[str, Callable], optional
        函数字典
    remove_columns_and_indexes : List[str], optional
        需要删除的行或者列, by default list()
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            values = func(*args, **kwargs)
            if isinstance(values, pd.DataFrame):
                for column, function_name in function_fields.items():
                    if column not in values.columns:
                        continue
                    values[column] = values[column].apply(function_name)
                for column in remove_columns_and_indexes:
                    if column in values.columns:
                        del values[column]
            elif isinstance(values, pd.Series):
                for index in remove_columns_and_indexes:
                    values = values.drop(index)
            return values

        return wrapper

    return decorator


class BasicSpider:
    def __init__(self) -> None:

        self.session = requests.Session()
        # 存储证券代码的实体
        self.Quote = namedtuple(
            'Quote', [
                'code', 'name', 'pinyin', 'id', 'jys', 'classify', 'market_type', 'security_typeName',
                'security_type', 'mkt_num', 'type_us', 'quote_id', 'unified_code', 'inner_code'
            ])

        # 请求头
        self.EASTMONEY_REQUEST_HEADERS = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 6.3; WOW64; Trident/7.0; Touch; rv:11.0) like Gecko',
            'Accept': '*/*',
            'Accept-Language': 'zh-CN,zh;q=0.8,zh-TW;q=0.7,zh-HK;q=0.5,en-US;q=0.3,en;q=0.2',
            'Referer': 'http://quote.eastmoney.com/center/gridlist.html',
        }

        self.EastmoneyFundHeaders = {
            'User-Agent': 'EMProjJijin/6.2.8 (iPhone; iOS 13.6; Scale/2.00)',
            'GTOKEN': '98B423068C1F4DEF9842F82ADF08C5db',
            'clientInfo': 'ttjj-iPhone10,1-iOS-iOS13.6',
            'Content-Type': 'application/x-www-form-urlencoded',
            'Host': 'fundmobapi.eastmoney.com',
            'Referer': 'https://mpservice.com/516939c37bdb4ba2b1138c50cf69a2e1/release/pages/FundHistoryNetWorth',
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

        # 股票基本信息表头
        self.EASTMONEY_STOCK_BASE_INFO_FIELDS = {
            'f57': '股票代码',
            'f58': '股票名称',
            'f162': '市盈率(动)',
            'f167': '市净率',
            'f127': '所处行业',
            'f116': '总市值',
            'f117': '流通市值',
            'f198': '板块编号',
            'f173': 'ROE',
            'f187': '净利率',
            'f105': '净利润',
            'f186': '毛利率'

        }
        # 股票、债券榜单表头
        self.EASTMONEY_QUOTE_FIELDS = {
            'f12': '代码',
            'f14': '名称',
            'f3': '涨跌幅',
            'f2': '最新价',
            'f15': '最高',
            'f16': '最低',
            'f17': '今开',
            'f4': '涨跌额',
            'f8': '换手率',
            'f10': '量比',
            'f9': '动态市盈率',
            'f5': '成交量',
            'f6': '成交额',
            'f18': '昨日收盘',
            'f20': '总市值',
            'f21': '流通市值',
            'f13': '市场编号'
        }

        self.FS_DICT = {
            'bond': 'b:MK0354',
            'stock': 'm:0 t:6,m:0 t:80,m:1 t:2,m:1 t:23',
            'futures': 'm:113,m:114,m:115,m:8,m:142'
        }

        # 各个市场编号
        self.MARKET_NUMBER_DICT = {
            '0': '深A',
            '1': '沪A',
            '105': '美股',
            '116': '港股',
            '113': '上期所',
            '114': '大商所',
            '115': '郑商所',
            '8': '中金所',
            '142': '上海能源期货交易所',
            '155': '英股'

        }

    @retry(tries=3, delay=1)
    def get_quote_id(self, stock_code: str) -> str:
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
    def search_quote(self, keyword: str, count: int = 1):
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
    def get_quote_history_single(self, code: str, beg: str = '19000101', end: str = '20500101', klt: int = 101,
                                 fqt: int = 1) -> pd.DataFrame:
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

    @retry(tries=3)
    @to_numeric
    def get_quote_history(self, fund_code: str, pz: int = 40000) -> pd.DataFrame:
        """
        根据基金代码和要获取的页码抓取基金净值信息
        Parameters
        ----------
        fund_code : str
            6 位基金代码
        pz : int, optional
            页码, 默认为 40000 以获取全部历史数据
        Returns
        -------
        DataFrame
            包含基金历史净值等数据
        Examples
        --------

        >>> self.get_quote_history('161725')
            日期    单位净值    累计净值     涨跌幅
        0    2021-06-11  1.5188  3.1499   -3.09
        1    2021-06-10  1.5673  3.1984    1.69
        2    2021-06-09  1.5412  3.1723    0.11
        3    2021-06-08  1.5395  3.1706    -6.5
        4    2021-06-07  1.6466  3.2777    1.61
        ...         ...     ...     ...     ...
        1469 2015-06-08  1.0380  1.0380  2.5692
        1470 2015-06-05  1.0120  1.0120  1.5045
        1471 2015-06-04  0.9970  0.9970      --
        1472 2015-05-29  0.9950  0.9950      --
        1473 2015-05-27  1.0000  1.0000      --
        """

        data = {
            'FCODE': f'{fund_code}',
            'IsShareNet': 'true',
            'MobileKey': '1',
            'appType': 'ttjj',
            'appVersion': '6.2.8',
            'cToken': '1',
            'deviceid': '1',
            'pageIndex': '1',
            'pageSize': f'{pz}',
            'plat': 'Iphone',
            'product': 'EFund',
            'serverVersion': '6.2.8',
            'uToken': '1',
            'userId': '1',
            'version': '6.2.8'
        }
        url = 'https://fundmobapi.eastmoney.com/FundMNewApi/FundMNHisNetList'
        json_response = requests.get(
            url,
            headers=self.EastmoneyFundHeaders,
            data=data).json()
        rows = []
        columns = ['日期', '单位净值', '累计净值', '涨跌幅']
        if json_response is None:
            return pd.DataFrame(rows, columns=columns)
        datas = json_response['Datas']
        if len(datas) == 0:
            return pd.DataFrame(rows, columns=columns)
        rows = []
        for stock in datas:
            date = stock['FSRQ']
            rows.append({
                '日期': date,
                '单位净值': stock['DWJZ'],
                '累计净值': stock['LJJZ'],
                '涨跌幅': stock['JZZZL']
            })
        df = pd.DataFrame(rows)
        return df

    @retry(tries=3)
    @to_numeric
    def get_realtime_increase_rate(self, fund_codes: Union[List[str], str]) -> pd.DataFrame:
        """
        获取基金实时估算涨跌幅度
        Parameters
        ----------
        fund_codes : Union[List[str], str]
            6 位基金代码或者 6 位基金代码构成的字符串列表
        Returns
        -------
        DataFrame
            单只或者多只基金实时估算涨跌情况
        Examples
        --------

        >>> # 单只基金
        >>> self.get_realtime_increase_rate('161725')
            基金代码              名称  估算涨跌幅              估算时间
        0  161725  招商中证白酒指数(LOF)A  -0.64  2021-06-15 11:13
        >>> # 多只基金
        >>> self.get_realtime_increase_rate(['161725','005827'])
            基金代码              名称  估算涨跌幅              估算时间
        0  161725  招商中证白酒指数(LOF)A  -0.60  2021-06-15 11:16
        1  005827       易方达蓝筹精选混合  -1.36  2021-06-15 11:16
        """

        if not isinstance(fund_codes, list):
            fund_codes = [fund_codes]
        data = {
            'pageIndex': '1',
            'pageSize': '300000',
            'Sort': '',
            'Fcodes': ",".join(fund_codes),
            'SortColumn': '',
            'IsShowSE': 'false',
            'P': 'F',
            'deviceid': '3EA024C2-7F22-408B-95E4-383D38160FB3',
            'plat': 'Iphone',
            'product': 'EFund',
            'version': '6.2.8',
        }
        columns = {
            'FCODE': '基金代码',
            'SHORTNAME': '基金名称',
            'GSZZL': '估算涨跌幅',
            'GZTIME': '估算时间'
        }
        url = 'https://fundmobapi.eastmoney.com/FundMNewApi/FundMNFInfo'
        json_response = requests.get(
            url,
            headers=self.EastmoneyFundHeaders,
            data=data).json()
        rows = jsonpath(json_response, '$..Datas[:]')
        if not rows:
            df = pd.DataFrame(columns=columns.values())
            return df
        df = pd.DataFrame(rows).rename(columns=columns)
        return df

    @retry(tries=3)
    def get_fund_codes(self, ft: str = None) -> pd.DataFrame:
        """
        获取天天基金网公开的全部公墓基金名单
        Parameters
        ----------
        ft : str, optional
            基金类型可选示例如下
            - ``'zq'`` : 债券类型基金
            - ``'gp'`` : 股票类型基金
            - ``None`` : 全部
        Returns
        -------
        DataFrame
            天天基金网基金名单数据
        Examples
        --------
        
        >>> # 全部类型的基金
        >>> self.get_fund_codes()
        >>> # 股票型基金
        >>> self.get_fund_codes(ft = 'gp')
            基金代码                  基金简称
        0     003834              华夏能源革新股票
        1     005669            前海开源公用事业股票
        2     004040             金鹰医疗健康产业A
        3     517793                 1.20%
        4     004041             金鹰医疗健康产业C
        ...      ...                   ...
        1981  012503      国泰中证环保产业50ETF联接A
        1982  012517  国泰中证细分机械设备产业主题ETF联接C
        1983  012600             中银内核驱动股票C
        1984  011043             国泰价值先锋股票C
        1985  012516  国泰中证细分机械设备产业主题ETF联接A
        """

        params = [
            ('op', 'ph'),
            ('dt', 'kf'),
            ('rs', ''),
            ('gs', '0'),
            ('sc', '6yzf'),
            ('st', 'desc'),
            ('qdii', ''),
            ('pi', '1'),
            ('pn', '50000'),
            ('dx', '1'),
        ]
        headers = {
            'Connection': 'keep-alive',
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/87.0.4280.141 Safari/537.36 Edg/87.0.664.75',
            'Accept': '*/*',
            'Referer': 'http://fund.eastmoney.com/data/fundranking.html',
            'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8,en-GB;q=0.7,en-US;q=0.6',
        }
        if ft is not None and ft in ['gp', 'zq']:
            params.append(('ft', ft))
        url = 'http://fund.eastmoney.com/data/rankhandler.aspx'
        response = requests.get(
            url,
            headers=headers,
            params=params)
        # results = re.findall('(\d{6}),(.*?),', response.text)
        columns = ['基金代码', '基金简称']
        # results = re.findall('(\d{6}),(.*?),', response.text)
        results = re.findall('\"(\d{6}),(.*?),', response.text)
        df = pd.DataFrame(results, columns=columns)

        # text = text[text.index('["') + 2:text.index('"]')]
        # fs = text.split('","')
        # datas = {'基金代码': [], '基金简称': [], '净值': []]}
        # for f in fs:
        #     data = f.split(',')
        #     datas['基金代码'].append(data[0])
        #     datas['基金简称'].append(data[1])
        return df

    @retry(tries=3)
    @to_numeric
    def get_inverst_position(self, fund_code: str, dates: Union[str, List[str]] = None) -> pd.DataFrame:
        """
        获取基金持仓占比数据
        Parameters
        ----------
        fund_code : str
            基金代码
        dates : Union[str, List[str]], optional
            日期或者日期构成的列表
            可选示例如下
            - ``None`` : 最新公开日期
            - ``'2020-01-01'`` : 一个公开持仓日期
            - ``['2020-12-31' ,'2019-12-31']`` : 多个公开持仓日期
        Returns
        -------
        DataFrame
            基金持仓占比数据
        Examples
        --------
        
        >>> # 获取最新公开的持仓数据
        >>> self.get_inverst_position('161725')
            基金代码    股票代码  股票简称   持仓占比  较上期变化
        0  161725  000858   五粮液  14.88   1.45
        1  161725  600519  贵州茅台  14.16  -0.86
        2  161725  600809  山西汾酒  14.03  -0.83
        3  161725  000568  泸州老窖  13.02  -2.96
        4  161725  002304  洋河股份  12.72   1.31
        5  161725  000799   酒鬼酒   5.77   1.34
        6  161725  603369   今世缘   3.46  -0.48
        7  161725  000596  古井贡酒   2.81  -0.29
        8  161725  600779   水井坊   2.52   2.52
        9  161725  603589   口子窖   2.48  -0.38
        >>> # 获取近 2 个公开持仓日数据
        >>> public_dates = self.get_public_dates('161725')
        >>> self.get_inverst_position('161725',public_dates[:2])
            基金代码    股票代码  股票简称   持仓占比  较上期变化
        0  161725  000858   五粮液  14.88   1.45
        2  161725  600809  山西汾酒  14.03  -0.83
        3  161725  000568  泸州老窖  13.02  -2.96
        4  161725  002304  洋河股份  12.72   1.31
        5  161725  000799   酒鬼酒   5.77   1.34
        6  161725  603369   今世缘   3.46  -0.48
        7  161725  000596  古井贡酒   2.81  -0.29
        8  161725  600779   水井坊   2.52   2.52
        9  161725  603589   口子窖   2.48  -0.38
        0  161725  000568  泸州老窖  15.98   1.27
        1  161725  600519  贵州茅台  15.02   2.35
        2  161725  600809  山西汾酒  14.86  -0.37
        3  161725  000858   五粮液  13.43   0.54
        4  161725  002304  洋河股份  11.41  -2.21
        5  161725  000799   酒鬼酒   4.43  -0.15
        6  161725  603369   今世缘   3.94  -0.09
        7  161725  000860  顺鑫农业   3.12  -0.70
        8  161725  000596  古井贡酒   3.10  -0.15
        9  161725  603589   口子窖   2.86   0.21
        """

        columns = {
            'GPDM': '股票代码',
            'GPJC': '股票简称',
            'JZBL': '持仓占比',
            'PCTNVCHG': '较上期变化',
        }
        df = pd.DataFrame(columns=columns.values())
        if not isinstance(dates, List):
            dates = [dates]
        if dates is None:
            dates = [None]
        for date in dates:
            params = [
                ('FCODE', fund_code),
                ('OSVersion', '14.3'),
                ('appType', 'ttjj'),
                ('appVersion', '6.2.8'),
                ('deviceid', '3EA024C2-7F22-408B-95E4-383D38160FB3'),
                ('plat', 'Iphone'),
                ('product', 'EFund'),
                ('serverVersion', '6.2.8'),
                ('version', '6.2.8'),
            ]
            if date is not None:
                params.append(('DATE', date))
            url = 'https://fundmobapi.eastmoney.com/FundMNewApi/FundMNInverstPosition'
            json_response = requests.get(url,
                                         headers=self.EastmoneyFundHeaders, params=params).json()
            stocks = jsonpath(json_response, '$..fundStocks[:]')
            if not stocks:
                continue
            date = json_response['Expansion']
            _df = pd.DataFrame(stocks)
            _df = _df.rename(columns=columns)
            _df['公开日期'] = [date for _ in range(len(_df))]
            df = pd.concat([df, _df], axis=0, ignore_index=True)
        df = df[columns.values()]
        df.insert(0, '基金代码', fund_code)
        return df

    @retry(tries=3)
    @to_numeric
    def get_period_change(self, fund_code: str) -> pd.DataFrame:
        """
        获取基金阶段涨跌幅度
        Parameters
        ----------
        fund_code : str
            6 位基金代码
        Returns
        -------
        DataFrame
            指定基金的阶段涨跌数据
        Examples
        --------
        
        >>> self.get_period_change('161725')
            基金代码     收益率   同类平均  同类排行  同类总数   时间段
        0  161725   -6.28   0.07  1408  1409   近一周
        1  161725   10.85   5.82   178  1382   近一月
        2  161725   25.32   7.10    20  1332   近三月
        3  161725   22.93  10.39    79  1223   近六月
        4  161725  103.76  33.58     7  1118   近一年
        5  161725  166.59  55.42     9   796   近两年
        6  161725  187.50  48.17     2   611   近三年
        7  161725  519.44  61.62     1   389   近五年
        8  161725    6.46   5.03   423  1243  今年以来
        9  161725  477.00                     成立以来
        """

        params = (
            ('AppVersion', '6.3.8'),
            ('FCODE', fund_code),
            ('MobileKey', '3EA024C2-7F22-408B-95E4-383D38160FB3'),
            ('OSVersion', '14.3'),
            ('deviceid', '3EA024C2-7F22-408B-95E4-383D38160FB3'),
            ('passportid', '3061335960830820'),
            ('plat', 'Iphone'),
            ('product', 'EFund'),
            ('version', '6.3.6'),
        )
        url = 'https://fundmobapi.eastmoney.com/FundMNewApi/FundMNPeriodIncrease'
        json_response = requests.get(
            url,
            headers=self.EastmoneyFundHeaders,
            params=params).json()
        columns = {

            'syl': '收益率',
            'avg': '同类平均',
            'rank': '同类排行',
            'sc': '同类总数',
            'title': '时间段'

        }
        titles = {'Z': '近一周',
                  'Y': '近一月',
                  '3Y': '近三月',
                  '6Y': '近六月',
                  '1N': '近一年',
                  '2Y': '近两年',
                  '3N': '近三年',
                  '5N': '近五年',
                  'JN': '今年以来',
                  'LN': '成立以来'}
        # 发行时间
        ESTABDATE = json_response['Expansion']['ESTABDATE']
        df = pd.DataFrame(json_response['Datas'])

        df = df[list(columns.keys())].rename(columns=columns)
        df['时间段'] = titles.values()
        df.insert(0, '基金代码', fund_code)
        return df

    def get_public_dates(self, fund_code: str) -> List[str]:
        """
        获取历史上更新持仓情况的日期列表
        Parameters
        ----------
        fund_code : str
            6 位基金代码
        Returns
        -------
        List[str]
            指定基金公开持仓的日期列表
        Examples
        --------
        
        >>> public_dates = self.get_public_dates('161725')
        >>> # 展示前 5 个
        >>> public_dates[:5]
        ['2021-03-31', '2021-01-08', '2020-12-31', '2020-09-30', '2020-06-30']
        """

        params = (
            ('FCODE', fund_code),
            ('OSVersion', '14.3'),
            ('appVersion', '6.3.8'),
            ('deviceid', '3EA024C2-7F22-408B-95E4-383D38160FB3'),
            ('plat', 'Iphone'),
            ('product', 'EFund'),
            ('serverVersion', '6.3.6'),
            ('version', '6.3.8'),
        )
        url = 'https://fundmobapi.eastmoney.com/FundMNewApi/FundMNIVInfoMultiple'
        json_response = requests.get(
            url,
            headers=self.EastmoneyFundHeaders,
            params=params).json()
        if json_response['Datas'] is None:
            return []
        return json_response['Datas']

    ## TODO 浩哥有时间处理一下这个url 被反爬了似乎
    url = 'https://fundmobapi.eastmoney.com/FundMNewApi/FundMNAssetAllocationNew'

    @retry(tries=3)
    @to_numeric
    def get_types_persentage(self, fund_code: str, dates: Union[List[str], str, None] = None) -> pd.DataFrame:
        """
        获取指定基金不同类型占比信息
        Parameters
        ----------
        fund_code : str
            6 位基金代码
        dates : Union[List[str], str, None]
            可选值类型示例如下(后面有获取 dates 的例子)
            - ``None`` : 最新公开日期
            - ``'2020-01-01'`` : 一个公开持仓日期
            - ``['2020-12-31' ,'2019-12-31']`` : 多个公开持仓日期
        Returns
        -------
        DataFrame
            指定基金的在不同日期的不同类型持仓占比信息
        Examples
        --------
        
        >>> # 获取持仓公开日期
        >>> public_dates = self.get_public_dates('005827')
        >>> # 取前两个公开持仓日期
        >>> dates = public_dates[:2]
        >>> self.get_types_persentage('005827',dates)
            基金代码   股票比重 债券比重  现金比重         总规模(亿元) 其他比重
        0  005827   94.4   --  6.06  880.1570625231    0
        0  005827  94.09   --  7.63   677.007455712    0
        """

        columns = {
            'GP': '股票比重',
            'ZQ': '债券比重',
            'HB': '现金比重',
            'JZC': '总规模(亿元)',
            'QT': '其他比重'
        }
        df = pd.DataFrame(columns=columns.values())
        if not isinstance(dates, List):
            dates = [dates]
        elif dates is None:
            dates = [None]
        for date in dates:
            params = [
                ('FCODE', fund_code),
                ('OSVersion', '14.3'),
                ('appVersion', '6.3.8'),
                ('deviceid', '3EA024C2-7F21-408B-95E4-383D38160FB3'),
                ('plat', 'Iphone'),
                ('product', 'EFund'),
                ('serverVersion', '6.3.6'),
                ('version', '6.3.8'),
            ]
            if date is not None:
                params.append(('DATE', date))
            params = tuple(params)
            url = 'https://fundmobapi.eastmoney.com/FundMNewApi/FundMNAssetAllocationNew'
            json_response = requests.get(url, params=params).json()
            print(json_response)
            if len(json_response['Datas']) == 0:
                continue
            _df = pd.DataFrame(json_response['Datas'])[columns.keys()]
            _df = _df.rename(columns=columns)
            df = pd.concat([df, _df], axis=0, ignore_index=True)
        df.insert(0, '基金代码', fund_code)
        return df

    @retry(tries=3)
    @to_numeric
    def get_base_info_single(self, fund_code: str) -> pd.Series:
        """
        获取基金的一些基本信息
        Parameters
        ----------
        fund_code : str
            6 位基金代码
        Returns
        -------
        Series
            基金的一些基本信息
        """

        params = (
            ('FCODE', fund_code),
            ('deviceid', '3EA024C2-7F22-408B-95E4-383D38160FB3'),
            ('plat', 'Iphone'),
            ('product', 'EFund'),
            ('version', '6.3.8'),
        )
        url = 'https://fundmobapi.eastmoney.com/FundMNewApi/FundMNNBasicInformation'
        json_response = requests.get(
            url,
            headers=self.EastmoneyFundHeaders,
            params=params).json()
        columns = {
            'FCODE': '基金代码',
            'SHORTNAME': '基金简称',
            'ESTABDATE': '成立日期',
            'RZDF': '涨跌幅',
            'DWJZ': '最新净值',
            'JJGS': '基金公司',
            'FSRQ': '净值更新日期',
            'COMMENTS': '简介',
        }
        items = json_response['Datas']
        if not items:
            rich.print('基金代码', fund_code, '可能有误')
            return pd.Series(index=columns.values())

        s = pd.Series(json_response['Datas']).rename(
            index=columns)[columns.values()]

        s = s.apply(lambda x: x.replace('\n', ' ').strip()
        if isinstance(x, str) else x)
        return s

    def get_base_info_muliti(self, fund_codes: List[str]) -> pd.Series:
        """
        获取多只基金基本信息
        Parameters
        ----------
        fund_codes : List[str]
            6 位基金代码列表
        Returns
        -------
        Series
            多只基金基本信息
        """

        ss = []

        @multitasking.task
        @retry(tries=3, delay=1)
        def start(fund_code: str) -> None:
            s = self.get_base_info_single(fund_code)
            ss.append(s)
            pbar.update()
            pbar.set_description(f'Processing => {fund_code}')

        pbar = tqdm(total=len(fund_codes))
        for fund_code in fund_codes:
            start(fund_code)
        multitasking.wait_for_tasks()
        df = pd.DataFrame(ss)
        return df

    def get_base_info(self, fund_codes: Union[str, List[str]]) -> Union[pd.Series, pd.DataFrame]:
        """
        获取基金的一些基本信息
        Parameters
        ----------
        fund_codes : Union[str, List[str]]
            6 位基金代码 或多个 6 位 基金代码构成的列表
        Returns
        -------
        Union[Series, DataFrame]
            基金的一些基本信息
            - ``Series`` : 包含单只基金基本信息(当 ``fund_codes`` 是字符串时)
            - ``DataFrane`` : 包含多只股票基本信息(当 ``fund_codes`` 是字符串列表时)
        Raises
        ------
        TypeError
            当 fund_codes 类型不符合要求时
        Examples
        --------
        
        >>> self.get_base_info('161725')
        基金代码                                 161725
        基金简称                         招商中证白酒指数(LOF)A
        成立日期                             2015-05-27
        涨跌幅                                   -6.03
        最新净值                                 1.1959
        基金公司                                   招商基金
        净值更新日期                           2021-07-30
        简介        产品特色：布局白酒领域的指数基金，历史业绩优秀，外资偏爱白酒板块。
        dtype: object
        >>> # 获取多只基金基本信息
        >>> self.get_base_info(['161725','005827'])
            基金代码            基金简称        成立日期   涨跌幅    最新净值   基金公司      净值更新日期                                    简介00:00,  6.38it/s]
        0  005827       易方达蓝筹精选混合  2018-09-05 -2.98  2.4967  易方达基金  2021-07-30  明星消费基金经理另一力作，A+H股同步布局，价值投资典范，适合长期持有。
        1  161725  招商中证白酒指数(LOF)A  2015-05-27 -6.03  1.1959   招商基金  2021-07-30     产品特色：布局白酒领域的指数基金，历史业绩优秀，外资偏爱白酒板块。
        """

        if isinstance(fund_codes, str):
            return self.get_base_info_single(fund_codes)
        elif hasattr(fund_codes, '__iter__'):
            return self.get_base_info_muliti(fund_codes)
        raise TypeError(f'所给的 {fund_codes} 不符合参数要求')

    # *******************************************************************
    # 股票信息
    # *******************************************************************

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

    def get_all_fundings_info(self) -> pd.DataFrame:
        raw_data = self.get_fund_codes()
        data = defaultdict(list)
        for index, row in raw_data.iterrows():
            if index > 30: break
            try:
                inverst_position_raw = self.get_inverst_position(row['基金代码'])
                period_change_raw = self.get_period_change(row['基金代码'])
                base_info_single_raw = self.get_base_info_single(row['基金代码'])
                realtime_increase_rate_raw = self.get_realtime_increase_rate(row['基金代码'])
            except:
                print(row['基金代码'], row['基金简称'])
                continue
            data['基金代码'].append(row['基金代码'])
            data['基金简称'].append(row['基金简称'])

            data['基金估算涨跌幅'].append(realtime_increase_rate_raw['估算涨跌幅'])
            data['基金估算涨跌幅时间'].append(realtime_increase_rate_raw['估算时间'])

            num_list = ['一', '二', '三', '四', '五', '六', '七', '八', '九', '十']
            for num in range(10):
                data['第' + num_list[num] + '一大重仓股简称'].append(inverst_position_raw['股票简称'][num])
                data['第' + num_list[num] + '大重仓股代码'].append(inverst_position_raw['股票代码'][num])
                data['第' + num_list[num] + '大重仓股占比'].append(inverst_position_raw['持仓占比'][num])
                base_info_single_raw_tmp = self.get_base_info_stock_single(inverst_position_raw['股票代码'][num])
                data['第' + num_list[num] + '大重仓股动态市盈率'].append(base_info_single_raw_tmp['市盈率(动)'])
                data['第' + num_list[num] + '大重仓股市净率'].append(base_info_single_raw_tmp['市净率'])
                data['第' + num_list[num] + '大重仓股所处行业'].append(base_info_single_raw_tmp['所处行业'])
                data['第' + num_list[num] + '大重仓股总市值'].append(base_info_single_raw_tmp['总市值'])
                data['第' + num_list[num] + '大重仓股流通市值'].append(base_info_single_raw_tmp['流通市值'])
                data['第' + num_list[num] + '大重仓股ROE'].append(base_info_single_raw_tmp['ROE'])
                data['第' + num_list[num] + '大重仓股净利润'].append(base_info_single_raw_tmp['净利率'])
                data['第' + num_list[num] + '大重仓股净利率'].append(base_info_single_raw_tmp['净利润'])
                data['第' + num_list[num] + '大重仓股毛利率'].append(base_info_single_raw_tmp['毛利率'])
            data['成立日期'].append(base_info_single_raw[2])
            data['昨日收盘涨跌幅'].append(base_info_single_raw[3])
            data['最新净值'].append(base_info_single_raw[4])
            data['净值更新日期'].append(base_info_single_raw[5])
            data['近一周收益率'].append(period_change_raw['收益率'][0])
            data['近一月收益率'].append(period_change_raw['收益率'][1])
            data['近三月收益率'].append(period_change_raw['收益率'][2])
            data['近六月收益率'].append(period_change_raw['收益率'][3])
            data['近一年收益率'].append(period_change_raw['收益率'][4])
            data['近两年收益率'].append(period_change_raw['收益率'][5])

        result = pd.DataFrame(data)
        result.to_csv('./基金基本信息.csv', index=False)
        return result


if __name__ == "__main__":
    a = BasicSpider()
    # ******************测试基金***********************
    a.get_all_fundings_info()
