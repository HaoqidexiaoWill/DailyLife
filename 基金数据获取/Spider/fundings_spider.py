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
class FundingsSpider(BasicSpider):
    def __init__(self):
        super(FundingsSpider, self).__init__()
    @retry(tries=3)
    @to_numeric
    def get_finding_history(self, fund_code: str, pz: int = 40000) -> pd.DataFrame:
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


if __name__ == "__main__":
    a = FundingsSpider()
    # ******************测试基金***********************
    print('获取基金历史净值信息')
    result = a.get_finding_history('320007')
    print('获取基金实时涨跌幅')
    result = a.get_realtime_increase_rate(['161725','005827'])
    print(result)
    print('获取天天基金网公开的全部公墓基金名单')
    result = a.get_fund_codes()
    print(result)
    print('获取基金持仓占比数据')
    result = a.get_inverst_position('161725')
    print('获取基金阶段涨跌幅')
    result = a.get_period_change('161725')
    print(result)
    print('获取基金更新持仓情况列表')
    result = a.get_public_dates('161725')
    print(result)
    print('获取基金的一些基本信息')
    result = a.get_base_info(['161725','005827'])
    print(result)