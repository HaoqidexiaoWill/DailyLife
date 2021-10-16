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





    # TODO 有空再处理这个

    # def get_all_fundings_info(self) -> pd.DataFrame:
    #     raw_data = self.get_fund_codes()
    #     data = defaultdict(list)
    #     for index, row in raw_data.iterrows():
    #         if index > 30: break
    #         try:
    #             inverst_position_raw = self.get_inverst_position(row['基金代码'])
    #             period_change_raw = self.get_period_change(row['基金代码'])
    #             base_info_single_raw = self.get_base_info_single(row['基金代码'])
    #             realtime_increase_rate_raw = self.get_realtime_increase_rate(row['基金代码'])
    #         except:
    #             print(row['基金代码'], row['基金简称'])
    #             continue
    #         data['基金代码'].append(row['基金代码'])
    #         data['基金简称'].append(row['基金简称'])

    #         data['基金估算涨跌幅'].append(realtime_increase_rate_raw['估算涨跌幅'])
    #         data['基金估算涨跌幅时间'].append(realtime_increase_rate_raw['估算时间'])

    #         num_list = ['一', '二', '三', '四', '五', '六', '七', '八', '九', '十']
    #         for num in range(10):
    #             data['第' + num_list[num] + '一大重仓股简称'].append(inverst_position_raw['股票简称'][num])
    #             data['第' + num_list[num] + '大重仓股代码'].append(inverst_position_raw['股票代码'][num])
    #             data['第' + num_list[num] + '大重仓股占比'].append(inverst_position_raw['持仓占比'][num])
    #             base_info_single_raw_tmp = self.get_base_info_stock_single(inverst_position_raw['股票代码'][num])
    #             data['第' + num_list[num] + '大重仓股动态市盈率'].append(base_info_single_raw_tmp['市盈率(动)'])
    #             data['第' + num_list[num] + '大重仓股市净率'].append(base_info_single_raw_tmp['市净率'])
    #             data['第' + num_list[num] + '大重仓股所处行业'].append(base_info_single_raw_tmp['所处行业'])
    #             data['第' + num_list[num] + '大重仓股总市值'].append(base_info_single_raw_tmp['总市值'])
    #             data['第' + num_list[num] + '大重仓股流通市值'].append(base_info_single_raw_tmp['流通市值'])
    #             data['第' + num_list[num] + '大重仓股ROE'].append(base_info_single_raw_tmp['ROE'])
    #             data['第' + num_list[num] + '大重仓股净利润'].append(base_info_single_raw_tmp['净利率'])
    #             data['第' + num_list[num] + '大重仓股净利率'].append(base_info_single_raw_tmp['净利润'])
    #             data['第' + num_list[num] + '大重仓股毛利率'].append(base_info_single_raw_tmp['毛利率'])
    #         data['成立日期'].append(base_info_single_raw[2])
    #         data['昨日收盘涨跌幅'].append(base_info_single_raw[3])
    #         data['最新净值'].append(base_info_single_raw[4])
    #         data['净值更新日期'].append(base_info_single_raw[5])
    #         data['近一周收益率'].append(period_change_raw['收益率'][0])
    #         data['近一月收益率'].append(period_change_raw['收益率'][1])
    #         data['近三月收益率'].append(period_change_raw['收益率'][2])
    #         data['近六月收益率'].append(period_change_raw['收益率'][3])
    #         data['近一年收益率'].append(period_change_raw['收益率'][4])
    #         data['近两年收益率'].append(period_change_raw['收益率'][5])

    #     result = pd.DataFrame(data)
    #     result.to_csv('./基金基本信息.csv', index=False)
    #     print('done')
    #     return result


if __name__ == "__main__":
    a = BasicSpider()
    # ******************测试基金***********************

