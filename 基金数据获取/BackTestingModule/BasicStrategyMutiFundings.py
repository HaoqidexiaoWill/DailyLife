import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from BasicStrategy import BasicStrategy
import sys
sys.path.append('../')

from Spider.BasicSpider import BasicSpider
pd.set_option('expand_frame_repr', False)  # 当列太多时不换行
pd.set_option('display.max_rows', 5000)  # 最多显示数据的行数
from functools import reduce


class BasicStrategyMultiFundings:
    def __init__(self) -> None:
        self.name = '多基金轮动策略'
        # 万分之1.5，买卖手续费相同，无印花税
        self.trade_rate = 1.5 / 10000  
        self.N = 20  # 计算多少天的涨幅

        self.quotes_dict = {
            '沪深300指数':'000300',
            '创业板指数':'399006'
        }
        self.spider = BasicSpider()

    def load_quotes(self):
        quote_list = []
        for each_quote_name,each_quote_code in self.quotes_dict.items():
            df_quote_history = self.spider.get_quote_history_single(each_quote_code)
            df_quote_history[each_quote_name+'N日涨跌幅'] = df_quote_history['收盘'].pct_change(periods=self.N)
            # 重命名行
            df_quote_history.rename(columns={
                '开盘': each_quote_name +'开盘价', 
                '收盘': each_quote_name +'收盘价',
                '涨跌幅': each_quote_name +'涨跌幅'
                }, inplace=True)
            
            quote_list.append(df_quote_history[[
                '日期',
                each_quote_name +'开盘价',
                each_quote_name +'收盘价',
                each_quote_name +'涨跌幅',
                each_quote_name +'N日涨跌幅'
                ]])
        df = reduce(lambda left,right:pd.merge(left,right,on = '日期'),quote_list)
        print(df.tail(10))
        return df


    def transfer_positions(self, df):
        # 策略三 哪个涨的多买哪个

        # 调仓时机
        pass

if __name__ == "__main__":
    a = BasicStrategyMultiFundings()
    a.load_quotes()