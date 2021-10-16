import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
sys.path.append('../')
sys.path.append('../../')
from Spider.fundings_spider import FundingsSpider
from Spider.quotes_spider import QuotesSpider
pd.set_option('expand_frame_repr', False)  # 当列太多时不换行
pd.set_option('display.max_rows', 5000)    # 最多显示数据的行数
from functools import reduce


class BasicStrategyMultiFundings:
    def __init__(self) -> None:
        self.name = '多基金轮动策略'
        # 万分之1.5，买卖手续费相同，无印花税
        self.trade_rate = 1.5 / 10000  
        self.N = 20  # 计算多少天的涨幅

        # self.quotes_dict = {
        #     '沪深300指数':'000300',
        #     '创业板指数':'399006',
        #     '中证500指数':'399905'
        # }
        self.quotes_dict = {
            '白酒指数':'161725',
            '诺安成长':'320007',
            '中证医疗':'162412'
        }
        # self.spider = QuotesSpider()
        self.spider = FundingsSpider()



    def load_quotes(self):
        quote_list = []
        for each_quote_name,each_quote_code in self.quotes_dict.items():
            # df_quote_history = self.spider.get_quote_history_single(each_quote_code)
            df_quote_history = self.spider.get_finding_history(each_quote_code)
            df_quote_history = df_quote_history.iloc[::-1]

            df_quote_history['收盘'] = df_quote_history['单位净值']
            df_quote_history['开盘'] = df_quote_history['单位净值'].shift(1) - 1
            df_quote_history[each_quote_name+'当日涨跌幅'] = df_quote_history['收盘'] / df_quote_history['收盘'].shift(1) - 1
            df_quote_history[each_quote_name+'N日涨跌幅']  = df_quote_history['收盘'].pct_change(periods=self.N)
            # 重命名行
            df_quote_history.rename(columns={
                '开盘': each_quote_name +'开盘价', 
                '收盘': each_quote_name +'收盘价',
                }, inplace=True)
            
            quote_list.append(df_quote_history[[
                '日期',
                each_quote_name +'开盘价',
                each_quote_name +'收盘价',
                each_quote_name +'当日涨跌幅',
                each_quote_name +'N日涨跌幅'
                ]])
        df = reduce(lambda left,right:pd.merge(left,right,on = '日期'),quote_list)
        df['日期'] = pd.to_datetime(df['日期'])
        df.dropna(inplace=True)
        return df


    def compute_momentum_byNdays(self, df):
        for each_quote_name,each_quote_code in self.quotes_dict.items():
            momentum_byNdays = each_quote_name + 'N日涨跌幅'
            each_quote_close_price = each_quote_name + '收盘价'
            df[momentum_byNdays] = df[each_quote_close_price].pct_change(periods=self.N)
        return df

    
    
    
    def transfer_positions(self, df):
        # 策略二 沪深300 与 创业板 轮动策略 + 空仓控制风险

        # # 调仓时机
        # df.loc[df['沪深300N日涨跌幅'] > df['创业板N日涨跌幅'], '行情'] = '沪深300行情'
        # df.loc[df['沪深300N日涨跌幅'] < df['创业板N日涨跌幅'], '行情'] = '创业板行情'
        # df.loc[(df['沪深300N日涨跌幅'] < 0) & (df['创业板N日涨跌幅'] < 0), '行情'] = '市场暴跌空仓保命'

        for each_quote_name,each_quote_code in self.quotes_dict.items(): 
            momentum_byNdays = each_quote_name + 'N日涨跌幅'     
            data_Nday = {k:v for k,v in enumerate([column for column in df]) if 'N日涨跌幅' in v}  
            df.loc[df[momentum_byNdays] == df[list(data_Nday.values())].apply(lambda x:max(x),axis=1),'行情'] = each_quote_name
        df["N日最大涨幅"] = df[list(data_Nday.values())].apply(lambda x:max(x),axis=1)
        df.loc[df['N日最大涨幅']<0,'行情'] = '空仓'

        # 相等时维持原来的仓位。
        df['行情'].fillna(method='ffill', inplace=True)
        # 收盘才能确定风格，实际的调仓要晚一天。
        df['持仓'] = df['行情'].shift(1)
        # 删除持仓为nan的天数（创业板2010年才有）
        df.dropna(subset=['持仓'], inplace=True)
        

        # 计算策略的整体涨跌幅策略当日涨跌幅
        for each_quote_name,each_quote_code in self.quotes_dict.items(): 
            momentum_by1days = each_quote_name + '当日涨跌幅'
            df.loc[df['持仓'] == each_quote_name, '策略当日涨跌幅'] = df[momentum_by1days]
        df.loc[df['持仓'] == '空仓', '策略当日涨跌幅'] = 0

        # 调仓时间
        df.loc[df['持仓'] != df['持仓'].shift(1), '调仓日期'] = df['日期']

        # 将调仓日的涨跌幅修正为开盘价买入涨跌幅（并算上交易费用，没有取整数100手，所以略有误差）
        for each_quote_name,each_quote_code in self.quotes_dict.items():
            each_quote_name_close_price = each_quote_name + '收盘价'
            each_quote_name_open_price  = each_quote_name + '开盘价'
            df.loc[(df['调仓日期'].notnull()) & (df['持仓'] == each_quote_name),'调仓日的涨跌幅修正'] = df[each_quote_name_close_price] / (df[each_quote_name_open_price] * (1 + self.trade_rate)) - 1
        df.loc[df['调仓日期'].isnull(), '调仓日的涨跌幅修正'] = df['策略当日涨跌幅']
        return df




    def remove_cost(self, df):
        # 扣除交易成本
        df.loc[(df['调仓日期'].shift(-1) != pd.NaT), '调仓日的涨跌幅修正'] = (1 + df['策略当日涨跌幅']) * (1 - self.trade_rate) - 1
        df['调仓日的涨跌幅修正'].fillna(value=0.0, inplace=True)
        del df['策略当日涨跌幅'], df['行情']
        return df

        # 空仓的日子，涨跌幅用0填充


    def run(self):
        df = self.load_quotes()
        df = self.compute_momentum_byNdays(df)
        df = self.transfer_positions(df)
        # 扣除卖出手续费
        df = self.remove_cost(df)


        df.reset_index(drop=True, inplace=True)

        # 计算净值
        for each_quote_name,each_quote_code in self.quotes_dict.items():
            each_quote_name_close_price = each_quote_name + '收盘价'
            each_quote_name_momentum_byNdays = each_quote_name + '累计涨幅'
            df[each_quote_name_momentum_byNdays] = df[each_quote_name_close_price] / df[each_quote_name_close_price][0]

        df['策略累计涨幅'] = (1 + df['调仓日的涨跌幅修正']).cumprod()


        df['日期'] =pd.to_datetime(df['日期'])
        # 详细评估策略的好坏
        res = self.evaluate_investment(df, '策略累计涨幅', time='日期')
        # 保存文件
        print(df.tail(10))
        self.draw_plots(df)
        print(res)
        df.to_csv('大小盘风格切换.csv', encoding='gbk', index=False)
    



    def draw_plots(self, df):
        # 绘制图形& 解决中文乱码问题
        from pylab import mpl
        mpl.rcParams['font.sans-serif'] = ['SimHei']
        # plt.plot(df['日期'], df['策略累计涨幅']   , label='当前策略收益')
        # plt.plot(df['日期'], df['沪深300累计涨幅'], label='沪深300收益')
        # plt.plot(df['日期'], df['创业板累计涨幅'] , label='创业板指数收益')
        for each_quote_name,each_quote_code in self.quotes_dict.items():
            each_quote_name_momentum_byNdays = each_quote_name + '累计涨幅'
            plt.plot(df['日期'], df[each_quote_name_momentum_byNdays] , label=each_quote_name + '收益')
        plt.plot(df['日期'], df['策略累计涨幅'], label='当前策略收益')
        plt.title(self.name)
        plt.legend()
        plt.show()

    def evaluate_investment(self, source_data, column, time='交易日期'):
        temp = source_data.copy()
        # ===新建一个dataframe保存回测指标
        results = pd.DataFrame()

        # ===计算累积净值
        results.loc[0, '累积净值'] = round(temp[column].iloc[-1], 2)

        # ===计算年化收益
        annual_return = (temp[column].iloc[-1]) ** ('1 days 00:00:00' / (temp[time].iloc[-1] - temp[time].iloc[0]) * 365) - 1
        results.loc[0, '年化收益'] = str(round(annual_return * 100, 2)) + '%'
        # ===计算最大回撤
        # 计算当日之前的资金曲线的最高点
        temp['max2here'] = temp[column].expanding().max()
        # 计算到历史最高值到当日的跌幅，drowdwon
        temp['dd2here'] = temp[column] / temp['max2here'] - 1
        # 计算最大回撤，以及最大回撤结束时间
        end_date, max_draw_down = tuple(temp.sort_values(
            by=['dd2here']).iloc[0][[time, 'dd2here']])
        # 计算最大回撤开始时间
        start_date = temp[temp[time] <= end_date].sort_values(
            by=column, ascending=False).iloc[0][time]
        # 将无关的变量删除
        temp.drop(['max2here', 'dd2here'], axis=1, inplace=True)
        results.loc[0, '最大回撤'] = format(max_draw_down, '.2%')
        results.loc[0, '最大回撤开始时间'] = str(start_date)
        results.loc[0, '最大回撤结束时间'] = str(end_date)

        # ===年化收益/回撤比：我个人比较关注的一个指标
        results.loc[0, '年化收益/回撤比'] = round(annual_return / abs(max_draw_down), 2)

        return results.T

    # def run(self):

    #     # _, _, df = self.load_data()

    #     # # 计算N日涨跌幅
    #     # df = self.compute_momentum_byNdays(df)

    #     # df = self.transfer_positions(df)
    #     df = self.load_quotes()


if __name__ == "__main__":
    a = BasicStrategyMultiFundings()
    result = a.run()
