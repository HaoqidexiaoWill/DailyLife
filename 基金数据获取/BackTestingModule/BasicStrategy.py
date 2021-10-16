import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
pd.set_option('expand_frame_repr', False)  # 当列太多时不换行
pd.set_option('display.max_rows', 5000)  # 最多显示数据的行数


class BasicStrategy:
    def __init__(self) -> None:
        # 设置参数
        self.name = '基本轮动策略'
        self.trade_rate = 0.6 / 10000  # 万分之0.6，买卖手续费相同，无印花税
        self.N = 20  # 计算多少天的涨幅

    def load_data(self):
        # 读取数据
        df_sh000300 = pd.read_csv('sh000300.csv', parse_dates=['candle_end_time'])
        df_sz399006 = pd.read_csv('sz399006.csv', parse_dates=['candle_end_time'])

        # 计算大小盘每天的涨跌幅amplitude
        df_sh000300['沪深300当日涨幅'] = df_sh000300['close'] / df_sh000300['close'].shift(1) - 1
        df_sz399006['创业板当日涨幅']  = df_sz399006['close'] /  df_sz399006['close'].shift(1) - 1

        # 重命名行
        df_sh000300.rename(
            columns={'open': '沪深300开盘价', 'close': '沪深300收盘价'}, inplace=True)
        df_sz399006.rename(
            columns={'open': '创业板开盘价', 'close': '创业板收盘价'}, inplace=True)
        # 合并数据
        df = pd.merge(
            left=df_sh000300[['candle_end_time','沪深300开盘价', '沪深300收盘价', '沪深300当日涨幅']],
            left_on=['candle_end_time'],
            right=df_sz399006[['candle_end_time','创业板开盘价', '创业板收盘价', '创业板当日涨幅']],
            right_on=['candle_end_time'],
            how='left'
        )
        df.rename(columns={'candle_end_time': '开盘日期'}, inplace=True)
        return df_sh000300, df_sz399006, df

    def compute_momentum_byNdays(self, df):
        # 计算N日的涨跌幅
        df['沪深300N日涨跌幅'] = df['沪深300收盘价'].pct_change(periods=self.N)
        df['创业板N日涨跌幅']  = df['创业板收盘价'].pct_change(periods=self.N)
        return df

    def transfer_positions(self, df):
        # 策略一 沪深300 与 创业板 轮动策略

        # 调仓时机
        df.loc[df['沪深300N日涨跌幅'] > df['创业板N日涨跌幅'], '行情'] = '沪深300行情'
        df.loc[df['沪深300N日涨跌幅'] < df['创业板N日涨跌幅'], '行情'] = '创业板行情'
        # 相等时维持原来的仓位。
        df['行情'].fillna(method='ffill', inplace=True)
        # 收盘才能确定风格，实际的调仓要晚一天。
        df['持仓'] = df['行情'].shift(1)
        # 删除持仓为nan的天数（创业板2010年才有）
        df.dropna(subset=['持仓'], inplace=True)
        # 计算策略的整体涨跌幅策略当日涨跌幅
        df.loc[df['持仓'] == '沪深300行情', '策略当日涨跌幅'] = df['沪深300当日涨幅']
        df.loc[df['持仓'] == '创业板行情', '策略当日涨跌幅'] = df['创业板当日涨幅']

        # 调仓时间
        df.loc[df['持仓'] != df['持仓'].shift(1), '调仓日期'] = df['开盘日期']

        # 将调仓日的涨跌幅修正为开盘价买入涨跌幅（并算上交易费用，没有取整数100手，所以略有误差）
        df.loc[(df['调仓日期'].notnull()) & (df['持仓'] == '沪深300行情'),'调仓日的涨跌幅修正'] = df['沪深300收盘价'] / (df['沪深300开盘价'] * (1 + self.trade_rate)) - 1
        df.loc[(df['调仓日期'].notnull()) & (df['持仓'] == '创业板行情') ,'调仓日的涨跌幅修正'] = df['创业板收盘价']  / (df['创业板开盘价']  * (1 + self.trade_rate)) - 1
        df.loc[df['调仓日期'].isnull(), '调仓日的涨跌幅修正'] = df['策略当日涨跌幅']

        return df

    def remove_cost(self, df):
        # 扣除交易成本
        df.loc[(df['调仓日期'].shift(-1) != pd.NaT), '调仓日的涨跌幅修正'] = (1 + df['策略当日涨跌幅']) * (1 - self.trade_rate) - 1
        del df['策略当日涨跌幅'], df['行情']
        return df

    def draw_plots(self, df):
        # 绘制图形& 解决中文乱码问题
        from pylab import mpl
        mpl.rcParams['font.sans-serif'] = ['SimHei']
        plt.plot(df['开盘日期'], df['策略累计涨幅']   , label='当前策略收益')
        plt.plot(df['开盘日期'], df['沪深300累计涨幅'], label='沪深300收益')
        plt.plot(df['开盘日期'], df['创业板累计涨幅'] , label='创业板指数收益')
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

    def run(self):

        _, _, df = self.load_data()

        # 计算N日涨跌幅
        df = self.compute_momentum_byNdays(df)

        df = self.transfer_positions(df)
        # 扣除卖出手续费
        df = self.remove_cost(df)

        df.reset_index(drop=True, inplace=True)

        # 计算净值
        df['沪深300累计涨幅'] = df['沪深300收盘价'] / df['沪深300收盘价'][0]
        df['创业板累计涨幅'] = df['创业板收盘价'] / df['创业板收盘价'][0]
        df['策略累计涨幅'] = (1 + df['调仓日的涨跌幅修正']).cumprod()

        # 详细评估策略的好坏
        res = self.evaluate_investment(df, '策略累计涨幅', time='开盘日期')
        # 保存文件
        print(df.tail(10))
        self.draw_plots(df)
        print(res)
        df.to_csv('大小盘风格切换.csv', encoding='gbk', index=False)


class BasicStrategywithEmpty(BasicStrategy):
    def __init__(self) -> None:
        super(BasicStrategywithEmpty,self).__init__() 
        self.name = '带有空仓期的基本轮动策略'

    def transfer_positions(self, df):
        # 策略二 沪深300 与 创业板 轮动策略 + 空仓控制风险

        # 调仓时机
        df.loc[df['沪深300N日涨跌幅'] > df['创业板N日涨跌幅'], '行情'] = '沪深300行情'
        df.loc[df['沪深300N日涨跌幅'] < df['创业板N日涨跌幅'], '行情'] = '创业板行情'
        df.loc[(df['沪深300N日涨跌幅'] < 0) & (df['创业板N日涨跌幅'] < 0), '行情'] = '市场暴跌空仓保命'


        # 相等时维持原来的仓位。
        df['行情'].fillna(method='ffill', inplace=True)
        # 收盘才能确定风格，实际的调仓要晚一天。
        df['持仓'] = df['行情'].shift(1)
        # 删除持仓为nan的天数（创业板2010年才有）
        df.dropna(subset=['持仓'], inplace=True)
        # 计算策略的整体涨跌幅策略当日涨跌幅
        df.loc[df['持仓'] == '沪深300行情', '策略当日涨跌幅'] = df['沪深300当日涨幅']
        df.loc[df['持仓'] == '创业板行情', '策略当日涨跌幅'] = df['创业板当日涨幅']
        df.loc[df['持仓'] == '市场暴跌空仓保命', '策略当日涨跌幅'] = 0

        # 调仓时间
        df.loc[df['持仓'] != df['持仓'].shift(1), '调仓日期'] = df['开盘日期']

        # 将调仓日的涨跌幅修正为开盘价买入涨跌幅（并算上交易费用，没有取整数100手，所以略有误差）
        df.loc[(df['调仓日期'].notnull()) & (df['持仓'] == '沪深300行情'),'调仓日的涨跌幅修正'] = df['沪深300收盘价'] / (df['沪深300开盘价'] * (1 + self.trade_rate)) - 1
        df.loc[(df['调仓日期'].notnull()) & (df['持仓'] == '创业板行情') ,'调仓日的涨跌幅修正'] = df['创业板收盘价']  / (df['创业板开盘价']  * (1 + self.trade_rate)) - 1
        df.loc[df['调仓日期'].isnull(), '调仓日的涨跌幅修正'] = df['策略当日涨跌幅']


        return df

    def remove_cost(self, df):
        # 扣除交易成本
        df.loc[(df['调仓日期'].shift(-1) != pd.NaT), '调仓日的涨跌幅修正'] = (1 + df['策略当日涨跌幅']) * (1 - self.trade_rate) - 1
        df['调仓日的涨跌幅修正'].fillna(value=0.0, inplace=True)
        del df['策略当日涨跌幅'], df['行情']
        return df

        # 空仓的日子，涨跌幅用0填充


if __name__ == "__main__":
    a = BasicStrategy()
    print(f"{a.name}")
    a.run()
    a = BasicStrategywithEmpty()
    print(f"{a.name}")
    a.run()
