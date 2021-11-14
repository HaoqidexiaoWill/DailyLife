from Spider.fundings_spider import FundingsSpider
from Config.FundingDict import FUNDING_DICT
from MailModule.BasicQQMailModule import BasicQQMailModule
import pandas as pd
class MainStrategy:
    def __init__(self) -> None:
        self.funding_dict = FUNDING_DICT
        self.funding_spider = FundingsSpider()
        self.mail = BasicQQMailModule()

    def edit_email_text(self,each_fund,each_code,*args):
        email_text = ''
        # 统一格式
        EMA20,EMA10,EMA5,MIN60,MIN30,RANGE,quote_realtime_price = map(lambda x:float('%.3f'% float(x)),args)
        # 先处理跌的逻辑
        if RANGE < 0.0 :
            if quote_realtime_price <= min(EMA20,EMA10,EMA5):
                # print(f'{each_fund}-{each_code} 当前价格 {quote_realtime_price} 已经跌到20日均线附近 {EMA20}')
                email_text += f'{each_fund}-{each_code} 当前价格 {quote_realtime_price} 已经跌到20日均线附近 {EMA20}\n'
            if quote_realtime_price <= MIN30:
                # print(f'{each_fund}-{each_code} 当前价格 {quote_realtime_price} 已经跌到30日最低点附近 {MIN30}')
                email_text += f'{each_fund}-{each_code} 当前价格 {quote_realtime_price} 已经跌到30日最低点附近 {MIN30}\n'
            if quote_realtime_price <= MIN60:
                # print(f'{each_fund}-{each_code} 当前价格 {quote_realtime_price} 已经跌到60日最低点附近 {MIN60}')    
                email_text += f'{each_fund}-{each_code} 当前价格 {quote_realtime_price} 已经跌到30日最低点附近 {MIN30}\n'
        return email_text

    def run(self):
        all_email_text = ''
        for each_fund,each_code in self.funding_dict.items():
            df_quote_history = self.funding_spider.get_finding_history(each_code)
            # 计算滑动平均值(N日均值)
            df_quote_history_60 = df_quote_history[:60][::-1]
            df_quote_history_60['EMA20'] = df_quote_history_60['单位净值'].ewm(span=20,min_periods=0,adjust=False,ignore_na=False).mean()
            df_quote_history_60['EMA10'] = df_quote_history_60['单位净值'].ewm(span=10,min_periods=0,adjust=False,ignore_na=False).mean()
            df_quote_history_60['EMA5'] = df_quote_history_60['单位净值'].ewm(span=5,min_periods=0,adjust=False,ignore_na=False).mean()
            EMA20 = df_quote_history_60['EMA20'].tail(1).values[0]
            EMA10 = df_quote_history_60['EMA10'].tail(1).values[0]
            EMA5 = df_quote_history_60['EMA5'].tail(1).values[0]
            # 计算近期最低价
            df_quote_history_30 = df_quote_history[:30][::-1]
            MIN30 = min(df_quote_history_30['单位净值'])
            MIN60 = min(df_quote_history_60['单位净值'])
            # 当前估算价格
            quote_realtime_price = self.funding_spider.get_realtime_increase_rate(str(each_code))['GSZ'].values[0]
            RANGE = self.funding_spider.get_realtime_increase_rate(str(each_code))['估算涨跌幅'].values[0]

            each_email_text = self.edit_email_text(each_fund,each_code,EMA20,EMA10,EMA5,MIN60,MIN30,RANGE,quote_realtime_price)
            print(each_email_text)
            all_email_text += each_email_text
            all_email_text + '\n'
        if all_email_text:
            self.mail.sent_email_single(text_content=all_email_text)




            




if __name__ == "__main__":
    a = MainStrategy()
    result = a.run()