import smtplib
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
from email.mime.application import MIMEApplication
import random,time
import sys
sys.path.append('../')
from Config.CommonConfig import SOURCE_EMAIL,TARGET_EMAIL_LIST,EMAIL_PASSWORD


class BasicQQMailModule:
    def __init__(self) -> None:
        pass
    #写成了一个通用的函数接口，想直接用的话，把参数的注释去掉就好
    def sent_email_single(self,source_email = SOURCE_EMAIL,passwd = EMAIL_PASSWORD,target_email = SOURCE_EMAIL,
                               subject = "今日基金信息" ,text_content= '自动邮件，测试群发今日基金信息',file_path='../基金基本信息.csv')-> None:
        '''
        source_email    = '111111111@qq.com'      发送方邮箱
        passwd          = '111111111'             填入发送方邮箱的授权码
        target_email    = '1111111118@qq.com'     收件人邮箱
        '''

        msg             = MIMEMultipart()
        text            = MIMEText(text_content)
        msg.attach(text)
    
    
        docApart = MIMEApplication(open(file_path, 'rb').read())
        docApart.add_header('Content-Disposition', 'attachment', filename=file_path)
        msg.attach(docApart)
        
        s = smtplib.SMTP_SSL("smtp.qq.com", 465)
        s.login(source_email, passwd)
        
        msg['Subject']  = subject
        msg['From']     = source_email
        msg['To']       = target_email        
        s.sendmail(source_email, target_email, msg.as_string())
        print(f"发送给{target_email}成功")
    def sent_email_single(self,source_email = SOURCE_EMAIL,passwd = EMAIL_PASSWORD,target_email_list = TARGET_EMAIL_LIST,text_content= '',file_path='../基金基本信息.csv'):
        for each_target in target_email_list:
            self.sent_email_single(target_email = each_target)
            # 防止被腾讯邮箱封号，随机停止几秒之后再发
            time.sleep(random.randint(5,10))
            print(f"发送给{each_target}成功")



if __name__ == "__main__":
    a = BasicQQMailModule()
    # ******************测试基金***********************
    a.sent_email_single()