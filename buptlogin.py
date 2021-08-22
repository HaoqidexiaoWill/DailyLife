import re
import requests


def getcookie():
    session = requests.session()
    result = session.post(url, data=data)
    cookies = requests.utils.dict_from_cookiejar(session.cookies)
    return str(cookies)[15:31]


def login(data):
    postdata = data
    headers = {'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8',
               'Accept-Encoding': 'gzip, deflate',
               'Accept-Language': 'zh-CN,zh;q=0.9',
               'Cache-Control': 'max-age=0',
               'Connection': 'keep-alive',
               'Content-Length': '44',
               'Content-Type': 'application/x-www-form-urlencoded',
               'Cookie': 'SessionId=%s' % getcookie(),
               'Host': 'ngw.bupt.edu.cn',
               'Origin': 'http://ngw.bupt.edu.cn',
               'Referer': 'http://ngw.bupt.edu.cn/index',
               'Upgrade-Insecure-Requests': '1',
               'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/70.0.3538.67 Safari/537.36'}
    post_result = requests.post(url, data=postdata)
    response = post_result.content.decode("utf-8")
    # print(response)
    return response


def logout(url):
    requests.get(url)


if __name__ == "__main__":
    url = 'http://ngw.bupt.edu.cn/index'
    result = requests.get(url=url)
    response = result.content.decode("utf-8")
    match = re.search('登录成功', response)
    # print(response)
    if match:
        confirm_msg = input("Logout confirmed?(yes/no)")
        while True:
            if confirm_msg == "yes":
                url = "http://ngw.bupt.edu.cn/logout"
                logout(url)
                break
            elif confirm_msg == "no":
                break
            else:
                confirm_msg = input("Logout confirmed?(yes/no)")
    else:
        url = 'http://ngw.bupt.edu.cn/login'
        data = {
            'user': "学号",
            'pass': "登录密码",
            'line': ""
        }
        while True:
            print("请输入你的学号和密码")
            userid = input("学号：")
            password = input("密码：")
            data["user"] = userid
            data["pass"] = password
            response = login(data)
            match = re.search('登录成功', response)
            if match:
                print("登录成功")
                break
            else:
                print("登录失败")
