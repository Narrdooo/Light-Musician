import requests
import os

os.environ['NO_PROXY'] = 'weixin.qq.com'

 
class SendWeiXinWork():
    def __init__(self):
        self.CORP_ID = 'ww5689d4a9ac77b35d'  # 企业号的标识
        self.SECRET = 'gaCpGaAcVGeTo0RWpcFg27DCR_aZ67gRa16TCHRKXW4'  # 管理组凭证密钥
        self.AGENT_ID = '1000002'  # 应用ID
        self.token = self.get_token()
 
    def get_token(self):
        url = "https://qyapi.weixin.qq.com/cgi-bin/gettoken"
        data = {
            "corpid": self.CORP_ID,
            "corpsecret": self.SECRET
        }
        requests.packages.urllib3.disable_warnings()
        req = requests.get(url=url, params=data, verify=False)
        res = req.json()
        if res['errmsg'] == 'ok':
            return res["access_token"]
        else:
            return res
 
    def send(self, to_user, content):
        url = "https://qyapi.weixin.qq.com/cgi-bin/message/send?access_token=%s" % self.token
        data = {
            "touser": to_user,  # 发送个人就填用户账号
            #"toparty": to_user,  # 发送组内成员就填部门ID
            "msgtype": "text",
            "agentid": self.AGENT_ID,
            "text": {"content": content},
            "safe": "0"
        }
 
        req = requests.post(url=url, json=data)
        res = req.json()
        if res['errmsg'] == 'ok':
            return "send message sucessed"
        else:
            return res
 
 
if __name__ == '__main__':
    SendWeiXinWork = SendWeiXinWork()
    #SendWeiXinWork.send("ronnnhui", "测试成功🥰🥰")