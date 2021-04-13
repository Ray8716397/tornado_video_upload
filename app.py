# coding=utf-8
# @CREATE_TIME: 2021/4/7 下午5:57
# @LAST_MODIFIED: 2021/4/7 下午5:57
# @FILE: app.py
# @AUTHOR: Ray
import os
from configparser import ConfigParser

import socketio
import tornado.ioloop
import tornado.web


from handlers import MainHandler
from handlers.SioHandler import sio

# load config.ini
config_file = "./config.ini"
config = ConfigParser()
config.read(config_file)

users_ws_count = {}  # [req sid]


def main():
    class Application(tornado.web.Application):  # 引入Application类，重写方法，这样做的好处在于可以自定义，添加另一些功能
        def __init__(self):
            handlers = [
                tornado.web.url(r'/', MainHandler.IndexHandler, name='index'),
                tornado.web.url(r'/camera', MainHandler.CamHandler, name='camera'),
                tornado.web.url(r'/login', MainHandler.LoginHandler, name='login'),
                tornado.web.url(r'/logout', MainHandler.LogoutHandler, name='logout'),
                (r"/socket.io/", socketio.get_tornado_handler(sio))
            ]
            settings = dict(
                debug=True,  # 调试模式，修改后自动重启服务，不需要自动重启，生产情况下切勿开启，安全性
                template_path=os.path.join(os.path.dirname(__file__), "templates"),
                static_path=os.path.join(os.path.dirname(__file__), "static"),
                login_url='/login',  # 没有登录则跳转至此
                cookie_secret='guijutech@!',  # 加密cookie的字符串
                xsrf_cookie=True,
                pycket={  # 固定写法packet，用于保存用户登录信息
                    'engine': 'redis',
                    'storage': {
                        'host': 'localhost',
                        'port': 6379,
                        'db_sessions': 15,
                        'db_notifications': 11,
                        'max_connections': 2 ** 33,
                    },
                    'cookie': {
                        'expires_days': 1,
                        'max_age': 100
                    }
                }
            )

            super(Application, self).__init__(handlers,
                                              **settings)  # 用super方法将父类的init方法重新执行一遍，然后将handlers和settings传进去，完成初始化

    app = Application()  # 实例化
    app.listen(config["h5record_video"].getint("tornado_port"))
    tornado.ioloop.IOLoop.instance().start()


if __name__ == "__main__":
    main()
