# coding=utf-8
# @CREATE_TIME: 2021/4/7 下午7:08
# @LAST_MODIFIED: 2021/4/7 下午7:08
# @FILE: MainHandler.py
# @AUTHOR: Ray
import datetime
from typing import Optional, Awaitable

import tornado.web
from configparser import ConfigParser
from pycket.session import SessionMixin

# load config.ini
config_file = "./config.ini"
config = ConfigParser()
config.read(config_file)

version = datetime.datetime.now().timestamp()
users_db = [f"7001_%03d" % i for i in range(1, 6)]
users_db.extend([f"7002_%03d" % i for i in range(1, 27)])


class BaseHandler(tornado.web.RequestHandler, SessionMixin):

    def data_received(self, chunk: bytes) -> Optional[Awaitable[None]]:
        pass

    def get_current_user(self):  # 重写get_current_user()方法
        return self.session.get('id', None)  # session是一种会话状态，跟数据库的session可能不一样


class IndexHandler(BaseHandler):
    @tornado.web.authenticated  # @tornado.web.authenticated装饰器包裹get方法时，表示这个方法只有在用户合法时才会调用，authenticated装饰器会调用get_current_user()方法获取current_user的值，若值为False，则重定向到登录url装饰器判断有没有登录，如果没有则跳转到配置的路由下去，但是要在app.py里面设置login_url
    def get(self, *args, **kwargs):
        self.redirect('/camera')


class LoginHandler(BaseHandler):
    def get(self, *args, **kwargs):
        if self.current_user:  # 若用户已登录
            self.redirect('/')  # 那么直接跳转到主页
        else:
            nextname = self.get_argument('next', '')  # 将原来的路由赋值给nextname
            self.render('login.html', nextname=nextname, msg='')  # 否则去登录界面

    def post(self, *args, **kwargs):
        username = self.get_argument('user_id', None)

        if username in users_db:
            self.session.set('id', username)  # 将前面设置的cookie设置为username，保存用户登录信息
            self.redirect('/camera')
        else:
            self.render('login.html', title="michimura", msg='ユーザが存在しません')  # 不通过，有问题


class LogoutHandler(BaseHandler):
    @tornado.web.authenticated
    def get(self, *args, **kwargs):
        # self.session.set('user_info','') #将用户的cookie清除
        self.session.delete('user_info')
        self.redirect('/login')


class CamHandler(BaseHandler):
    @tornado.web.authenticated
    def get(self, *args, **kwargs):
        self.render('camera_record.html', interval=config["h5record_video"].getint("interval"),
                    chunk_num=config["h5record_video"].getint("chunk_num"),
                    websocket_uri=f"record_video", version=version, id=self.session.get('id'))
