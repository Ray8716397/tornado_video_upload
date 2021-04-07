# coding=utf-8
# @CREATE_TIME: 2021/4/7 下午7:04
# @LAST_MODIFIED: 2021/4/7 下午7:04
# @FILE: BaseHandler.py
# @AUTHOR: Ray
from typing import Optional, Awaitable

from tornado.web import RequestHandler
from pycket.session import SessionMixin


class BaseHandler(RequestHandler, SessionMixin):
    def data_received(self, chunk: bytes) -> Optional[Awaitable[None]]:
        pass

    def get_current_user(self): #重写get_current_user()方法
        return self.session.get('id', None) #session是一种会话状态，跟数据库的session可能不一样
