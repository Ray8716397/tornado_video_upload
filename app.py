# coding=utf-8
# @CREATE_TIME: 2021/4/7 下午5:57
# @LAST_MODIFIED: 2021/4/7 下午5:57
# @FILE: app.py
# @AUTHOR: Ray
import os
from configparser import ConfigParser

import tornado.ioloop
import tornado.web

import socketio

from handlers import MainHandler

# load config.ini
config_file = "./config.ini"
config = ConfigParser()
config.read(config_file)

users_ws_count = {}  # [req sid]
users_db = [f"7001_%03d" % i for i in range(1, 6)]
users_db.extend([f"7002_%03d" % i for i in range(1, 27)])

sio = socketio.AsyncServer(async_mode='tornado')



@sio.event
async def my_event(sid, message):
    await sio.emit('my_response', {'data': message['data']}, room=sid)


@sio.event
async def my_broadcast_event(sid, message):
    await sio.emit('my_response', {'data': message['data']})


@sio.event
async def join(sid, message):
    sio.enter_room(sid, message['room'])
    await sio.emit('my_response', {'data': 'Entered room: ' + message['room']},
                   room=sid)


@sio.event
async def leave(sid, message):
    sio.leave_room(sid, message['room'])
    await sio.emit('my_response', {'data': 'Left room: ' + message['room']},
                   room=sid)


@sio.event
async def close_room(sid, message):
    await sio.emit('my_response',
                   {'data': 'Room ' + message['room'] + ' is closing.'},
                   room=message['room'])
    await sio.close_room(message['room'])


@sio.event
async def my_room_event(sid, message):
    await sio.emit('my_response', {'data': message['data']},
                   room=message['room'])


@sio.event
async def disconnect_request(sid):
    await sio.disconnect(sid)


@sio.event
async def connect(sid, environ):
    print(environ['QUERY_STRING'])
    # sio.start_background_task(background_task)
    await sio.emit('my_response', {'data': 'Connected', 'count': 0}, room=sid)


@sio.event
def disconnect(sid):
    print('Client disconnected')


def main():
    class Application(tornado.web.Application):  # 引入Application类，重写方法，这样做的好处在于可以自定义，添加另一些功能
        def __init__(self):
            handlers = [
                tornado.web.url(r'/', MainHandler.IndexHandler, name='index'),
                tornado.web.url(r'/camera', MainHandler.CamHandler, name='camera'),
                tornado.web.url(r'/login', MainHandler.LoginHandler, name='login'),
                tornado.web.url(r'/logout', MainHandler.LogoutHandler, name='logout')
            ]
            settings = dict(
                debug=True,  # 调试模式，修改后自动重启服务，不需要自动重启，生产情况下切勿开启，安全性
                template_path=os.path.join(os.path.dirname(__file__), "templates"),
                static_path=os.path.join(os.path.dirname(__file__), "static"),
                login_url='/login',  # 没有登录则跳转至此
                cookie_secret='guijutech@!',  # 加密cookie的字符串
                pycket={  # 固定写法packet，用于保存用户登录信息
                    'engine': 'redis',
                    'storage': {
                        'host': 'localhost',
                        'port': 6379,
                        'db_sessions': 5,
                        'db_notifications': 11,
                        'max_connections': 2 ** 33,
                    },
                    'cookie': {
                        'expires_days': 38,
                        'max_age': 100
                    }
                }
            )

            super(Application, self).__init__(handlers,
                                              **settings)  # 用super方法将父类的init方法重新执行一遍，然后将handlers和settings传进去，完成初始化

    app = Application()  # 实例化
    app.listen(config["h5record_video"].getint("flask_port"))
    tornado.ioloop.IOLoop.current().start()


if __name__ == "__main__":
    main()
