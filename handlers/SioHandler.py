# coding=utf-8
# @CREATE_TIME: 2021/4/8 上午10:08
# @LAST_MODIFIED: 2021/4/8 上午10:08
# @FILE: SioHandler.py
# @AUTHOR: Ray
import datetime
import glob
import os
import traceback
from configparser import ConfigParser

import socketio
import redis
from urllib import parse

# load config.ini
from lib.common.common_util import logging

config_file = "./config.ini"
config = ConfigParser()
config.read(config_file)

sio = socketio.AsyncServer(async_mode='tornado')
redis_pool = redis.ConnectionPool(host='localhost', port=6379, decode_responses=True, db=2)
redis.Redis(connection_pool=redis_pool).flushdb()


class RecordVideoNamespace(socketio.AsyncNamespace):
    namespace = '/record_video'

    async def on_connect(self, sid, environ):
        try:
            dict_qs = parse.parse_qs(environ['QUERY_STRING'])
            if 'uid' in dict_qs.keys():
                user_id = dict_qs['uid'][0]

                await sio.save_session(sid, {'user_id': user_id}, self.namespace)

                rc = redis.Redis(connection_pool=redis_pool)
                if rc.get(user_id) is not None:
                    await self.emit('warning', {"msg": "already have one recording", "gohome": False}, room=sid,
                                    namespace=self.namespace)
                else:
                    rc.set(user_id, sid)
                    await self.emit('connect_succeed', None, room=sid,
                                    namespace=self.namespace)

        except Exception as e:
            logging(f"[websocket|connect][user_id{user_id}][{datetime.datetime.now().strftime('%Y-%m-%d_%I:%M:%S')}]:"
                    f"{traceback.format_exc()}",
                    f"logs/error.log")

    async def on_disconnect(self, sid):
        try:
            session = await sio.get_session(sid, self.namespace)
            user_id = session['user_id']
            rc = redis.Redis(connection_pool=redis_pool)

            online_sid = rc.get(user_id)
            if online_sid is not None and online_sid == sid:
                rc.delete(session['user_id'])
            await sio.disconnect(sid)

            for f in glob.glob(os.path.join(config['video_format'].get('input_dir'), f"{sid}.*")):
                os.remove(f)

        except Exception as e:
            logging(
                f"[websocket|disconnect][user_id{user_id}][{datetime.datetime.now().strftime('%Y-%m-%d_%I:%M:%S')}]:"
                f"{traceback.format_exc()}",
                f"logs/error.log")

    async def on_recv(self, sid, data):
        chunk_num = config["h5record_video"].getint("chunk_num")
        """"监听发送来的消息,并使用socketio向所有客户端发送消息"""
        session = await sio.get_session(sid, self.namespace)
        rc = redis.Redis(connection_pool=redis_pool)
        user_id = session.get('user_id')

        print(f"{user_id} recv data length: {len(data[2])}")
        if data is not None:
            try:
                if user_id not in rc.keys():
                    return

                start_record_time = data[0]
                cur_count = int(data[1])
                video_data = data[2]

                with open(os.path.join(config['video_format'].get('input_dir'),
                                       f"{sid}.{start_record_time}_%03d.{user_id}") % cur_count, 'wb') as f:
                    f.write(video_data)

                if cur_count == chunk_num:
                    # TODO
                    pass
                    # redis_db0.set(f"{user_id}*{start_record_time}",
                    #               "{session.get('fps')}")

            except Exception as e:
                logging(f"[websocket|recv][user_id|{user_id}][{datetime.datetime.now().strftime('%Y-%m-%d_%I:%M:%S')}]:"
                        f"{traceback.format_exc()}",
                        f"logs/error.log")


sio.register_namespace(RecordVideoNamespace(RecordVideoNamespace.namespace))
