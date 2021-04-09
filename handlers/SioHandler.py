# coding=utf-8
# @CREATE_TIME: 2021/4/8 上午10:08
# @LAST_MODIFIED: 2021/4/8 上午10:08
# @FILE: SioHandler.py
# @AUTHOR: Ray
import socketio
import redis
from urllib import parse

sio = socketio.AsyncServer(async_mode='tornado')
redis_pool = redis.ConnectionPool(host='localhost', port=6379, decode_responses=True, db=2)


class RecordVideoNamespace(socketio.AsyncNamespace):
    namespace = '/record_video'

    async def on_connect(self, sid, environ):
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

    async def on_disconnect(self, sid):
        session = await sio.get_session(sid, self.namespace)
        rc = redis.Redis(connection_pool=redis_pool)

        online_sid = rc.get(session['user_id'])
        if online_sid is not None and online_sid == sid:
            rc.delete(session['user_id'])
        await sio.disconnect(sid)

    async def on_my_event(self, sid, data):
        await self.emit('my_response', data, room=sid)


sio.register_namespace(RecordVideoNamespace(RecordVideoNamespace.namespace))
