# coding=utf-8
# @CREATE_TIME: 2021/4/8 上午10:08
# @LAST_MODIFIED: 2021/4/8 上午10:08
# @FILE: SioHandler.py
# @AUTHOR: Ray
import socketio

sio = socketio.AsyncServer(async_mode='tornado')


class RecordVideoNamespace(socketio.AsyncNamespace):
    async def on_connect(self, sid, environ):
        pass

    async def on_disconnect(self, sid):
        print(sid)
        await sio.disconnect(sid)

    async def on_my_event(self, sid, data):
        await self.emit('my_response', data)


sio.register_namespace(RecordVideoNamespace('/record_video'))
