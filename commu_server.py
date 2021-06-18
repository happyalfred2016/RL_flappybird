import eventlet
import socketio
import time

sio = socketio.Server(binary=True)
# app = socketio.WSGIApp(sio, static_files={
#     '/': {'content_type': 'text/html', 'filename': 'index.html'}
# })
app = socketio.WSGIApp(sio)

@sio.on('connect')
def connect(sid, environ):
    print('connect ', sid)

@sio.on('chat')
def chat(sid, user, meg):
    print('chat: ', user, meg)

@sio.on('picture')
def chat(sid, pic_byte):
    # TODO: learning process
    time.sleep(1)
    print('chat: ', pic_byte)

    # TODO: send action to client
    sio.emit('action', 'click')


@sio.on('disconnect')
def disconnect(sid):
    print('disconnect ', sid)


def background_thread():
    count = 0

    while True:
        socketio.sleep(5)

        count += 1

        t = time.strftime('%M:%S', time.localtime())

        # 获取系统时间（只取分:秒）

        cpus = psutil.cpu_percent(interval=None, percpu=True)

        # 获取系统cpu使用率 non-blocking

        socketio.emit('server_response',

                      {'data': [t, cpus], 'count': count},

                      namespace='/test')

        socketio.emit('messageEventNew',

                      {'encryptkey': 'key'},

                      namespace='/test')

        # 注意：这里不需要客户端连接的上下文，默认 broadcast = True

if __name__ == '__main__':
    # compatible to both polling and websocket
    #TODO: create learning modle



    eventlet.wsgi.server(eventlet.listen(('127.0.0.1', 5000)), app, )