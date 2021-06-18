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


if __name__ == '__main__':
    # compatible to both polling and websocket
    #TODO: create learning modle

    eventlet.wsgi.server(eventlet.listen(('127.0.0.1', 5000)), app, )