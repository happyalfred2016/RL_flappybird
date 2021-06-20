import eventlet
import socketio
import io
from PIL import Image
import numpy as np
import logging
import time

logging.basicConfig(level='INFO')

sio = socketio.Server(binary=True)
# app = socketio.WSGIApp(sio, static_files={
#     '/': {'content_type': 'text/html', 'filename': 'index.html'}
# })
app = socketio.WSGIApp(sio)
# RL model
model = None
imgs = []


# TODO: fix the case that too many obs that algorithm can not handle.

@sio.on('connect')
def connect(sid, environ):
    sio.emit('test', '123')
    sio.emit('action', 0)
    print('connect ', sid)


@sio.on('chat')
def chat(sid, user, meg):
    print('chat: ', user, meg)


@sio.on('obs')
def obs(sid, pic_byte, status):
    '''

    Args:
        sid:
        pic_byte:
        status: list of status, finish?, reward?

    Returns:

    '''

    # rlflappybird.add_obs(pic_byte, status={})
    sio.start_background_task(rl_model, pic_byte, status)
    print('received bytes')


@sio.on('disconnect')
def disconnect(sid):
    print('disconnect ', sid)


def byte2img(bytes):
    # no idea why have a extra byte at the beginning
    picture_stream = io.BytesIO(bytes[1:])
    picture = Image.open(picture_stream)
    imMat = np.asarray(picture)[:, :, :3]

    # TODO: resize the image
    return imMat


def rl_model(bytes, status: list):
    img = byte2img(bytes)
    logging.info('Obs Reveived')
    logging.info('Status: %s' % status)

    # TODO: learning process
    # model.fit
    action = 1
    time.sleep(1.5)

    sio.emit('test', 123.)
    sio.emit('action', action)

    logging.info('Action Sent')


if __name__ == '__main__':
    eventlet.wsgi.server(eventlet.listen(('127.0.0.1', 5000)), app, )
