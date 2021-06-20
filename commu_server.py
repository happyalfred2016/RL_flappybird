import eventlet
import socketio
import io
import torch
from PIL import Image
import numpy as np
import logging
import time
from DQN_test import Net, DQN, MEMORY_CAPACITY

logging.basicConfig(level='INFO')

sio = socketio.Server(binary=True)
# app = socketio.WSGIApp(sio, static_files={
#     '/': {'content_type': 'text/html', 'filename': 'index.html'}
# })
app = socketio.WSGIApp(sio)


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
    imMat = np.asarray(picture)[:, :, :3].T

    # TODO: resize the image
    return imMat


dqn = DQN()
record = dict(
    first_pic=True,
    ep_r=0,
    state_last=None,
    score_last=0,
    i_episode=0
)


def rl_model(bytes, status: list):
    state = byte2img(bytes)
    done, score = status
    reward = 0
    if score != record['score_last']:
        reward = 1
    if done == 1:
        reward = -1
    logging.info('Obs Received')

    if done or record['first_pic']:
        sio.emit('action', 1)
        record['ep_r'] = 0
        record['score_last'] = 0
        record['first_pic'] = False
        record['state_last'] = state
        record['score_last'] = score
        return

    # take action based on the current state
    action_last = dqn.choose_action(record['state_last'])
    #action_last = 1
    sio.emit('action', int(action_last))

    # store the transitions of states
    dqn.store_transition(record['state_last'], action_last, reward, state)

    record['ep_r'] += 1
    if dqn.memory_counter > MEMORY_CAPACITY:
        dqn.learn()
        print('learn')
        if done:
            record['i_episode'] += 1
            print('Ep: ', record['i_episode'], ' |', 'Ep_r: ', record['ep_r'])
    if done:
        return
    # use next state to update the current state.
    record['state_last'] = state
    record['score_last'] = score


    time.sleep(0.1)


if __name__ == '__main__':
    eventlet.wsgi.server(eventlet.listen(('127.0.0.1', 5000)), app, )
