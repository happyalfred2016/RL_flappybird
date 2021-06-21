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
    action_last=None,
    score_last=0,
    i_episode=0,
    last_score=0,
    done=False,
    learning=False
)


def rl_model(bytes, status: list):
    logging.info('Obs Received')
    state = byte2img(bytes)

    

    done, score = status
    if record['done']:
        if done != 1:  # if reset
            record['state_last'] = None
            record['action_last'] = None
            record['first_pic'] = True
            record['done'] = False
            record['last_score'] = 0
        sio.emit('action', 0)

    if not record['done']:
        if done == 1:
            reward = -1
            record['done'] = True
            logging.info('Done')
        elif score > record['last_score']:
            reward = 1
            record['last_score'] = score
        else:
            reward = 0

        if record['first_pic']:
            action_last = 1
            record['ep_r'] = 0
            record['first_pic'] = False
        else:
            # import matplotlib.pyplot as plt
            # # plt.figure()
            # plt.imshow(record['state_last'].transpose(2,1,0))
            # plt.show()
            # plt.imshow(state.transpose(2,1,0))
            # plt.show()
            
            
            dqn.store_transition(record['state_last'], record['action_last'], reward, state)
            if record['learning']:
                action_last = dqn.choose_action(state)
            else:
                action_last = np.random.randint(0, 2)

        record['state_last'] = state
        record['action_last'] = action_last
        sio.emit('action', int(action_last))

        record['ep_r'] += 1
        if dqn.memory_counter > MEMORY_CAPACITY:
            record['learning'] = True
            dqn.learn()
            logging.info('learning')
        else:
            logging.info('Memory: %d' % dqn.memory_counter)
            # if done:
            #     record['i_episode'] += 1
            #
            # logging.info('Ep: ', record['i_episode'], ' |', 'Ep_r: ', record['ep_r'])


if __name__ == '__main__':
    eventlet.wsgi.server(eventlet.listen(('127.0.0.1', 5000)), app, )
