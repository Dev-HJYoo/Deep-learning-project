from os import walk, path, makedirs
import matplotlib.pyplot as plt
import numpy as np
import json

def check_scale(scale):
    '''

    :param scale: 손상 정도에 따른 string 값
    :return: 손상 정도에 따른 int 값
    '''
    chk = 0
    if scale == 'no-damage':
        chk = 0
    elif scale == 'minor-damage':
        chk = 1
    elif scale == 'major-damage':
        chk = 2
    else :
        chk = 3

    return chk


def check_class(p):
    a = next(walk(p))[1]

    d = dict()
    for directory in a:
        post_j = path.join(p, directory, directory + '_post_disaster.json')

        with open(post_j) as labels:
            c = json.load(labels)

        for content in c['features']['xy']:
            # scale의 경우 해당 집이 어떤 피해를 입은지 알려준다.
            scale = content['properties']['subtype']

            if scale in d:
                d[scale] += 1
            else:
                d[scale] = 1

    print(d.items())

if __name__ == '__main__':
    check_class('data/pre_train')