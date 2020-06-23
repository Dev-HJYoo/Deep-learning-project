from os import walk, path, makedirs
import matplotlib.pyplot as plt
import numpy as np
import json
import cv2

def calcurate_pre_post(pre, post):
    # pre/post image 높이와 너비
    pre_h, pre_w = pre.shape[0], pre.shape[1]
    post_h, post_w = post.shape[0], post.shape[1]

    post = np.transpose(post, (2, 0, 1))

    # pre-image 높이와 너비에 맞추기
    new_post = np.zeros((3, pre_h, pre_w))
    for i in range(3):
        # 높이 비교
        if pre_h > post_h:
            new_post[i, :post_h, :post_w] = post[i, :, :pre_w]
        elif pre_h < post_h:
            new_post[i, :, :post_w] = post[i, :-(post_h - pre_h), :pre_w]

        # 너비 비교
        if pre_w > post_w:
            new_post[i, :post_h, :post_w] = post[i, : pre_h, :]
        elif pre_w < post_w:
            new_post[i, :post_h, :] = post[i, :pre_h, :-(post_w - pre_w)]

    new_post = np.transpose(new_post, (1, 2, 0))
    return abs(pre - new_post)

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
    else:
        chk = 3

    return chk



def main(p):
    directory = next(walk(p))[1][0]
    pre_i = path.join(p, directory, directory + '_pre_disaster.png')
    post_i = path.join(p, directory, directory + '_post_disaster.png')
    pre_j = path.join(p, directory, directory + '_pre_disaster.json')
    post_j = path.join(p, directory, directory + '_post_disaster.json')

    originals = []  # pre images
    orii = []  # post images
    types = []  # damage types
    total = []  # pre - post images
    idx = 0  # total images number

    # pre image를 가져오기
    pre = plt.imread(pre_i)
    with open(pre_j) as labels:  # 이렇게 해야 json 파일 연다.
        b = json.load(labels)

    # json 파일에서 xy축 가져오기
    for content in b['features']['xy']:
        # json에서 추출된거 뭔지 알고 싶을 때
        # print(content)

        # 점들 뽑기
        Dots = content['wkt'].split('((')[1].split('))')[0].split(',')  # 이건... 노가다의 결과

        # 최대 최소점 찾기
        min_x = 2000
        max_x = -1
        min_y = 2000
        max_y = -1
        for dot in Dots:
            dot = dot.strip().split(' ')
            x = int(float(dot[0]))
            y = int(float(dot[1]))
            min_x = min(min_x, x)
            max_x = max(max_x, x)
            min_y = min(min_y, y)
            max_y = max(max_y, y)

        # 최대/최소 점 알고 싶을 때
        # print(min_x, min_y, max_x, max_y)

        # transpose는 잘 놔누기 쉽게 할려고 한거.
        b = pre[min_y:max_y]
        b = np.transpose(b, (1, 0, 2))
        b = b[min_x:max_x]
        b = np.transpose(b, (1, 0, 2))

        # total에 저장 -> 추후에 밑에서 post에서 뺀다.
        total += [b]
        originals += [b]
        # pre 이미지 확인
        # plt.imshow(b)
        # plt.show()

    # 아무것도 안 들어 있을 때 -> 예를들어 처음에 집들이 없는 경우
    if total == []:
        print("This area don't have house.")
        return

    # post image 가져오기 -> 위의 pre image와 똑같다. 단지, total에 저장할 때 pre image와 비교해서 저장한다.
    post = plt.imread(post_i)
    with open(post_j) as labels:
        c = json.load(labels)

    for content in c['features']['xy']:
        # scale의 경우 해당 집이 어떤 피해를 입은지 알려준다.
        scale = content['properties']['subtype']
        types += [scale]  # 제외

        # 점들 추출 및 최소/최대 점 찾기
        Dots = content['wkt'].split('((')[1].split('))')[0].split(',')
        min_x = 2000
        max_x = -1
        min_y = 2000
        max_y = -1
        for dot in Dots:
            dot = dot.strip().split(' ')
            x = int(float(dot[0]))
            y = int(float(dot[1]))
            min_x = min(min_x, x)
            max_x = max(max_x, x)
            min_y = min(min_y, y)
            max_y = max(max_y, y)

        # 최소점 최대점 로 만들기
        b = post[min_y:max_y]
        b = np.transpose(b, (1, 0, 2))
        b = b[min_x:max_x]
        b = np.transpose(b, (1, 0, 2))

        # pre - post 하기 (이게 pre와 post의 차이를 알기위해 하는 것)
        if total[idx].shape != b.shape:  # 모양이 다른 경우
            total[idx] = calcurate_pre_post(total[idx], b)
        else:
            total[idx] = abs(total[idx] - b)
        orii += [b]  # 제거
        idx += 1

    for to, ors, oi, ty in zip(total, originals, orii, types):
        fig = plt.figure(figsize=(4, 10))
        ax1 = fig.add_subplot(321)
        ax2 = fig.add_subplot(322)
        ax3 = fig.add_subplot(323)
        ax1.title.set_text('pre image')
        ax2.title.set_text('post image')
        ax3.title.set_text('pre-post image')
        plt.subplot(311)
        plt.suptitle('pre image')
        plt.imshow(to)
        plt.subplot(312)
        plt.suptitle('post image')
        plt.imshow(ors)
        plt.subplot(313)
        plt.suptitle('pre-post image')
        plt.imshow(oi)
        plt.show()

if __name__ == '__main__':
    main('test')