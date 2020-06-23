from os import walk, path, makedirs
import matplotlib.pyplot as plt
import numpy as np
import json
import cv2


# pre에 대한 shape을 기준으로 padding 하거나 crop 해서 맞추기!!
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


def test_module(p, directory, pre_i, post_i, pre_j, post_j):
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
        plt.subplot(311)
        plt.imshow(to)
        plt.subplot(312)
        plt.imshow(ors)
        plt.subplot(313)
        plt.imshow(oi)
        plt.show()


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

def make_tensor(p):
    '''

    :param p: directory path
    :return: (total, total_scale) -> (전체 집 이미지, 전체 집 손상 정도)
    '''
    # p 이하의 디렉토리 다 불러오기 -> next(walk(p))[1]
    a = next(walk(p))[1]

    # case 갯수
    i = 0

    # 전체 이미지 저장하는 리스트
    total = []

    # 전체 이미지 피해 상황 리스트
    total_scale = []

    # test 넘버
    test_number = 3000

    # total의 index number
    idx = 0

    # max/min 높이와 너비 찾기
    min_h = 1000
    min_w = 1000
    max_h = -1
    max_w = -1

    # 평균 내기
    mean_h = 0
    mean_w = 0

    # directory 당 시작.
    '''
        directory 모양
        p ---- guatemala-volcano_00000 ---  이름_post_disaster.json
            |                           |
            |                           --  이름_post_disaster.png
            |                           |
            |                           --  이름_pre_disaster.json
            |                           |
            |                           --  이름_pre_disaster.png
            |
            -- guatemala-vocano_00001 --- ...
            |
            ...
    '''
    for directory in a:
        # 각각 전/후 이미지와 json 파일 분리
        pre_i = path.join(p, directory, directory + '_pre_disaster.png')
        post_i = path.join(p, directory, directory + '_post_disaster.png')
        pre_j = path.join(p, directory, directory + '_pre_disaster.json')
        post_j = path.join(p, directory, directory + '_post_disaster.json')

        # pre image를 가져오기
        pre = plt.imread(pre_i)

        # json 파일 open
        with open(pre_j) as labels:
            pre_contents = json.load(labels)

        # json 파일에서 xy축 가져오기
        for content in pre_contents['features']['xy']:
            # json에서 추출된거 뭔지 알고 싶을 때
            # print(content)

            # 점들 뽑기
            # 본 데이터셋은 집을 다각형으로 annotation 했음.
            Dots = content['wkt'].split('((')[1].split('))')[0].split(',')  # 이건... 노가다의 결과

            # 최대 최소점 찾기
            # 사각형으로 만들
            pre_min_x = 2000
            pre_max_x = -1
            pre_min_y = 2000
            pre_max_y = -1
            for dot in Dots:
                dot = dot.strip().split(' ')
                x = int(float(dot[0])) if float(dot[0]) >= 0 else 0
                y = int(float(dot[1])) if float(dot[1]) >= 0 else 0
                pre_min_x = min(pre_min_x, x)
                pre_max_x = max(pre_max_x, x)
                pre_min_y = min(pre_min_y, y)
                pre_max_y = max(pre_max_y, y)

            # 최대/최소 점 알고 싶을 때
            # print(min_x, min_y, max_x, max_y)

            # transpose는 잘 놔누기 쉽게 할려고 한거.
            c = pre[pre_min_y:pre_max_y]
            c = np.transpose(c, (1, 0, 2))
            c = c[pre_min_x:pre_max_x]
            c = np.transpose(c, (1, 0, 2))

            # total에 저장 -> 추후에 밑에서 post에서 뺀다.
            total += [c]

            # pre 이미지 확인
            # plt.imshow(b)
            # plt.show()

        # 아무것도 안 들어 있을 때 -> 예를들어 처음 이미지에 집이 없는 경우
        if total == []:
            i += 1
            continue

        # 인덱스와 현재 전처리 중인 것 확인할 때
        # print(i, pre_i)

        # post-image 가져오기 -> 위의 pre-image 와 똑같다. 단지, total에 저장할 때 pre-image 와 비교해서 저장한다.
        post = plt.imread(post_i)
        with open(post_j) as labels:
            post_contents = json.load(labels)

        for content in post_contents['features']['xy']:
            # scale의 경우 해당 집이 어떤 피해를 입은지 알려준다.
            scale = content['properties']['subtype']

            # annotation 안 된것은 제외
            # https://wikidocs.net/16040
            if scale == 'un-classified':
                del total[idx]
                continue
            total_scale += [check_scale(scale)]

            # 점들 추출 및 최소/최대 점 찾기
            post_Dots = content['wkt'].split('((')[1].split('))')[0].split(',')
            min_x = 2000
            max_x = -1
            min_y = 2000
            max_y = -1
            for dot in post_Dots:
                dot = dot.strip().split(' ')
                x = int(float(dot[0])) if float(dot[0]) >= 0 else 0
                y = int(float(dot[1])) if float(dot[1]) >= 0 else 0
                min_x = min(min_x, x)
                max_x = max(max_x, x)
                min_y = min(min_y, y)
                max_y = max(max_y, y)

            mean_h += max_y - min_y
            mean_w += max_x - min_x

            # 최소점 최대점 로 만들기
            b = post[min_y:max_y]
            b = np.transpose(b, (1, 0, 2))
            b = b[min_x:max_x]
            b = np.transpose(b, (1, 0, 2))

            # post 이미지 확인
            # plt.imshow(b)
            # plt.show()

            # pre - post 하기 (이게 pre와 post의 차이를 알기위해 하는 것)
            if total[idx].shape != b.shape:  # 모양이 다른 경우
                total[idx] = calcurate_pre_post(total[idx], b)
            else:
                total[idx] = abs(total[idx] - b)
            idx += 1

        # image와 json 파일 확인
        # print(pre_i, post_i, pre_j, post_j)

        # 테스트 모듈
        if i == test_number:
            test_module(p, directory, pre_i, post_i, pre_j, post_j)

        min_h = min(min_h, total[idx-1].shape[0])
        min_w = min(min_w, total[idx-1].shape[1])
        max_h = max(max_h, total[idx-1].shape[0])
        max_w = max(max_w, total[idx-1].shape[1])
        i += 1

        # 높이나 너빅 0 인 경우를 확인 하는 것.
        if total[idx-1].shape[0] <= 0 or total[idx-1].shape[1] <= 0:
            print(pre_i)
            print(total[idx-1].shape[0])
            print(total[idx-1].shape[1])
            print('pre', pre_min_y, pre_max_y, pre_min_x, pre_max_x)
            print('post', min_y, max_y, min_x, max_x)
            print(post_Dots)
            plt.imshow(c)
            plt.show()
            plt.imshow(b)
            plt.show()

    print('Data Processing Success!')
    print(f'max_h = {max_h}\tmax_w = {max_w}\tmin_h = {min_h}\tmin_w = {min_w}')
    print(f'mean_h = {mean_h / len(total)}\tmean_w = {mean_w / len(total)}')
    print(len(total))
    return total, total_scale


if __name__ == '__main__':
    make_tensor('data/pre_train')
