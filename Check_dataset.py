from os import walk, path, makedirs
import matplotlib.pyplot as plt
import numpy as np
import json

def check_class(p):
    '''
    class 당 총 갯수를 확인하기 위한 함수.
    여기서는 un-classified 라는 class도 존재하지만 학습에서는 제외시켰다.
    학습 시에는 destroyed에 포함시켰다.
    :param p: 검사할 folder path
    :return: reture은 없고 dictionary 형태로 각 class별 갯수를 출력한다.
    '''
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
    labels = d.keys()
    values = d.values()

    # https://dailyheumsi.tistory.com/98
    # 막대 그래프로 갯수 출력
    fig, ax = plt.subplots()
    ax.bar(labels, values, 0.35)
    for p in ax.patches:
        left, bottom, width, height = p.get_bbox().bounds
        ax.annotate("%d" % (height), (left + width / 2, height * 1.01), ha='center')
    ax.set_title('The number of class')
    plt.show()

def check_shape(p):
    '''
    Directory 안에 있는 파일들의 shape 분포와 평균 shape에 대한 확인
    :param p: 검사할 directory
    :return: x
    '''
    a = next(walk(p))[1]

    # set을 사용해서 공통은 제외
    store = set()
    mean_h = 0
    mean_w = 0
    num = 0.0

    for directory in a:
        post_j = path.join(p, directory, directory + '_post_disaster.json')

        with open(post_j) as labels:
            c = json.load(labels)

        for content in c['features']['xy']:
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

            # 평균을 위해 덧셈
            mean_h += max_y - min_y
            mean_w += max_x - min_x

            # 분포를 위해 set에 추가
            store.add((max_x - min_x, max_y - min_y))

            # 총 갯수
            num += 1.0

    # 평균 내기
    mean_h /= num
    mean_w /= num

    # 그래프로 출력
    fig = plt.figure(figsize=(10, 10))
    ax1 = fig.add_subplot(221)
    ax2 = fig.add_subplot(222)

    ax1.title.set_text('Distribution for height and width')
    for x, y in store:
        ax1.scatter(x, y)
    ax1.set_xlabel('width', fontsize=19)
    ax1.set_ylabel('height', fontsize=19)

    ax2.title.set_text('Average of height and width')
    ax2.bar(('width', 'height'), (mean_w, mean_h), 0.35)
    for p in ax2.patches:
        left, bottom, width, height = p.get_bbox().bounds
        ax2.annotate("%0.4f" % (height), (left + width / 2, height * 1.01), ha='center')

    plt.show()

if __name__ == '__main__':
    check_class('data/pre_train')
    check_shape('data/pre_train')