from os import walk, path, makedirs
from shutil import copy2 as cp
'''
    결과적으로 start에 속해 있는 파일들을 dest에 저장한다.
    <start 모양>
    start ----- images ----- 1_pre.jpg 1_post.jpg 2_pre.jpg 2_post.jpg ...
          |
          |
          ----- labels ----- 1_pre.json 1_post.json 2_pre.json 2_post.json ...
    
'''

# 파일 이름들 가져오기
def get_files(base_dir):
    # Minmizing (halfing) list to just pre image files
    base_dir = base_dir + 'images'
    files = [f for f in next(walk(base_dir))[2] if "pre" in f]

    return files

# pre 를 post로 바꾸기
def pre_to_post(file):
    file = file.split('_')
    file[2] = 'post'
    file = '_'.join(file)

    return file

if __name__ == "__main__":


    # data 뒤에 이름 바꾸기 ( train, test 해당 )
    start = 'data/test/'
    dest = 'data/pre_test/'

    # file 이름으로 가져오기
    files = get_files(start)
    i = 0

    for file in files:
        pre = file
        post = pre_to_post(pre)
        pre_json = pre.split('.')[0] + '.json'
        post_json = post.split('.')[0] + '.json'
        name = post.split('_post')[0]

        # make diractory
        makedirs(path.join(dest, name))

        cp(path.join(start, 'images', pre), path.join(dest, name))
        cp(path.join(start, 'images', post), path.join(dest, name))
        cp(path.join(start, 'labels', pre_json), path.join(dest, name))
        cp(path.join(start, 'labels', post_json), path.join(dest, name))

        if i % 1000 == 0:
            print(pre,post, name)
        i += 1