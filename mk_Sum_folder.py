from os import walk, path, makedirs
from shutil import copy2 as cp
'''
    original folder를 New folder로 바꾼다.
    <original folder 모양>
    original ----- images ----- 1_pre.jpg 1_post.jpg 2_pre.jpg 2_post.jpg ...
            |
            |
             ----- labels ----- 1_pre.json 1_post.json 2_pre.json 2_post.json ...
             
    <New folder 모양>
    New  ----- ex1 ----- ex1_pre.jpg ex1_pre.json ex1_post.jpg ex1_post.json 
        |
        |
         ----- ex2 ----- ex2_pre.jpg ex2_pre.json ex2_post.jpg ex2_post.json
        |
        |
         ----- ex3 ----- ex3_pre.jpg ex3_pre.json ex3_post.jpg ex3_post.json
         .
         .
         .
        All dataset           
    
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
    # original는 고정, new를 바꾸면 됩니다.
    original = 'data/test/'
    new = 'data/pre_test/'

    # file 이름으로 가져오기
    files = get_files(original)
    i = 0

    for file in files:
        pre = file
        post = pre_to_post(pre)
        pre_json = pre.split('.')[0] + '.json'
        post_json = post.split('.')[0] + '.json'
        name = post.split('_post')[0]

        # make diractory
        makedirs(path.join(new, name))

        cp(path.join(original, 'images', pre), path.join(new, name))
        cp(path.join(original, 'images', post), path.join(new, name))
        cp(path.join(original, 'labels', pre_json), path.join(new, name))
        cp(path.join(original, 'labels', post_json), path.join(new, name))

        # 1000번 째 파일마다 출력
        if i % 1000 == 0:
            print(pre, post, name)
        i += 1
