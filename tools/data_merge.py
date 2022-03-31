import os
import json
import shutil

media_list = ["Breitbart","CBS","CNN","Fox","HuffPost","NPR","NYtimes","usatoday","wallstreet","washington"]
topic_list = ["climate-change","corporate-tax","drug-policy","gay-marriage","obamacare"]
root_path= './data'


for topic in topic_list:
    data_path = "/home/xiaobo/media-position/data/data_{}/42/all/original".format(topic)
    class_path = "/home/xiaobo/media-position/data/data_{}/42/all/class".format(topic)
    masked_path = "/home/xiaobo/media-position/data/data_{}/42/all/masked".format(topic)
    if not os.path.exists(class_path):
        os.makedirs(class_path)
    if not os.path.exists(masked_path):
        os.makedirs(masked_path)
    shutil.copy(os.path.join(data_path,'en.masked.bigram_outer'),os.path.join(masked_path,'en.masked.bigram_outer'))
    shutil.copy(os.path.join(data_path,'en.train'),os.path.join(class_path,'en.train'))
    shutil.copy(os.path.join(data_path,'en.valid'),os.path.join(class_path,'en.valid'))

for topic in topic_list:
    for file_type in ['train','valid']:
        data = list()
        for media in media_list:
            ori_file = os.path.join(os.path.join(os.path.join(os.path.join(os.path.join(root_path, "data_"+topic),"42"),media),"original"),"en."+file_type)
            with open(ori_file,mode='r',encoding='utf8') as fp:
                for line in fp.readlines():
                    data.append(line.strip())
        path = os.path.join(os.path.join(os.path.join(os.path.join(root_path, "data_"+topic),"42"),"all"),"original")
        if not os.path.exists(path):
            os.makedirs(path)
        target_file = os.path.join(os.path.join(os.path.join(os.path.join(os.path.join(root_path, "data_"+topic),"42"),"all"),"original"),"en."+file_type)
        with open(target_file, mode='w',encoding='utf8') as fp:
            for item in data:
                fp.write(item+'\n')
