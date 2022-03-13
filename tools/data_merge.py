import os
import json

media_list = ["Breitbart","CBS","CNN","Fox","HuffPost","NPR","NYtimes","usatoday","wallstreet","washington"]
topic_list = ["climate-change","corporate-tax","drug-policy","gay-marriage","obamacare"]
root_path= '/home/xiaobo/data/media-position'
for topic in topic_list:
    for media in media_list:
        data = list()
        for file_type in ['train','valid']:
            ori_file = os.path.join(os.path.join(os.path.join(os.path.join(os.path.join(root_path, "data_"+topic),"42"),media),"no_augmentation/1"),"en."+file_type)
            with open(ori_file,mode='r',encoding='utf8') as fp:
                for line in fp.readlines():
                    try:
                        item = json.loads(line.strip())
                        item = item['original']
                    except json.decoder.JSONDecodeError:
                        item = line.strip()
                    data.append({'original':item})
        target_file = os.path.join(os.path.join(os.path.join(os.path.join(os.path.join(root_path, "data_"+topic),"42"),media),"original"),"en.full")
        with open(target_file, mode='w',encoding='utf8') as fp:
            for item in data:
                fp.write(json.dumps(item,ensure_ascii=False)+'\n')
