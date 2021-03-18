import os
import random
republican_media_list= ['FoxNews', 'BreitbartNews']
democrat_media_list= ['CNN', 'nytimes', 'NPR']


media_dict = {"republican":republican_media_list, "democrat":democrat_media_list}
data_dir = '/data/xiaobo/media-position/data/source'

for name, media_list in media_dict.items():
  train_data = list()
  valid_data = list()
  data_path = os.path.join(data_dir,name)
  data_path = os.path.join(data_path,'twitter')
  if not os.path.exists(data_path):
    os.makedirs(data_path)
  for media_name in media_list:
    source_dir = os.path.join(os.path.join(data_dir, media_name),'twitter')
    with open(os.path.join(source_dir, 'en.train'),mode='r',encoding='utf8') as fp:
      for line in fp.readlines():
        train_data.append(line.strip())
    with open(os.path.join(source_dir, 'en.valid'),mode='r',encoding='utf8') as fp:
      for line in fp.readlines():
        valid_data.append(line.strip())
  random.seed(123)
  random.shuffle(train_data)
  random.shuffle(valid_data)

  train_file = os.path.join(data_path, 'en.train')
  with open(train_file, mode='w', encoding='utf8') as fp:
      for text in train_data:
          fp.write(text+'\n')
  eval_file = os.path.join(data_path, 'en.valid')
  with open(eval_file, mode='w', encoding='utf8') as fp:
      for text in valid_data:
          fp.write(text+'\n')

