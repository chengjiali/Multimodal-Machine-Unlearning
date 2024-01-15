import os
import json
import numpy as np
import pandas as pd


datasets = ['coco', 'flickr30k', 'vqa', 'nlvr']
datasets = []
df_size = 10000

for data in datasets:
    os.makedirs(f'Df/{data}', exist_ok=True)

    if data == 'coco':
        df = json.load(open('./lavis_cache/coco/annotations/coco_karpathy_train.json'))
        df = pd.DataFrame(df)
        dtrain_image_id = sorted(df['image'].unique())

    elif data == 'flickr30k':
        df = json.load(open('./lavis_cache/flickr30k/annotations/train.json'))
        df = pd.DataFrame(df)
        dtrain_image_id = sorted(df['image'].unique())

    elif data == 'vqa':
        df = json.load(open('./lavis_cache/coco/annotations/vqa_train.json'))
        df = pd.DataFrame(df)
        dtrain_image_id = df['question_id']
        dtrain_image_id = sorted(df['image'].unique())

    elif data == 'nlvr':
        df = json.load(open('./lavis_cache/nlvr/annotations/train.json'))
        df = pd.DataFrame(df)
        dtrain_image_id = df['images'].apply(tuple).unique()

    for seed in [42, 87, 21]:
        np.random.seed(seed)

#             # Image deletion
#             df_image_id = np.random.choice(dtrain_image_id, df_size, replace=False)

#             # Text deletion
#             df_text_id = np.random.choice(dtrain_text_id, df_size, replace=False)

        # Image-text deletion
        dtrain_image_id = np.random.choice(dtrain_image_id, df_size, replace=False)

        with open(f'Df/{data}/image-{seed}.txt', 'w') as f:
            for i in dtrain_image_id:
                f.write(str(i))
                f.write('\n')


data = 've'
os.makedirs(f'Df/{data}', exist_ok=True)

df = json.load(open('./lavis_cache/snli/annotations/ve_train.json'))
df = pd.DataFrame(df)
dtrain_image_id = set(df['image'].unique())


for seed in [42, 87, 21]:
    np.random.seed(seed)

    with open(f'Df/flickr30k/image-{seed}.txt', 'r') as f:
        flickr = f.readlines()
    flickr = [i.strip() for i in flickr if i.strip() != '']
    flickr = set([i[len('flickr30k-images/'):-len('.jpg')] for i in flickr])

    # Use existing Flickr deleted images as many as possible
    intersect = list(dtrain_image_id & flickr)
    extra = np.array(list(dtrain_image_id - flickr))
    num_to_select = df_size - len(intersect)
    print(len(intersect), num_to_select)
    
    # Image-text deletion
    sel = np.random.choice(extra, num_to_select, replace=False)

    with open(f'Df/{data}/image-{seed}.txt', 'w') as f:
        for i in sorted(intersect):
            f.write(str(i))
            f.write('\n')
        for i in sel:
            f.write(str(i))
            f.write('\n')
