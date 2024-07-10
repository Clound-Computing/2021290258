from variables import rumor_root, weibo_fake_root, weibo_real_root, weibo21_fake_root, weibo21_real_root, \
    get_workbook
import photohash
from tqdm import tqdm
import cv2
from PIL import Image, ImageTk
import torchvision.transforms as transforms
import pandas as pd
import os
import json
import numpy as np

to_tensor = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),]
)

def reload_xlsxs_gossipcop_llm(dataset_name=['gossip']):
    root_path = '/root/autodl-tmp/data/GossipCop-LLM-Data-examples'

    filenames = [
        'gossipcop_v3-1_style_based_fake.json',
        'gossipcop_v3-2_content_based_fake.json',
        'gossipcop_v3-3_integration_based_fake_tn200.json',
        'gossipcop_v3-4_story_based_fake.json',
        'gossipcop_v3-5_style_based_legitimate.json',
        'gossipcop_v3-7_integration_based_legitimate_tn300.json'
    ]

    with open(os.path.join(root_path, filenames[0])) as f:
        style_fake_data = json.load(f)
    with open(os.path.join(root_path, filenames[1])) as f:
        content_fake_data = json.load(f)
    with open(os.path.join(root_path, filenames[2])) as f:
        integration_fake_data = json.load(f)
    with open(os.path.join(root_path, filenames[3])) as f:
        story_fake_data = json.load(f)
    with open(os.path.join(root_path, filenames[4])) as f:
        style_legitimate_data = json.load(f)
    with open(os.path.join(root_path, filenames[5])) as f:
        integration_legitimate_data = json.load(f)

    real_records_list = []
    fake_records_list = []

    for key, item in style_fake_data.items():
        if item['origin_label'] == 'legitimate':
            label = 1
        elif item['origin_label'] == 'fake':
            label = 0
        else:
            raise NotImplementedError()
        content = item['origin_text']
        image_name = key
        ##  NEWS TYPE: 0 MULTIMODAL 1 IMAGE 2 TEXT
        if item['has_top_img']: news_type = 0
        else: news_type = 2
        records = {
            'content': content,
            'image': image_name,
            'label': label,
            'type': news_type
        }
        if label:
            real_records_list.append(records)
        else:
            fake_records_list.append(records)


        if item['generated_label'] == 'real':
            label = 1
        elif item['generated_label'] == 'fake':
            label = 0
        else:
            raise NotImplementedError()
        content = item['generated_text']
        image_name = key
        ##  NEWS TYPE: 0 MULTIMODAL 1 IMAGE 2 TEXT
        if item['has_top_img']: news_type = 0
        else: news_type = 2
        records = {
            'content': content,
            'image': image_name,
            'label': label,
            'type': news_type
        }
        if label:
            real_records_list.append(records)
        else:
            fake_records_list.append(records)

    
    for key, item in content_fake_data.items():
        if item['origin_label'] == 'legitimate':
            label = 1
        elif item['origin_label'] == 'fake':
            label = 0
        else:
            raise NotImplementedError()
        content = item['origin_text']
        image_name = key
        ##  NEWS TYPE: 0 MULTIMODAL 1 IMAGE 2 TEXT
        if item['has_top_img']: news_type = 0
        else: news_type = 2
        records = {
            'content': content,
            'image': image_name,
            'label': label,
            'type': news_type
        }
        if label:
            real_records_list.append(records)
        else:
            fake_records_list.append(records)
       
        label = 0
        content = item['generated_text_glm4']
        image_name = key
        ##  NEWS TYPE: 0 MULTIMODAL 1 IMAGE 2 TEXT
        if item['has_top_img']: news_type = 0
        else: news_type = 2
        records = {
            'content': content,
            'image': image_name,
            'label': label,
            'type': news_type
        }
        if label:
            real_records_list.append(records)
        else:
            fake_records_list.append(records)

    
    for key, item in integration_fake_data.items():
        if item['doc_1_label'] == 'legitimate':
            label = 1
        elif item['doc_1_label'] == 'fake':
            label = 0
        else:
            raise NotImplementedError()
        content = item['doc_1_text']
        image_name = item['doc_1_id']
        ##  NEWS TYPE: 0 MULTIMODAL 1 IMAGE 2 TEXT
        if item['has_top_img']: news_type = 0
        else: news_type = 2
        records = {
            'content': content,
            'image': image_name,
            'label': label,
            'type': news_type
        }
        if label:
            real_records_list.append(records)
        else:
            fake_records_list.append(records)

        if item['doc_2_label'] == 'legitimate':
            label = 1
        elif item['doc_2_label'] == 'fake':
            label = 0
        else:
            raise NotImplementedError()
        content = item['doc_2_text']
        image_name = item['doc_2_id']
        ##  NEWS TYPE: 0 MULTIMODAL 1 IMAGE 2 TEXT
        if item['has_top_img']: news_type = 0
        else: news_type = 2
        records = {
            'content': content,
            'image': image_name,
            'label': label,
            'type': news_type
        }
        if label:
            real_records_list.append(records)
        else:
            fake_records_list.append(records)

        label = 0
        content = item['generated_text']
        image_name = key
        ##  NEWS TYPE: 0 MULTIMODAL 1 IMAGE 2 TEXT
        if item['has_top_img']: news_type = 0
        else: news_type = 2
        records = {
            'content': content,
            'image': image_name,
            'label': label,
            'type': news_type
        }
        if label:
            real_records_list.append(records)
        else:
            fake_records_list.append(records)


    for key, item in story_fake_data.items():
        if item['origin_label'] == 'legitimate':
            label = 1
        elif item['origin_label'] == 'fake':
            label = 0
        else:
            raise NotImplementedError()
        content = item['origin_text']
        image_name = key
        ##  NEWS TYPE: 0 MULTIMODAL 1 IMAGE 2 TEXT
        if item['has_top_img']: news_type = 0
        else: news_type = 2
        records = {
            'content': content,
            'image': image_name,
            'label': label,
            'type': news_type
        }
        if label:
            real_records_list.append(records)
        else:
            fake_records_list.append(records)

        label = 0
        content = item['generated_text']
        image_name = key
        ##  NEWS TYPE: 0 MULTIMODAL 1 IMAGE 2 TEXT
        if item['has_top_img']: news_type = 0
        else: news_type = 2
        records = {
            'content': content,
            'image': image_name,
            'label': label,
            'type': news_type
        }
        if label:
            real_records_list.append(records)
        else:
            fake_records_list.append(records)


    for key, item in style_legitimate_data.items():
        
        label = 1
        content = item['origin_text']
        image_name = key
        ##  NEWS TYPE: 0 MULTIMODAL 1 IMAGE 2 TEXT
        if item['has_top_img']: news_type = 0
        else: news_type = 2
        records = {
            'content': content,
            'image': image_name,
            'label': label,
            'type': news_type
        }
        if label:
            real_records_list.append(records)
        else:
            fake_records_list.append(records)

        label = 1
        content = item['generated_text_t015']
        image_name = key
        ##  NEWS TYPE: 0 MULTIMODAL 1 IMAGE 2 TEXT
        if item['has_top_img']: news_type = 0
        else: news_type = 2
        records = {
            'content': content,
            'image': image_name,
            'label': label,
            'type': news_type
        }
        if label:
            real_records_list.append(records)
        else:
            fake_records_list.append(records)


    for key, item in integration_legitimate_data.items():
        label = 1
        content = item['doc_1_text']
        image_name = item['doc_1_id']
        ##  NEWS TYPE: 0 MULTIMODAL 1 IMAGE 2 TEXT
        if item['has_top_img']: news_type = 0
        else: news_type = 2
        records = {
            'content': content,
            'image': image_name,
            'label': label,
            'type': news_type
        }
        if label:
            real_records_list.append(records)
        else:
            fake_records_list.append(records)

        label = 1
        content = item['doc_2_text']
        image_name = item['doc_2_id']
        ##  NEWS TYPE: 0 MULTIMODAL 1 IMAGE 2 TEXT
        if item['has_top_img']: news_type = 0
        else: news_type = 2
        records = {
            'content': content,
            'image': image_name,
            'label': label,
            'type': news_type
        }
        if label:
            real_records_list.append(records)
        else:
            fake_records_list.append(records)


        label = 1
        content = item['generated_text_t01']
        image_name = key
        ##  NEWS TYPE: 0 MULTIMODAL 1 IMAGE 2 TEXT
        if item['has_top_img']: news_type = 0
        else: news_type = 2
        records = {
            'content': content,
            'image': image_name,
            'label': label,
            'type': news_type
        }
        if label:
            real_records_list.append(records)
        else:
            fake_records_list.append(records)

    fake_records_list.extend(
        fake_records_list[:len(real_records_list)-len(fake_records_list)]
    )

    train_indices = np.random.choice(list(range(len(real_records_list))), int(len(real_records_list)*0.9), replace=False)
    test_indices = [i for i in list(range(len(real_records_list))) if i not in train_indices]

    train_real_records_list = [real_records_list[i] for i in train_indices]
    train_fake_records_list = [fake_records_list[i] for i in train_indices]

    test_real_records_list = [real_records_list[i] for i in test_indices]
    test_fake_records_list = [fake_records_list[i] for i in test_indices]

    train_records_list = []
    train_records_list.extend(train_real_records_list)
    train_records_list.extend(train_fake_records_list)
    test_records_list = []
    test_records_list.extend(test_real_records_list)
    test_records_list.extend(test_fake_records_list)

    train_df = pd.DataFrame(train_records_list)
    train_df.to_excel('./dataset/gossipcop_LLM/train_datasets_gossipcop_LLM.xlsx')
    test_df = pd.DataFrame(test_records_list)
    test_df.to_excel('./dataset/gossipcop_LLM/test_datasets_gossipcop_LLM.xlsx')

    # # dataset_name = ['gossip', 'politi']
    # # fold_name = 'FakeNewsNet'
    # fold_name = dataset_name[0]
    # sub_folders = ['train','test']
    # num_invalid_format, num_too_small, num_invalid_hashing, num_length = 0, 0, 0, 0
    # hashing_not_allowed, ban_images = {}, os.listdir('E:/AAAI_dataset/AAAI_dataset/Images/ban_images')
    # for ban_image in ban_images:
    #     hash = photohash.average_hash('E:/AAAI_dataset/AAAI_dataset/Images/ban_images/' + ban_image)
    #     hashing_not_allowed[ban_image] = hash

    # for dataset in dataset_name:
    #     for sub_folder in sub_folders:
    #         xlsx = "{}/{}_{}.xlsx".format(root_path,dataset,sub_folder)
    #         sheet_rumor, rows_rumor = get_workbook(xlsx)
    #         records_list = []
    #         for i in tqdm(range(2, rows_rumor + 1)):  # title label content image B C E F
    #             records, news_type = {}, "multi"
    #             images_name = str(sheet_rumor['C' + str(i)].value)
    #             label = int(sheet_rumor['D' + str(i)].value)
    #             content = str(sheet_rumor['B' + str(i)].value)
    #             image_full_path = "{}/Images/{}_{}/{}".format(root_path,dataset,sub_folder,images_name)
    #             if len(content)<15:
    #                 news_type = "image"
    #                 print("Length not enough {} skip..".format(image_full_path))
    #                 num_length += 1
    #                 # if not NO_FILTER_OUT or label==1:
    #                 continue

    #             image_open = cv2.imread(image_full_path)
    #             if image_open is None:
    #                 image_open = Image.open(image_full_path)
    #                 image_tensor = to_tensor(image_open)
    #                 if image_open is None:
    #                     print("PIL still cannot open {} skip..".format(image_full_path))
    #                     num_invalid_format += 1
    #                     continue
    #                 image_width, image_height = image_tensor.shape[1], image_tensor.shape[2]
    #             else:
    #                 image_width, image_height = image_open.shape[0], image_open.shape[1]

    #             ## CONDUCT FILTERING OUT INVALID NEWS IF NOT NO_FILTER_OUT
    #             # IMAGE SIZE

    #             if image_width<100 or image_height<100:
    #                 news_type = "text"
    #                 print("Size too small {} skip..".format(image_full_path))
    #                 num_too_small += 1
    #                 # if not NO_FILTER_OUT or label==1:
    #                 continue
    #             # IMAGE HASHING
    #             found_invalid_hashing = False
    #             item1_hash = photohash.average_hash(image_full_path)
    #             for key in hashing_not_allowed:
    #                 item2_hash = hashing_not_allowed[key]
    #                 if item1_hash is not None and photohash.hashes_are_similar(item1_hash, item2_hash, tolerance=0.5):
    #                     found_invalid_hashing = True
    #                     break
    #             if found_invalid_hashing:
    #                 news_type = "text"
    #                 print("Invalid image found {} skip..".format(image_full_path))
    #                 num_invalid_hashing += 1
    #                 # if not NO_FILTER_OUT or label==1:
    #                 continue
    #             records['content'] = content
    #             records['image'] = images_name
    #             records['label'] = label
    #             ##  NEWS TYPE: 0 MULTIMODAL 1 IMAGE 2 TEXT
    #             records['type'] = news_type
    #             records_list.append(records)

    #         df = pd.DataFrame(records_list)
    #         df.to_excel('./dataset/{}/origin_do_not_modify/{}_{}.xlsx'.format(fold_name, dataset,sub_folder))
    # print("num_invalid_format, num_too_small, num_invalid_hashing, num_word_len {} {} {}"
    #       .format(num_invalid_format, num_too_small, num_invalid_hashing, num_length))

if __name__ == '__main__':
    reload_xlsxs_gossipcop_llm()