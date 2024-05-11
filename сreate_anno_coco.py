import os
import json
from PIL import Image


class COCOFormatMaker():
    """
    Create annotation in COCO format just by dir with images and boxes
    """
    def __init__(self, train_path: str, val_path: str = None):
        self.train = train_path
        self.val = val_path
        self.categ = []
        self.data = {"images": [], "annotations": [], "categories": self.categ}

    def categories(self):
        with open('/Users/alina/PycharmProjects/obj_detection/datasets/traffic_sign/label_map.json', 'r') as f:
            categ = json.load(f)
        for key in categ.keys():
            d = {}
            d['id'] = int(categ[key])
            d['name'] = key
            self.categ.append(d)

    def img_info(self, name: str, img_id: int, img_path: str):
        data = {}
        data['id'] = img_id
        data['file_name'] = name
        with Image.open(img_path+'images/'+name) as img:
            width, height = img.size
        data['height'] = height
        data['width'] = width

        return data

    def annotation_inf(self, file_data: str, img_id: int, ann_id: int):
        data = {}
        data['id'] = ann_id
        data['image_id'] = img_id

        info = file_data.split(' ')

        # because start from 1
        data['category_id'] = int(info[0]) + 1
        data['bbox'] = [float(i) for i in info[1:]]
        data['iscrowd'] = 0
        data['area'] = data['bbox'][2] * data['bbox'][3]
        data['segmentation'] = []

        return data

    def create_anno(self, path_main_json, path_val_json):
        img_id = 0
        ann_id = 0
        loss = 0

        self.categories()

        for img in os.listdir(self.train + 'images/'):
            try:
                info_images = self.img_info(img, img_id, self.train)
            except:
                loss += 1
                continue
            img_id += 1

            try:
                box_data = open(self.train + 'labels/' + img[:-4] + '.txt', 'r').read()
            except:
                loss += 1
                continue

            self.data['images'].append(info_images)
            for box in box_data.split('\n'):
                if box != '':
                    self.data['annotations'].append(self.annotation_inf(box, img_id, ann_id))
                    ann_id += 1

        with open(path_main_json, "w") as json_file:
            json.dump(self.data, json_file)

        if self.val:
            self.data = {"images": [], "annotations": [], "categories": self.categ}
            img_id = 0
            ann_id = 0
            for img in os.listdir(self.val + 'images/'):
                try:
                    self.data['images'].append(self.img_info(img, img_id, self.val))
                except:
                    loss += 1
                    continue
                img_id += 1
                box_data = open(self.val + 'labels/' + img[:-4] + '.txt', 'r').read()

                for box in box_data.split('\n'):
                    if box != '':
                        self.data['annotations'].append(self.annotation_inf(box, img_id, ann_id))
                        ann_id += 1

        with open(path_val_json, "w") as json_file:
            json.dump(self.data, json_file)

        print(loss)


if __name__ == "__main__":
    creator = COCOFormatMaker(
        "/Users/alina/PycharmProjects/obj_detection/datasets/traffic_sign/dataset/train/",
        "/Users/alina/PycharmProjects/obj_detection/datasets/traffic_sign/dataset/valid/"
    )
    save_path = "/Users/alina/PycharmProjects/obj_detection/datasets/traffic_sign/dataset"
    creator.create_anno(save_path+'/train/train_anno.json', save_path+'/valid/val_anno.json')