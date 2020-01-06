# __author__ = 'ChienHung Chen'

from tkinter import *
import os, sys
from PIL import Image, ImageTk
import json
import cv2
import pickle
import numpy as np
import matplotlib
from tkinter import ttk
import xml.etree.ElementTree as ET
import argparse

matplotlib.use("TkAgg")


parse = argparse.ArgumentParser(description="DetVisGUI")
parse.add_argument('-f', '--format', default='VOC', help='VOC or COCO dataset format')
parse.add_argument('--detBoxColor', default=(255, 255, 0), help='detection box color')
parse.add_argument('--gtBoxColor', default=(255, 255, 255), help='groundtruth box color')
parse.add_argument('--has_anno', default=True,
                   help='There are bounding box annotations in json file / Annotataions folder.')
parse.add_argument('--output', default='output', help='image save folder')


args = parse.parse_args()


class dataInfoCOCO:
    def __init__(self):
        self.dataset = 'COCO'
        # self.img_root = r'C:\Users\Song\Desktop\Hung\vistool\coco_small\val2017'
        # self.anno_root = r'C:\Users\Song\Desktop\Hung\vistool\coco_small\instances_val2017.json'
        # self.det_file = r'C:\Users\Song\Desktop\Hung\vistool\coco_small\detections.pkl'

        # self.img_root = 'data/COCO/val2017_small'
        # self.anno_root = 'data/COCO/instances_val2017_small.json'
        # self.det_file = 'data/COCO/coco_val_results.pkl'
        # self.has_anno = True

        self.img_root = 'data/COCO/test2017_small'
        self.anno_root = 'data/COCO/image_info_test-dev2017_small.json'
        self.det_file = 'data/COCO/coco_test_results.pkl'

        self.has_anno = args.has_anno

        def jsonParser2(train_anno, has_anno):
            with open(train_anno) as f:
                data = json.load(f)

            print(list(data.keys()))

            info = data['info']
            licenses = data['licenses']
            print("info : ", info)
            print("licenses : ", licenses)

            # categories[0] : {'name': 'person', 'id': 1, 'supercategory': 'person'}
            category = [c['name'] for c in data['categories']]  # 80 classes
            # ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
            #  'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
            #  'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
            #  'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
            #  'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
            #  'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
            #  'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
            #  'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
            #  'scissors', 'teddy bear', 'hair drier', 'toothbrush']

            if has_anno:
                annotations = data['annotations']
            images = data['images']

            categoryDict = {c['id']: c['name'] for c in data['categories']}  # 80 classes
            maxCategoryId = max(categoryDict.keys())
            """
            # annotations[0]
            {'iscrowd': 0, 'area': 702.1057499999998, 'bbox': [473.07, 395.93, 38.65, 28.67], 
             'segmentation': [[510.66, 423.01, 511.72, 420.03, ... ..., 510.03, 423.01, 510.45, 423.01]],
             'image_id': 289343, 'id': 1768, 'category_id': 18}

            # images[0] 
            {'license': 4, 'id': 397133, 'date_captured': '2013-11-14 17:02:52', 
             'height': 427, 'flickr_url': 'http://farm7.staticflickr.com/6116/6255196340_da26cf2c9e_z.jpg', 
             'coco_url': 'http://images.cocodataset.org/val2017/000000397133.jpg', 
             'file_name': '000000397133.jpg', 'width': 640}
            """

            # id to image mapping
            imageDict = {}
            img_list = list()

            for image in images:
                key = image['id']
                imageDict[key] = [image['file_name'], image['width'], image['height']]
                img_list.append(image['file_name'])

            category_count = [0 for _ in range(maxCategoryId)]


            total_annotations = {}

            if has_anno:
                for a in annotations:
                    image_id = a["image_id"]

                    image_name = imageDict[a["image_id"]][0].replace('.jpg', '')
                    width = imageDict[a["image_id"]][1]
                    height = imageDict[a["image_id"]][2]
                    idx = a['category_id']
                    single_ann = []
                    single_ann.append(categoryDict[idx])
                    single_ann.extend(list(map(int, a['bbox'])))
                    single_ann.extend([width, height])

                    if image_name not in total_annotations:
                        total_annotations[image_name] = []
                    # print(len(category_count), idx)
                    category_count[idx - 1] += 1
                    total_annotations[image_name].append(single_ann)

                count = 0
                a = list(total_annotations.keys())
                a = list(map(int, a))
                print(a)
                for image in images:
                    if image['id'] not in a:
                        count += 1
                        print(str(image['id']), image['file_name'])
                print(count)

                print('\n==============[ {} json info ]=============='.format(self.dataset))
                print("Total Annotations: {}".format(len(annotations)))
                print("Total Image: {} / {}".format(len(total_annotations), len(images)))
                print("Total Category: {}".format(len(category)))
                print("{:^20}| count".format("class"))
                print('----------------------------')
                for c, cnt in zip(category, category_count):
                    if cnt != 0:
                        print("{:^20}| {}".format(c, cnt))

                print('')
            return category, img_list, total_annotations

        # according json to get category, image list, and annotations.
        self.category, self.img_list, self.total_annotations = jsonParser2(self.anno_root, self.has_anno)
        self.category = Category(self.category)

        self.results = self.get_det_results() if self.det_file != '' else None

        if self.det_file != '':
            self.img_det = {self.img_list[i]: self.results[:, i] for i in range(len(self.img_list))}

    def get_det_results(self):
        det_file = self.det_file
        if det_file != None:
            print(det_file)
            f = open(det_file, 'rb')

            det_results = np.asarray(pickle.load(f))  # [cls(bg + cls), images]
            f.close()

            print("saved det results : ", det_results.shape)

            # must be (class, image)
            # mmdetection : (image, class)
            det_results = np.transpose(det_results, (1, 0))  # (20, 483)

            print("output det results : ", det_results.shape)
            return det_results

        else:
            return None

    def get_img_by_name(self, name):
        im = Image.open(os.path.join(self.img_root, name)).convert('RGB')
        return im

    def get_img_by_index(self, idx):
        im = Image.open(os.path.join(self.img_root, self.img_list[idx])).convert('RGB')
        return im

    def get_singleImg_gt(self, name):
        return self.total_annotations[name.replace('.jpg', '')]

    def get_singleImg_dets(self, name):
        return self.img_det[name]

# pascal voc dataset
class dataInfoVOC:
    def __init__(self):
        self.dataset = 'PASCAL VOC'
        self.img_root = 'data/VOCdevkit/VOC2007/JPEGImages'
        self.anno_root = 'data/VOCdevkit/VOC2007/Annotations'
        # self.data_root = r'C:\Users\Song\Desktop\Hung\vistool\VOC2007'
        # self.det_file = r'C:\Users\Song\Desktop\Hung\vistool\det_files\pascal_voc\detections.pkl'

        self.det_file = 'data/VOCdevkit/voc_train_results.pkl'
        # self.det_file = 'data/VOCdevkit/voc_test_results.pkl'
        # self.det_file = 'data/VOCdevkit/detections_small.pkl'
        self.txt = 'data/VOCdevkit/VOC2007/ImageSets/Main/train.txt'

        self.data_root = self.anno_root.replace('/Annotations', '')
        self.has_anno = args.has_anno

        # according txt to get image list
        self.img_list = self.get_img_list()

        self.results = self.get_det_results() if self.det_file != '' else None
        self.category = Category(['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
                                  'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
                                  'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train',
                                  'tvmonitor'])

        if self.det_file != '':
            self.img_det = {self.img_list[i]: self.results[:, i] for i in range(len(self.img_list))}

    def get_img_list(self):

        with open(self.txt, 'r') as f:
            data = f.readlines()

        return [x.strip() + ".jpg" for x in data]


    def get_det_results(self):
        det_file = self.det_file
        if det_file != None:

            f = open(det_file, 'rb')

            det_results = np.asarray(pickle.load(f))  # [cls(bg + cls), images]
            f.close()

            print("saved det results : ", det_results.shape)

            # mmdetection : (image, class)
            det_results = np.transpose(det_results, (1, 0))  # (20, 483)

            print("output det results : ", det_results.shape)
            return det_results

        else:
            return None

    def get_img_by_name(self, name):
        im = Image.open(os.path.join(self.data_root, 'JPEGImages', name)).convert('RGB')
        return im

    def get_img_by_index(self, idx):
        im = Image.open(os.path.join(self.data_root, 'JPEGImages', self.img_list[idx])).convert('RGB')
        return im

    def get_singleImg_gt(self, name):
        # get annotations by image name

        # print(name)  # objs example : [['Carcharhiniformes', 709, 317, 119, 76] , ['Carcharhiniformes', 500, 370, 132, 23]]

        xml_path = os.path.join(self.data_root, 'Annotations', name.replace('.jpg','') + ".xml")
        tree = ET.parse(xml_path)
        root = tree.getroot()
        size = root.find('size')

        # img_infos.append(dict(id=img_id, filename=filename, width=width, height=height))
        img_anns = []
        for obj in root.findall('object'):
            single_ann = []
            name = obj.find('name').text
            bndbox = obj.find('bndbox')
            bbox = [
                int(bndbox.find('xmin').text),
                int(bndbox.find('ymin').text),
                int(bndbox.find('xmax').text) - int(bndbox.find('xmin').text),
                int(bndbox.find('ymax').text) - int(bndbox.find('ymin').text)
            ]
            single_ann.append(name)
            single_ann.extend(bbox)
            # print(name, bbox)
            img_anns.append(single_ann)
        # print(img_anns)
        return img_anns

    def get_singleImg_dets(self, name):
        return self.img_det[name]


# main GUI
class vis_tool:
    def __init__(self):
        self.window = Tk()
        self.menubar = Menu(self.window)

        self.info = StringVar()
        self.info_label = Label(self.window, bg='yellow', width=4, textvariable=self.info)

        self.listBox1 = Listbox(self.window, width=50, height=30, font=('Times New Roman', 10))
        self.listBox3 = Listbox(self.window, width=50, font=('Times New Roman', 10))

        self.scrollbar = Scrollbar(self.window, width=15, orient="vertical")
        self.scrollbar3 = Scrollbar(self.window, width=15, orient="vertical")

        self.listBox1_info = StringVar()
        self.listBox1_label = Label(self.window, font=('Arial', 11), bg='yellow', width=4, height=1, textvariable=self.listBox1_info)

        self.listBox3_info = StringVar()
        self.listBox3_label = Label(self.window, font=('Arial', 11), bg='yellow', width=4, height=1, textvariable=self.listBox3_info)

        if args.format == 'COCO':
            self.dataInfo = dataInfoCOCO()
        elif args.format == 'VOC':
            self.dataInfo = dataInfoVOC()
        self.info.set("DATASET: {}".format(self.dataInfo.dataset))

        # load image and show it on the window
        self.img = self.dataInfo.get_img_by_index(0)
        self.photo = ImageTk.PhotoImage(self.img)
        self.label_img = Label(self.window, image=self.photo)

        self.ShowCategory = IntVar(value=1)
        self.checkbutton_cat = Checkbutton(self.window, text='LabelText', font=('Arial', 10, 'bold'), variable=self.ShowCategory, command=self.change_img)

        self.ShowDets = IntVar(value=1)
        self.checkbutton_det = Checkbutton(self.window, text='Detections', font=('Arial', 10, 'bold'), variable=self.ShowDets, command=self.change_checkbutton_det, fg='#0000FF')

        self.ShowGTs = IntVar(value=1)
        self.checkbutton_gt = Checkbutton(self.window, text='Groundtruth', font=('Arial', 10, 'bold'), variable=self.ShowGTs, command=self.change_img, fg='#FF8C00')

        self.combo_label = Label(self.window, bg='yellow', width=10, height=1, text='Show Category', font=('Arial', 11))
        self.comboCategory = ttk.Combobox(self.window, font=('Arial', 11), values=self.dataInfo.category.comboList)
        self.comboCategory.current(0)

        self.th_label = Label(self.window, font=('Arial', 11), bg='yellow', width=10, height=1, text="Threshold")
        self.threshold = 0.5
        self.th_entry = Entry(self.window, font=('Arial', 11), textvariable=StringVar(self.window, value=str(self.threshold)), width=10)
        self.th_button = Button(self.window, text='Enter', height=1, command=self.changeThreshold)

        self.find_label = Label(self.window, font=('Arial', 11), bg='yellow', width=10, height=1, text="find")
        self.find_name = ""
        self.find_entry = Entry(self.window, font=('Arial', 11), textvariable=StringVar(self.window, value=str(self.find_name)), width=10)
        self.find_button = Button(self.window, text='Enter', height=1, command=self.findname)

        self.listBox1_idx = 0  # image listBox

        # ====== ohter attribute ======
        self.img_name = ''
        self.show_img = None
        self.output = args.output

        if not os.path.isdir(self.output):
            os.makedirs(self.output)
            
        self.img_list = self.dataInfo.img_list


    def change_checkbutton_det(self):
        self.change_img()


    def changeThreshold(self, event=None):
        self.threshold = float(self.th_entry.get())
        self.change_img()       # change image after changing threshold
        self.listBox1.focus()   # focus on listBox1 for easy control


    # draw groundtruth
    def draw_gt_boxes(self, img, objs):

        for obj in objs:  # objs example : [['Carcharhiniformes', 709, 317, 119, 76] , ['Carcharhiniformes', 500, 370, 132, 23]]

            cls_name = obj[0]

            # according combobox to decide whether to plot this category
            show_category = self.dataInfo.category.category if self.comboCategory.get() == 'All' else [
                self.comboCategory.get()]
            if cls_name not in show_category:
                continue

            box = obj[1:]
            xmin = max(box[0], 0)
            ymin = max(box[1], 0)
            xmax = min(box[0] + box[2], self.img_width)
            ymax = min(box[1] + box[3], self.img_height)

            font = cv2.FONT_HERSHEY_SIMPLEX

            if self.ShowCategory.get():
                if ymax + 30 >= self.img_height:
                    cv2.rectangle(img, (xmin, ymin), (xmin + len(cls_name) * 10, int(ymin - 20)), (255,140,0), cv2.FILLED)
                    cv2.putText(img, cls_name, (xmin, int(ymin - 5)), font, 0.5, (255, 255, 255), 1)
                else:
                    cv2.rectangle(img, (xmin, ymax), (xmin + len(cls_name) * 10, int(ymax + 20)), (255,140,0), cv2.FILLED)
                    cv2.putText(img, cls_name, (xmin, int(ymax + 15)), font, 0.5, (255, 255, 255), 1)

            cv2.rectangle(img, (xmin, ymin), (xmax, ymax), args.gtBoxColor, 1)

        return img


    def draw_det_boxes(self, img, single_detection):

        for idx, cls_objs in enumerate(single_detection):

            category = self.dataInfo.category.category[idx]

            # according combobox to decide whether to plot this category
            show_category = self.dataInfo.category.category if self.comboCategory.get() == 'All' else [
                self.comboCategory.get()]
            if category not in show_category:
                continue

            for obj in cls_objs:  # objs example : [496.2, 334.8, 668.4, 425.1, 0.99] -> [xmin, ymin, xmax, ymax, confidence]

                [score, box] = [round(obj[4], 2), obj[:4]]

                if score >= self.threshold:
                    box = list(map(int, list(map(round, box))))
                    xmin = max(box[0], 0)
                    ymin = max(box[1], 0)
                    xmax = min(box[2], self.img_width)
                    ymax = min(box[3], self.img_height)

                    if self.ShowCategory.get():
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        text = category + " : " + str(score)

                        if ymax + 30 >= self.img_height:
                            cv2.rectangle(img, (xmin, ymin), (xmin + len(text) * 9, int(ymin - 20)), (0, 0, 255), cv2.FILLED)
                            cv2.putText(img, text, (xmin, int(ymin - 5)), font, 0.5, (255, 255, 255), 1)
                        else:
                            cv2.rectangle(img, (xmin, ymax), (xmin + len(text) * 9, int(ymax + 20)), (0, 0, 255), cv2.FILLED)
                            cv2.putText(img, text, (xmin, int(ymax + 15)), font, 0.5, (255, 255, 255), 1)

                    cv2.rectangle(img, (xmin, ymin), (xmax, ymax), args.detBoxColor, 2)

        return img


    def change_img(self, event=None):
        if len(self.listBox1.curselection()) != 0:
            self.listBox1_idx = self.listBox1.curselection()[0]

        # Display image Number
        self.listBox1_info.set("Image  {:6}  / {:6}".format(self.listBox1_idx + 1, self.listBox1.size()))


        name = self.listBox1.get(self.listBox1_idx)

        img = self.dataInfo.get_img_by_name(name)
        self.img_width, self.img_height = img.width, img.height

        img = np.asarray(img)

        self.img_name = name
        self.img = img

        if self.dataInfo.has_anno and self.ShowGTs.get():
            objs = self.dataInfo.get_singleImg_gt(name)
            img = self.draw_gt_boxes(img, objs)

        if self.dataInfo.results != None and self.ShowDets.get():
            dets = self.dataInfo.get_singleImg_dets(name)
            img = self.draw_det_boxes(img, dets)
            self.clear_add_listBox3()

        self.show_img = img

        img = Image.fromarray(img)

        img = self.scale_img(img)
        self.photo = ImageTk.PhotoImage(img)
        self.label_img.config(image=self.photo)
        self.window.update_idletasks()

        if self.img_name in os.listdir(self.output):
            self.listBox1_label.config(bg='#CCFF99')
        else:
            self.listBox1_label.config(bg='yellow')


    def scale_img(self, img):
        [s_w, s_h] = [1, 1]

        # if window size is (1920, 1080), the default max image size is (1440, 810)
        (fix_width, fix_height) = (1440, 810)

        # change image size according to window size
        if self.window.winfo_width() != 1:
            fix_width = (self.window.winfo_width() - self.listBox1.winfo_width() - self.scrollbar.winfo_width() - 5)
            fix_height = int(fix_width * 9 / 16)

        # handle image size is too big
        if img.width > fix_width:
            s_w = fix_width / img.width
        if img.height > fix_height:
            s_h = fix_height / img.height

        scale = min(s_w, s_h)
        img = img.resize((int(img.width * scale), int(img.height * scale)), Image.ANTIALIAS)
        return img

    def clear_add_listBox3(self):  # object listBox
        self.listBox3.delete(0, 'end')

        single_detection = self.dataInfo.get_singleImg_dets(self.img_list[self.listBox1_idx])

        num = 0

        for idx, cls_objs in enumerate(single_detection):

            category = self.dataInfo.category.category[idx]
            for obj in cls_objs:  # objs example : [496.2, 334.8, 668.4, 425.1, 0.99] -> [xmin, ymin, xmax, ymax, confidence]
                score = round(obj[4], 2)
                if score >= self.threshold:
                    self.listBox3.insert('end', category + " : " + str(score))
                    num += 1

        # Display Object Number
        self.listBox3_info.set("Detected Object : {:3}".format(num))

    def changeThresholdButton(self, v):
        self.threshold += v

        if self.threshold <= 0:
            self.threshold = 0
        elif self.threshold >= 1:
            self.threshold = 1

        self.th_entry.delete(0, END)
        self.th_entry.insert(0, str(round(self.threshold, 2)))
        self.changeThreshold()

    def save_img(self):
        print('save_img' + self.img_name)
        cv2.imwrite(os.path.join(self.output, self.img_name), cv2.cvtColor(self.show_img, cv2.COLOR_BGR2RGB))
        self.listBox1_label.config(bg='#CCFF99')


    def eventhandler(self, event):
        if event.keysym == 'Left':
            if self.window.focus_get() == self.listBox1:
                self.change_img()

        if event.keysym == 'Right':
            self.changeThresholdButton(0.1)
        if event.keysym == 'Left':
            self.changeThresholdButton(-0.1)
        elif event.keysym == 'Escape':
            self.window.quit()
        elif event.keysym == 's':
            if self.window.focus_get() == self.listBox1:
                self.save_img()


    def comboboxChange(self, event=None):
        self.listBox1.focus()
        self.change_img()


    def clear_add_listBox1(self):

        self.listBox1.delete(0, 'end')  # delete listBox1 0 ~ end items

        # add image name to listBox1
        for item in self.img_list:
            self.listBox1.insert('end', item)

        self.listBox1.select_set(0)
        self.listBox1.focus()
        self.change_img()


    def findname(self, event=None):
        self.find_name = self.find_entry.get()
        new_list = []

        if self.find_name == '':
            new_list = self.dataInfo.img_list
        else:
            for im_name in self.dataInfo.img_list:
                if self.find_name in im_name:
                    new_list.append(im_name)

        self.img_list = new_list
        self.clear_add_listBox1()
        self.clear_add_listBox3()


    def run(self):
        self.window.title("DATASET : " + self.dataInfo.dataset)
        self.window.geometry('1280x800+350+100')

        self.menubar.add_command(label='QUIT', command=self.window.quit)
        self.window.config(menu=self.menubar)                               # display the menu
        self.scrollbar.config(command=self.listBox1.yview)
        self.listBox1.config(yscrollcommand=self.scrollbar.set)
        self.scrollbar3.config(command=self.listBox3.yview)
        self.listBox3.config(yscrollcommand=self.scrollbar3.set)

        layer1 = 0
        layer2 = 50

        # ======================= layer 1 =========================

        # combobox
        self.combo_label.grid(row=layer1 + 30, column=0, sticky=W + E + N + S, padx=3, pady=3, columnspan=6)
        self.comboCategory.grid(row=layer1 + 30, column=6, sticky=W + E + N + S, padx=3, pady=3, columnspan=6)

        # show label
        self.checkbutton_det.grid(row=layer1 + 40, column=0, sticky=N + S, padx=3, pady=3, columnspan=4)
        # show gt
        self.checkbutton_gt.grid(row=layer1 + 40, column=4, sticky=N + S, padx=3, pady=3, columnspan=4)
        # show det
        self.checkbutton_cat.grid(row=layer1 + 40, column=8, sticky=N + S, padx=3, pady=3, columnspan=4)

        # ======================= layer 2 =========================

        self.listBox1_label.grid(row=layer2 + 0, column=0, sticky=N + S + E + W, columnspan=12)

        # find name
        self.find_label.grid(row=layer2 + 20, column=0, sticky=E + W, columnspan=4)
        self.find_entry.grid(row=layer2 + 20, column=4, sticky=E + W, columnspan=4)
        self.find_button.grid(row=layer2 + 20, column=8, sticky=E + W, pady=3, columnspan=4)

        self.scrollbar.grid(row=layer2 + 30, column=11, sticky=N + S + W)
        self.label_img.grid(row=layer1 + 30, column=12, sticky=N + E, padx=3, pady=3, rowspan=110)
        self.listBox1.grid(row=layer2 + 30, column=0, sticky=N + S + E + W, pady=3, columnspan=11)

        if self.dataInfo.det_file != '' != False:
            self.th_label.grid(row=layer2 + 40, column=0, sticky=E + W, columnspan=4)
            self.th_entry.grid(row=layer2 + 40, column=4, sticky=E + W, columnspan=4)
            self.th_button.grid(row=layer2 + 40, column=8, sticky=E + W, pady=3, columnspan=4)

            self.listBox3_label.grid(row=layer2 + 50, column=0, sticky=E + W, pady=3, columnspan=12)

            self.scrollbar3.grid(row=layer2 + 60, column=11, sticky=N + S + W, pady=3)
            self.listBox3.grid(row=layer2 + 60, column=0, sticky=N + S + E + W, pady=3, columnspan=11)

        self.clear_add_listBox1()
        self.listBox1.bind('<<ListboxSelect>>', self.change_img)
        self.listBox1.bind_all('<KeyRelease>', self.eventhandler)

        self.th_entry.bind('<Return>', self.changeThreshold)
        self.th_entry.bind('<KP_Enter>', self.changeThreshold)
        self.find_entry.bind('<Return>', self.findname)
        self.find_entry.bind('<KP_Enter>', self.findname)

        self.comboCategory.bind("<<ComboboxSelected>>", self.comboboxChange)

        self.window.mainloop()


class Category:
    def __init__(self, categories):
        self.category = categories
        self.comboList = categories.copy()
        self.comboList.insert(0, 'All')
        self.all = True

if __name__ == "__main__":
    vis_tool().run()
