# __author__ = 'ChienHung Chen in Academia Sinica IIS'

import argparse
import itertools
import json
import os
import pickle
import xml.etree.ElementTree as ET
from tkinter import (END, Button, Checkbutton, E, Entry, IntVar, Label,
                     Listbox, Menu, N, S, Scrollbar, StringVar, Tk, W, ttk)

import cv2
import matplotlib
import mmcv
from mmdet.apis import init_detector, inference_detector
import numpy as np
import platform
import pycocotools.mask as maskUtils
from PIL import Image, ImageTk

matplotlib.use('TkAgg')


def parse_args():
    parser = argparse.ArgumentParser(description='DetVisGUI')
    parser.add_argument('config', 
        default='./config/mask_rcnn_r50_fpn_1x_coco.py', 
        help='config file path')

    parser.add_argument('ckpt', 
        default='./checkpoints/mask_rcnn_r50_fpn_1x_coco_20200205-d4b0c5d6.pth', 
        help='checkpoint file path')

    parser.add_argument('img_root', 
        default='./data/test_images', 
        help='test image path')

    parser.add_argument('--device', default='cuda', help='inference device')

    parser.add_argument(
        '--no_gt',
        default=True,
        help='test images without groundtruth')

    parser.add_argument(
        '--det_box_color', default=(255, 255, 0), help='detection box color')

    parser.add_argument(
        '--gt_box_color',
        default=(255, 255, 255),
        help='groundtruth box color')

    parser.add_argument('--output', default='output', help='image save folder')

    args = parser.parse_args()
    return args


class COCO_dataset:

    def __init__(self, cfg, args):
        self.dataset = 'COCO'
        self.img_root = args.img_root
        self.config_file = args.config
        self.checkpoint_file = args.ckpt
        self.mask = False
        self.device = args.device

        # according json to get category, image list, and annotations.
        self.img_list = self.get_img_list()

        # coco categories
        self.aug_category = aug_category([
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 
            'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 
            'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 
            'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 
            'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 
            'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 
            'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 
            'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 
            'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 
            'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 
            'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 
            'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 
            'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 
            'scissors', 'teddy bear', 'hair drier', 'toothbrush'
            ])


    def get_img_list(self):
        img_list = list()
        for image in sorted(os.listdir(self.img_root)):
            img_list.append(image)

        return img_list

    def get_img_by_name(self, name):
        img = Image.open(os.path.join(self.img_root, name)).convert('RGB')
        return img

    def get_img_by_index(self, idx):
        img = Image.open(os.path.join(self.img_root,
                                      self.img_list[idx])).convert('RGB')
        return img


# main GUI
class vis_tool:

    def __init__(self):
        self.args = parse_args()
        cfg = mmcv.Config.fromfile(self.args.config)
        self.window = Tk()
        self.menubar = Menu(self.window)

        self.info = StringVar()
        self.info_label = Label(
            self.window, bg='yellow', width=4, textvariable=self.info)

        self.listBox_img = Listbox(
            self.window, width=50, height=25, font=('Times New Roman', 10))
        self.listBox_obj = Listbox(
            self.window, width=50, height=12, font=('Times New Roman', 10))

        self.scrollbar_img = Scrollbar(
            self.window, width=15, orient='vertical')
        self.scrollbar_obj = Scrollbar(
            self.window, width=15, orient='vertical')

        self.listBox_img_info = StringVar()
        self.listBox_img_label = Label(
            self.window,
            font=('Arial', 11),
            bg='yellow',
            width=4,
            height=1,
            textvariable=self.listBox_img_info)

        self.listBox_obj_info = StringVar()
        self.listBox_obj_label1 = Label(
            self.window,
            font=('Arial', 11),
            bg='yellow',
            width=4,
            height=1,
            textvariable=self.listBox_obj_info)
        self.listBox_obj_label2 = Label(
            self.window,
            font=('Arial', 11),
            bg='yellow',
            width=4,
            height=1,
            text='Object Class : Score')

        self.data_info = COCO_dataset(cfg, self.args)
        self.info.set('DATASET: {}'.format(self.data_info.dataset))

        # load image and show it on the window
        self.img = self.data_info.get_img_by_index(0)
        self.photo = ImageTk.PhotoImage(self.img)
        self.label_img = Label(self.window, image=self.photo)

        self.show_det_txt = IntVar(value=1)
        self.checkbn_det_txt = Checkbutton(
            self.window,
            text='Text',
            font=('Arial', 10, 'bold'),
            variable=self.show_det_txt,
            command=self.change_img,
            fg='#0000FF')

        self.show_dets = IntVar(value=1)
        self.checkbn_det = Checkbutton(
            self.window,
            text='Detections',
            font=('Arial', 10, 'bold'),
            variable=self.show_dets,
            command=self.change_img,
            fg='#0000FF')

        self.combo_label = Label(
            self.window,
            bg='yellow',
            width=10,
            height=1,
            text='Show Category',
            font=('Arial', 11))
        self.combo_category = ttk.Combobox(
            self.window,
            font=('Arial', 11),
            values=self.data_info.aug_category.combo_list)
        self.combo_category.current(0)

        self.th_label = Label(
            self.window,
            font=('Arial', 11),
            bg='yellow',
            width=10,
            height=1,
            text='Score Threshold')
        self.threshold = np.float32(0.5)
        self.th_entry = Entry(
            self.window,
            font=('Arial', 11),
            width=10,
            textvariable=StringVar(self.window, value=str(self.threshold)))
        self.th_button = Button(
            self.window, text='Enter', height=1, command=self.change_threshold)

        self.find_label = Label(
            self.window,
            font=('Arial', 11),
            bg='yellow',
            width=10,
            height=1,
            text='find')
        self.find_name = ''
        self.find_entry = Entry(
            self.window,
            font=('Arial', 11),
            width=10,
            textvariable=StringVar(self.window, value=str(self.find_name)))
        self.find_button = Button(
            self.window, text='Enter', height=1, command=self.findname)

        self.listBox_img_idx = 0

        # ====== ohter attribute ======
        self.img_name = ''
        self.show_img = None
        self.output = self.args.output
        self.model = init_detector(
            self.data_info.config_file, 
            self.data_info.checkpoint_file, 
            device=self.data_info.device)


        if not os.path.isdir(self.output):
            os.makedirs(self.output)

        self.img_list = self.data_info.img_list

        # flag for find/threshold button switch focused element
        self.button_clicked = False

    def change_threshold(self, event=None):
        try:
            self.threshold = np.float32(self.th_entry.get())
            self.change_img()

            # after changing threshold, focus on listBox for easy control
            if self.window.focus_get() == self.listBox_obj:
                self.listBox_obj.focus()
            else:
                self.listBox_img.focus()

            self.button_clicked = True

        except ValueError:
            self.window.title('Please enter a number as score threshold.')

    # draw groundtruth
    def draw_gt_boxes(self, img, objs):
        for obj in objs:
            cls_name = obj[0]

            # according combobox to decide whether to plot this category
            if self.combo_category.get() == 'All':
                show_category = self.data_info.aug_category.category
            else:
                show_category = [self.combo_category.get()]

            if cls_name not in show_category:
                continue

            box = obj[1:]
            xmin = max(box[0], 0)
            ymin = max(box[1], 0)
            xmax = min(box[0] + box[2], self.img_width)
            ymax = min(box[1] + box[3], self.img_height)

            font = cv2.FONT_HERSHEY_SIMPLEX

            if self.show_gt_txt.get():
                if ymax + 30 >= self.img_height:
                    cv2.rectangle(img, (xmin, ymin),
                                  (xmin + len(cls_name) * 10, int(ymin - 20)),
                                  (255, 140, 0), cv2.FILLED)
                    cv2.putText(img, cls_name, (xmin, int(ymin - 5)), font,
                                0.5, (255, 255, 255), 1)
                else:
                    cv2.rectangle(img, (xmin, ymax),
                                  (xmin + len(cls_name) * 10, int(ymax + 20)),
                                  (255, 140, 0), cv2.FILLED)
                    cv2.putText(img, cls_name, (xmin, int(ymax + 15)), font,
                                0.5, (255, 255, 255), 1)

            cv2.rectangle(img, (xmin, ymin), (xmax, ymax),
                          self.args.gt_box_color, 1)

        return img


    def draw_all_det_boxes(self, img, single_detection):

        for idx, cls_objs in enumerate(single_detection):
            category = self.data_info.aug_category.category[idx]

            if self.combo_category.get() == 'All':
                show_category = self.data_info.aug_category.category
            else:
                show_category = [self.combo_category.get()]

            if category not in show_category:
                continue

            for obj_idx, obj in enumerate(cls_objs):
                [score, box] = [round(obj[4], 2), obj[:4]]

                if score >= self.threshold:
                    box = list(map(int, list(map(round, box))))
                    xmin = max(box[0], 0)
                    ymin = max(box[1], 0)
                    xmax = min(box[2], self.img_width)
                    ymax = min(box[3], self.img_height)

                    if self.show_det_txt.get():
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        text = category + ' : ' + str(score)

                        if ymax + 30 >= self.img_height:
                            cv2.rectangle(
                                img, (xmin, ymin),
                                (xmin + len(text) * 9, int(ymin - 20)),
                                (0, 0, 255), cv2.FILLED)
                            cv2.putText(img, text, (xmin, int(ymin - 5)), font,
                                        0.5, (255, 255, 255), 1)
                        else:
                            cv2.rectangle(
                                img, (xmin, ymax),
                                (xmin + len(text) * 9, int(ymax + 20)),
                                (0, 0, 255), cv2.FILLED)
                            cv2.putText(img, text, (xmin, int(ymax + 15)),
                                        font, 0.5, (255, 255, 255), 1)

                    cv2.rectangle(
                        img, (xmin, ymin), (xmax, ymax), 
                        self.args.det_box_color, 2)

        return img

    def draw_all_det_boxes_masks(self, img, single_detection):
        img = np.require(img, requirements=['W'])
        boxes, masks = single_detection

        # draw segmentation masks
        # reference mmdetection/mmdet/models/detectors/base.py
        if self.combo_category.get() != 'All':
            show_idx = self.data_info.aug_category.category.index(
                self.combo_category.get())
            masks = np.asarray([masks[show_idx]])
            boxes = np.asarray([boxes[show_idx]])
            category = self.data_info.aug_category.category[show_idx]

        segms = list(itertools.chain(*masks))
        bboxes = np.vstack(boxes)

        inds = np.where(np.round(bboxes[:, -1], 2) >= self.threshold)[0]
        
        self.color_list = []       
        for i in inds:
            color_mask = np.random.randint(0, 256, (1, 3), dtype=np.uint8)

            if type(segms[0]) == np.ndarray:
                mask = segms[i]
            elif type(segms[0]) == dict:
                mask = maskUtils.decode(segms[i]).astype(np.bool)

            img[mask] = img[mask] * 0.5 + color_mask * 0.5
            self.color_list.append('#%02x%02x%02x' % tuple(color_mask[0]))
     
        # draw bounding box
        for idx, cls_objs in enumerate(boxes):
            if self.combo_category.get() == 'All':
                category = self.data_info.aug_category.category[idx]

            for obj_idx, obj in enumerate(cls_objs):
                [score, box] = [round(obj[4], 2), obj[:4]]

                if score >= self.threshold:
                    box = list(map(int, list(map(round, box))))
                    xmin = max(box[0], 0)
                    ymin = max(box[1], 0)
                    xmax = min(box[2], self.img_width)
                    ymax = min(box[3], self.img_height)

                    if self.show_det_txt.get():
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        text = category + ' : ' + str(score)

                        if ymax + 30 >= self.img_height:
                            cv2.rectangle(
                                img, (xmin, ymin),
                                (xmin + len(text) * 9, int(ymin - 20)),
                                (0, 0, 255), cv2.FILLED)
                            cv2.putText(img, text, (xmin, int(ymin - 5)), font,
                                        0.5, (255, 255, 255), 1)
                        else:
                            cv2.rectangle(
                                img, (xmin, ymax),
                                (xmin + len(text) * 9, int(ymax + 20)),
                                (0, 0, 255), cv2.FILLED)
                            cv2.putText(img, text, (xmin, int(ymax + 15)),
                                        font, 0.5, (255, 255, 255), 1)

                    cv2.rectangle(
                        img, (xmin, ymin), (xmax, ymax), 
                        self.args.det_box_color, 2)

        return img

    def get_dets(self, det_results):      # [(bg + cls), images]
        det_results = np.asarray(det_results, dtype=object)
        # dim should be (class, image), mmdetection format: (image, class)

        if len(det_results.shape) == 2:
            self.data_info.mask = True
        return det_results


    def change_img(self, event=None):
        if len(self.listBox_img.curselection()) != 0:
            self.listBox_img_idx = self.listBox_img.curselection()[0]

        self.listBox_img_info.set('Image  {:6}  / {:6}'.format(
            self.listBox_img_idx + 1, self.listBox_img.size()))

        name = self.listBox_img.get(self.listBox_img_idx)
        self.window.title('DATASET : ' + self.data_info.dataset + '   ' + name)

        img = self.data_info.get_img_by_name(name)
        self.img_width, self.img_height = img.width, img.height

        img = np.asarray(img)

        self.img_name = name
        self.img = img
        
        result = inference_detector(self.model, img)
        self.dets = self.get_dets(result)

        if self.show_dets.get():
            if self.data_info.mask is False:
                img = self.draw_all_det_boxes(img, self.dets)
            else:
                img = self.draw_all_det_boxes_masks(img, self.dets)

            self.clear_add_listBox_obj()

        self.show_img = img
        img = Image.fromarray(img)
        img = self.scale_img(img)
        self.photo = ImageTk.PhotoImage(img)
        self.label_img.config(image=self.photo)
        self.window.update_idletasks()

        if self.img_name in os.listdir(self.output):
            self.listBox_img_label.config(bg='#CCFF99')
        else:
            self.listBox_img_label.config(bg='yellow')

    def draw_one_det_boxes(self, img, single_detection, selected_idx=-1):
        idx_counter = 0
        for idx, cls_objs in enumerate(single_detection):

            category = self.data_info.aug_category.category[idx]
            if self.combo_category.get() == 'All':
                show_category = self.data_info.aug_category.category
            else:
                show_category = [self.combo_category.get()]

            if category not in show_category:
                continue

            for obj_idx, obj in enumerate(cls_objs):
                [score, box] = [round(obj[4], 2), obj[:4]]

                if score >= self.threshold:
                    if idx_counter == selected_idx:
                        box = list(map(int, list(map(round, box))))
                        xmin = max(box[0], 0)
                        ymin = max(box[1], 0)
                        xmax = min(box[2], self.img_width)
                        ymax = min(box[3], self.img_height)

                        if self.show_det_txt.get():
                            font = cv2.FONT_HERSHEY_SIMPLEX
                            text = category + ' : ' + str(score)

                            if ymax + 30 >= self.img_height:
                                cv2.rectangle(
                                    img, (xmin, ymin),
                                    (xmin + len(text) * 9, int(ymin - 20)),
                                    (0, 0, 255), cv2.FILLED)
                                cv2.putText(img, text, (xmin, int(ymin - 5)),
                                            font, 0.5, (255, 255, 255), 1)
                            else:
                                cv2.rectangle(
                                    img, (xmin, ymax),
                                    (xmin + len(text) * 9, int(ymax + 20)),
                                    (0, 0, 255), cv2.FILLED)
                                cv2.putText(img, text, (xmin, int(ymax + 15)),
                                            font, 0.5, (255, 255, 255), 1)

                        cv2.rectangle(
                            img, (xmin, ymin), (xmax, ymax), 
                            self.args.det_box_color, 2)

                        return img
                    else:
                        idx_counter += 1

    def draw_one_det_boxes_masks(self, img, single_detection, selected_idx=-1):
        img = np.require(img, requirements=['W'])
        boxes, masks = single_detection

        # draw segmentation masks
        # reference mmdetection/mmdet/models/detectors/base.py
        if self.combo_category.get() != 'All':
            show_idx = self.data_info.aug_category.category.index(
                self.combo_category.get())
            category = self.data_info.aug_category.category[
                show_idx]  # fixed category
            masks = np.asarray([masks[show_idx]])
            boxes = np.asarray([boxes[show_idx]])

        segms = list(itertools.chain(*masks))
        bboxes = np.vstack(boxes)

        inds = np.where(np.round(bboxes[:, -1], 2) >= self.threshold)[0]

        self.color_list = []
        for inds_idx, i in enumerate(inds):
            if inds_idx == selected_idx:
                color_mask = np.random.randint(0, 256, (1, 3), dtype=np.uint8)

                if type(segms[0]) == np.ndarray:
                    mask = segms[i]
                elif type(segms[0]) == dict:
                    mask = maskUtils.decode(segms[i]).astype(np.bool)

                img[mask] = img[mask] * 0.5 + color_mask * 0.5
                self.color_list.append('#%02x%02x%02x' % tuple(color_mask[0]))
                
        # draw bounding box
        idx_counter = 0
        for idx, cls_objs in enumerate(boxes):
            if self.combo_category.get() == 'All':
                category = self.data_info.aug_category.category[idx]

            for obj_idx, obj in enumerate(cls_objs):
                [score, box] = [round(obj[4], 2), obj[:4]]

                if score >= self.threshold:
                    if idx_counter == selected_idx:
                        box = list(map(int, list(map(round, box))))
                        xmin = max(box[0], 0)
                        ymin = max(box[1], 0)
                        xmax = min(box[2], self.img_width)
                        ymax = min(box[3], self.img_height)

                        if self.show_det_txt.get():
                            font = cv2.FONT_HERSHEY_SIMPLEX
                            text = category + ' : ' + str(score)

                            if ymax + 30 >= self.img_height:
                                cv2.rectangle(
                                    img, (xmin, ymin),
                                    (xmin + len(text) * 9, int(ymin - 20)),
                                    (0, 0, 255), cv2.FILLED)
                                cv2.putText(img, text, (xmin, int(ymin - 5)),
                                            font, 0.5, (255, 255, 255), 1)
                            else:
                                cv2.rectangle(
                                    img, (xmin, ymax),
                                    (xmin + len(text) * 9, int(ymax + 20)),
                                    (0, 0, 255), cv2.FILLED)
                                cv2.putText(img, text, (xmin, int(ymax + 15)),
                                            font, 0.5, (255, 255, 255), 1)

                        cv2.rectangle(img, (xmin, ymin), (xmax, ymax),
                                      self.args.det_box_color, 2)

                        return img
                    else:
                        idx_counter += 1

    # plot only one object
    def change_obj(self, event=None):
        if len(self.listBox_obj.curselection()) == 0:
            self.listBox_img.focus()
            return
        else:
            listBox_obj_idx = self.listBox_obj.curselection()[0]

        self.listBox_obj_info.set('Detected Object : {:4}  / {:4}'.format(
            listBox_obj_idx + 1, self.listBox_obj.size()))

        name = self.listBox_img.get(self.listBox_img_idx)
        img = self.data_info.get_img_by_name(name)
        self.img_width, self.img_height = img.width, img.height
        img = np.asarray(img)
        self.img_name = name
        self.img = img

        if self.show_dets.get():

            if self.data_info.mask is False:
                img = self.draw_one_det_boxes(img, self.dets, listBox_obj_idx)
            else:
                img = self.draw_one_det_boxes_masks(img, self.dets, listBox_obj_idx)

        self.show_img = img
        img = Image.fromarray(img)
        img = self.scale_img(img)
        self.photo = ImageTk.PhotoImage(img)
        self.label_img.config(image=self.photo)
        self.window.update_idletasks()

        if self.img_name in os.listdir(self.output):
            self.listBox_img_label.config(bg='#CCFF99')
        else:
            self.listBox_img_label.config(bg='yellow')

    # ============================================

    def scale_img(self, img):
        [s_w, s_h] = [1, 1]

        # if window size is (1920, 1080),
        # the default max image size is (1440, 810)
        (fix_width, fix_height) = (1440, 810)

        # change image size according to window size
        if self.window.winfo_width() != 1:
            fix_width = (
                self.window.winfo_width() - self.listBox_img.winfo_width() -
                self.scrollbar_img.winfo_width() - 5)
            # fix_height = int(fix_width * 9 / 16)
            fix_height = 750

        # handle image size is too big
        if img.width > fix_width:
            s_w = fix_width / img.width
        if img.height > fix_height:
            s_h = fix_height / img.height

        scale = min(s_w, s_h)
        img = img.resize((int(img.width * scale), int(img.height * scale)),
                         Image.ANTIALIAS)
        return img

    def clear_add_listBox_obj(self):
        self.listBox_obj.delete(0, 'end')

        if self.data_info.mask is False:
            single_detection = self.dets
        else:
            single_detection, single_mask = self.dets

        if self.combo_category.get() == 'All':
            show_category = self.data_info.aug_category.category
        else:
            show_category = [self.combo_category.get()]

        num = 0
        for idx, cls_objs in enumerate(single_detection):
            category = self.data_info.aug_category.category[idx]

            if category not in show_category:
                continue

            for obj_idx, obj in enumerate(cls_objs):
                score = np.round(obj[4], 2)
                if score >= self.threshold:
                    self.listBox_obj.insert('end', category + " : " + str(score))

                    num += 1

        self.listBox_obj_info.set('Detected Object : {:3}'.format(num))

    def change_threshold_button(self, v):
        self.threshold += v

        if self.threshold <= 0:
            self.threshold = 0
        elif self.threshold >= 1:
            self.threshold = 1

        self.th_entry.delete(0, END)
        self.th_entry.insert(0, str(round(self.threshold, 2)))
        self.change_threshold()

    def save_img(self):
        print('Save image to ' + os.path.join(self.output, self.img_name))
        cv2.imwrite(
            os.path.join(self.output, self.img_name),
            cv2.cvtColor(self.show_img, cv2.COLOR_BGR2RGB))
        self.listBox_img_label.config(bg='#CCFF99')

    def eventhandler(self, event):
        entry_list = [self.find_entry, self.th_entry]
        if self.window.focus_get() not in entry_list:
            if platform.system() == 'Windows':
                state_1key = 8
                state_2key = 12
            else:  # 'Linux'
                state_1key = 16
                state_2key = 20

            if event.state == state_1key and event.keysym == 'Left':
                self.change_threshold_button(-0.1)
            elif event.state == state_1key and event.keysym == 'Right':
                self.change_threshold_button(0.1)
            elif event.keysym == 'q':
                self.window.quit()
            elif event.keysym == 's':
                self.save_img()

            if self.button_clicked:
                self.button_clicked = False
            else:
                if event.keysym in ['KP_Enter', 'Return']:
                    self.listBox_obj.focus()
                    self.listBox_obj.select_set(0)
                elif event.keysym == 'Escape':
                    self.change_img()
                    self.listBox_img.focus()

    def combobox_change(self, event=None):
        self.listBox_img.focus()
        self.change_img()

    def clear_add_listBox_img(self):
        self.listBox_img.delete(0, 'end')  # delete listBox_img 0 ~ end items

        # add image name to listBox_img
        for item in self.img_list:
            self.listBox_img.insert('end', item)

        self.listBox_img.select_set(0)
        self.listBox_img.focus()
        self.change_img()

    def findname(self, event=None):
        self.find_name = self.find_entry.get()
        new_list = []

        if self.find_name == '':
            new_list = self.data_info.img_list
        else:
            for img_name in self.data_info.img_list:
                if self.find_name[0] == '!':
                    if self.find_name[1:] not in img_name:
                        new_list.append(img_name)
                else:
                    if self.find_name in img_name:
                        new_list.append(img_name)

        if len(new_list) != 0:
            self.img_list = new_list
            self.clear_add_listBox_img()
            self.clear_add_listBox_obj()
            self.button_clicked = True
        else:
            self.window.title("Can't find any image about '{}'".format(
                self.find_name))

    def run(self):
        self.window.title('DATASET : ' + self.data_info.dataset)
        self.window.geometry('1280x800+350+100')

        # self.menubar.add_command(label='QUIT', command=self.window.quit)
        self.window.config(menu=self.menubar)  # display the menu
        self.scrollbar_img.config(command=self.listBox_img.yview)
        self.listBox_img.config(yscrollcommand=self.scrollbar_img.set)
        self.scrollbar_obj.config(command=self.listBox_obj.yview)
        self.listBox_obj.config(yscrollcommand=self.scrollbar_obj.set)

        layer1 = 0
        layer2 = 50

        # ======================= layer 1 =========================

        # combobox
        self.combo_label.grid(
            row=layer1 + 30,
            column=0,
            sticky=W + E + N + S,
            padx=3,
            pady=3,
            columnspan=6)
        self.combo_category.grid(
            row=layer1 + 30,
            column=6,
            sticky=W + E + N + S,
            padx=3,
            pady=3,
            columnspan=6)

        # show det
        self.checkbn_det.grid(
            row=layer1 + 40,
            column=0,
            sticky=N + S,
            padx=3,
            pady=3,
            columnspan=4)
        # show det text
        self.checkbn_det_txt.grid(
            row=layer1 + 40,
            column=4,
            sticky=N + S,
            padx=3,
            pady=3,
            columnspan=2)

        # ======================= layer 2 =========================

        self.listBox_img_label.grid(
            row=layer2 + 0, column=0, sticky=N + S + E + W, columnspan=12)

        # find name
        self.find_label.grid(
            row=layer2 + 20, column=0, sticky=E + W, columnspan=4)
        self.find_entry.grid(
            row=layer2 + 20, column=4, sticky=E + W, columnspan=4)
        self.find_button.grid(
            row=layer2 + 20, column=8, sticky=E + W, pady=3, columnspan=4)

        self.scrollbar_img.grid(row=layer2 + 30, column=11, sticky=N + S + W)
        self.label_img.grid(
            row=layer1 + 30,
            column=12,
            sticky=N + E,
            padx=3,
            pady=3,
            rowspan=110)
        self.listBox_img.grid(
            row=layer2 + 30,
            column=0,
            sticky=N + S + E + W,
            pady=3,
            columnspan=11)

        
        self.th_label.grid(
            row=layer2 + 40, column=0, sticky=E + W, columnspan=6)
        self.th_entry.grid(
            row=layer2 + 40, column=6, sticky=E + W, columnspan=3)
        self.th_button.grid(
            row=layer2 + 40, column=9, sticky=E + W, columnspan=3)

        self.listBox_obj_label1.grid(
            row=layer2 + 60, column=0, sticky=E + W, pady=3, columnspan=12)

        self.listBox_obj_label2.grid(
            row=layer2 + 70, 
            column=0, 
            sticky=E + W, 
            pady=2, 
            columnspan=12)

        self.scrollbar_obj.grid(
            row=layer2 + 80, column=11, sticky=N + S + W, pady=3)
        self.listBox_obj.grid(
            row=layer2 + 80,
            column=0,
            sticky=N + S + E + W,
            pady=3,
            columnspan=11)

        self.clear_add_listBox_img()
        self.listBox_img.bind('<<ListboxSelect>>', self.change_img)
        self.listBox_img.bind_all('<KeyRelease>', self.eventhandler)

        self.listBox_obj.bind('<<ListboxSelect>>', self.change_obj)

        self.th_entry.bind('<Return>', self.change_threshold)
        self.th_entry.bind('<KP_Enter>', self.change_threshold)
        self.find_entry.bind('<Return>', self.findname)
        self.find_entry.bind('<KP_Enter>', self.findname)

        self.combo_category.bind('<<ComboboxSelected>>', self.combobox_change)

        self.window.mainloop()


class aug_category:

    def __init__(self, categories):
        self.category = categories
        self.combo_list = categories.copy()
        self.combo_list.insert(0, 'All')
        self.all = True


if __name__ == '__main__':
    vis_tool().run()


