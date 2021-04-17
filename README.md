# DetVisGUI

## UPDATE

2021/04/17 : Add save all images function.

2021/01/05 : Support json format for det_file.

2020/12/08 : Support inference model and directly show the results on GUI.

---

## Introuction

This is a lightweight GUI for visualizing the [mmdetection](https://github.com/open-mmlab/mmdetection) results. It could display detection results with **different threshold dynamically**, and would be convenient for verifying detection results and groundtruth. 

[![alt tag](./demo/demo.png)](https://www.youtube.com/watch?v=4imQyECTik0)


Video with text description : https://www.youtube.com/watch?v=4imQyECTik0 (**The command in the video is for the master branch. Please reference the following example.**)

## Dependencies
-- mmdetection
-- tqdm


## Code

Clone this repository.

```
git clone -b mmdetection https://github.com/Chien-Hung/DetVisGUI.git
cd DetVisGUI
```

## Demo

I sample a small part of COCO and VOC2007 dataset, running mmdetection for getting detection result(\*.pkl) and use these files for demo.

```
python DetVisGUI.py ${CONFIG_FILE} [--det_file ${RESULT_FILE}] [--stage ${STAGE}] [--output ${SAVE_DIRECTORY}]
```

Arguments:

- `CONFIG_FILE`: Config file of mmdetction.

Optional Arguments:

- `RESULT_FILE`: Filename of the output results in pickle / json format.
- `STAGE`: The stage [train / val / test] of the result file, default is 'val'.
- `SAVE_DIRECTORY`: The directory for saving display images, default is 'output'.


**Display the validation results of COCO segmentation:** 

```
$ python DetVisGUI.py configs/mask_rcnn_r50_fpn_1x.py --det_file results/mask_rcnn_r50_fpn_1x/val_results.pkl
```

**Display the test results of COCO segmentation(no groundtruth):**

```
$ python DetVisGUI.py configs/mask_rcnn_r50_fpn_1x.py --det_file results/mask_rcnn_r50_fpn_1x/test_results.pkl --stage test
```

**Display the validation results of COCO detection:** 

```
$ python DetVisGUI.py configs/cascade_rcnn_r50_fpn_1x.py --det_file results/cascade_rcnn_r50_c4_1x/val_results.pkl
```

**Display the test results of COCO detection(no groundtruth):**

```
$ python DetVisGUI.py configs/cascade_rcnn_r50_fpn_1x.py --det_file results/cascade_rcnn_r50_c4_1x/test_results.pkl --stage test
```

**Display the test results of Pascal VOC(no groundtruth):**

```
$ python DetVisGUI.py configs/ssd512_voc.py --det_file results/ssd512_voc/test_results.pkl --stage test
```

**Display the validation results of COCO segmentation by json output file:** 

```
$ python DetVisGUI.py configs/mask_rcnn_r50_fpn_1x.py --det_file results/mask_rcnn_r50_fpn_1x/val_results.segm.json
```

**Display the validation results of COCO detection by json output file:** 

```
$ python DetVisGUI.py configs/mask_rcnn_r50_fpn_1x.py --det_file results/mask_rcnn_r50_fpn_1x/val_results.bbox.json
```


**Display the COCO bounding box groundtruth:** 

```
$ python DetVisGUI.py configs/mask_rcnn_r50_fpn_1x.py
```

---

## Directly Inference model on GUI

If you want to inference model and directly show the results on GUI, please run the following command. For running the example, you need to download [faster_rcnn_r50_fpn_1x_coco](https://github.com/open-mmlab/mmdetection/tree/master/configs/faster_rcnn) / [mask_rcnn_r50_fpn_1x_coco](https://github.com/open-mmlab/mmdetection/tree/master/configs/mask_rcnn) checkpoints from mmdetection (the configs is prepared in this repo), and place checkpoints in the DetVisGUI folder.

```
python DetVisGUI_test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} ${TEST_IMAGES_FOLDER} [--device ${DEVICE}]
```

Arguments:

- `CONFIG_FILE`: Config file of mmdetction.
- `CHECKPOINT_FILE`: Trained model checkpoint.
- `TEST_IMAGES_FOLDER`: Test images folder path.

Optional Arguments:

- `DEVICE`: cpu or cuda, default is cuda.

**Display the faster rcnn results:**

```
$ python DetVisGUI_test.py configs/faster_rcnn_r50_fpn_1x_coco.py ./faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth data/test_images
```

**Display the mask rcnn results:** 

```
$ python DetVisGUI_test.py configs/mask_rcnn_r50_fpn_1x_coco.py ./mask_rcnn_r50_fpn_1x_coco_20200205-d4b0c5d6.pth data/test_images
```


---

## Hotkeys

|     KEY    | ACTION                                    |
|:----------:|-------------------------------------------|
|   ↑ , ↓    | change image.                              |
|   ← , →    | change score threshold.                    | 
| ctrl +  ← , →    | change IoU threshold.                    | 
|     s     | save displayed image in output folder.     |
|     q     | colse this GUI.                            |


## result(.pkl) format

![alt tag](./demo/result_format.png)


