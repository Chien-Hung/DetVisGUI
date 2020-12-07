# DetVisGUI

## UPDATE

Support inference model and directly show the results on GUI.


## Introuction

This is a lightweight GUI for visualizing the [mmdetection](https://github.com/open-mmlab/mmdetection) results. It could display detection results with **different threshold dynamically**, and would be convenient for verifying detection results and groundtruth. 

[![alt tag](./demo/demo.png)](https://www.youtube.com/watch?v=4imQyECTik0)


Video with text description : https://www.youtube.com/watch?v=4imQyECTik0 (**The command in the video is for the master branch. Please reference the following example.**)

## Dependencies
-- mmdetection

## Code

Clone this repository.

```
git clone -b mmdetection https://github.com/Chien-Hung/DetVisGUI.git
cd DetVisGUI
```

## Demo

I sample a small part of COCO and VOC2007 dataset, running mmdetection for getting detection result(\*.pkl) and use these files for demo.

```
python DetVisGUI.py ${CONFIG_FILE} ${RESULT_FILE} [--stage ${STAGE}] [--output ${SAVE_DIRECTORY}]
```

Arguments:

- `CONFIG_FILE`: Config file of mmdetction.
- `RESULT_FILE`: Filename of the output results in pickle format.

Optional Arguments:

- `STAGE`: The stage [train / val / test] of the result file, default is 'val'.
- `SAVE_DIRECTORY`: The directory for saving display images, default is 'output'.


**Display the validation results of COCO segmentation:** 

```
$ python DetVisGUI.py configs/mask_rcnn_r50_fpn_1x.py results/mask_rcnn_r50_fpn_1x/val_results.pkl
```

**Display the test results of COCO segmentation(no groundtruth):**

```
$ python DetVisGUI.py configs/mask_rcnn_r50_fpn_1x.py results/mask_rcnn_r50_fpn_1x/test_results.pkl --stage test
```

**Display the validation results of COCO detection:** 

```
$ python DetVisGUI.py configs/cascade_rcnn_r50_fpn_1x.py results/cascade_rcnn_r50_c4_1x/val_results.pkl
```

**Display the test results of COCO detection(no groundtruth):**

```
$ python DetVisGUI.py configs/cascade_rcnn_r50_fpn_1x.py results/cascade_rcnn_r50_c4_1x/test_results.pkl --stage test
```

**Display the test results of Pascal VOC(no groundtruth):**

```
$ python DetVisGUI.py configs/ssd512_voc.py results/ssd512_voc/test_results.pkl --stage test
```

---

## Directly Inference model on GUI

If you want to inference model and directly show the results on GUI, please run the following command. For running the example, you need to download [faster_rcnn_r50_fpn_1x_coco](https://github.com/open-mmlab/mmdetection/tree/master/configs/faster_rcnn) / [mask_rcnn_r50_fpn_1x_coco](https://github.com/open-mmlab/mmdetection/tree/master/configs/mask_rcnn) checkpoints from mmdetection github and place them in checkpoints.

```
python DetVisGUI_test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} ${TEST_IMAGES_FOLDER} [--device ${DEVICE}]
```

Arguments:

- `CONFIG_FILE`: Config file of mmdetction.
- `CHECKPOINT_FILE`: Trained model.
- `TEST_IMAGES_FOLDER`: Test images folder path.

Optional Arguments:

- `DEVICE`: cpu or cuda, Default is cuda.

**Display the faster rcnn results:**

```
$ python DetVisGUI_test.py configs/faster_rcnn_r50_fpn_1x_coco.py checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth data/test_images
```

**Display the mask rcnn results:** 

```
$ python DetVisGUI_test.py configs/mask_rcnn_r50_fpn_1x_coco.py checkpoints/mask_rcnn_r50_fpn_1x_coco_20200205-d4b0c5d6.pth data/test_images
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


