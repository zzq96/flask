import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import flask
# Some basic setup:
# Setup detectron2 logger
import detectron2
import torch
from detectron2.utils.logger import setup_logger

# import some common libraries
import time
import numpy as np
from PIL import Image
import json, cv2, random
import matplotlib.pyplot as plt
# from google.colab.patches import cv2_imshow

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
import io
app = flask.Flask(__name__)

model = None
from detectron2.structures import BoxMode
import matplotlib; 
# matplotlib.use('TkAgg')

def get_balloon_dicts(img_dir):
    dataset_dicts = []
    img_idx = -1
    for img_file in os.listdir(img_dir):
        if '.png' not in img_file:
            continue
        img_idx += 1
        img_file = os.path.join(img_dir, img_file)
        json_file = img_file.replace('png', 'json')
        with open(json_file) as f:
            img_ann = json.load(f)
        height, width = cv2.imread(img_file).shape[:2]
        record = {}

        record["file_name"] = img_file
        record["image_id"] = img_idx
        record["height"] = height
        record["width"] = width
        annos = img_ann["shapes"]

        objs = []
        for anno in annos:
            # assert not anno["points"]
            anno = anno["points"]
            px = [x[0] for x in anno]
            py = [x[1] for x in anno]
            poly = [(x + 0.5, y + 0.5) for x, y in zip(px, py)]
            poly = [p for x in poly for p in x]

            obj = {
                "bbox": [np.min(px), np.min(py), np.max(px), np.max(py)],
                # XYXY指box是按左上角，右下角的坐标表示，ABS表示是绝对坐标。 REL表示相对图像的坐标（范围0-1）
                "bbox_mode": BoxMode.XYXY_ABS,
                # list[list[float]]类型,每个list[float]表示实例的一个联通区域（一个实例不一定是联通，可能被某些物体遮挡为多部分）
                "segmentation": [poly],
                "category_id": 0,
            }
            objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)
    return dataset_dicts

for d in ["train", "val"]:
    DatasetCatalog.register("BoxData_" + d, lambda d=d: get_balloon_dicts("BoxData/" + d))
    MetadataCatalog.get("BoxData_" + d).set(thing_classes=["Box"])
balloon_metadata = MetadataCatalog.get("BoxData_train")
def load_model():
    """Load the pre-trained model, you can use your model just as easily.

    """
    global model
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # faster, and good enough for this toy dataset (default: 512)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (ballon)
    # cfg.MODEL.DEVICE = "cpu"
    cfg.MODEL.WEIGHTS = os.path.join('output_box', "model_final.pth")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set a custom testing threshold for this model
    cfg.DATASETS.TEST = ("BoxData_val", )
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.3

    model = DefaultPredictor(cfg)

time_all = 0
cnt =0
@app.route("/predict", methods=["POST"])
def predict():
    global time_all
    global cnt
    cnt +=1
    # Initialize the data dictionary that will be returned from the view.
    data = {"success": False}

    # Ensure an image was properly uploaded to our endpoint.
    print("flask.request.method", flask.request.method)
    if flask.request.method == 'POST':
        # if flask.request.files.get("image"):
        if 1:
            # Read the image in PIL format
            # im = flask.request.files["image"].read()
            im = flask.request.data
            im = np.array(Image.open(io.BytesIO(im)))[:, :, :3][:,:,::-1]
            print(im.shape)
            # print(im.shape)
            # print(im)
            # Preprocess the image and prepare it for classification.

            # Classify the input image and then initialize the list of predictions to return to the client.
            time_start = time.time()
            outputs = model(im)
            time_all += time.time() - time_start
            print("sigle:",time.time() - time_start)
            print("avg",time_all/ cnt)
            v = Visualizer(im,
                        metadata=balloon_metadata, 
                        scale=0.5, 
                        #    instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels. This option is only available for segmentation models
            )
            out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
            cv2.imwrite(r"imgs/"+str(cnt)+".png", out.get_image())
            # plt.figure(figsize=(20, 20))
            # plt.imshow(out.get_image()[:, :])
            # plt.show()
    #         results = torch.topk(preds.cpu().data, k=3, dim=1)

    #         data['predictions'] = list()

    #         # Loop over the results and add them to the list of returned predictions
    #         for prob, label in zip(results[0][0], results[1][0]):
    #             label_name = idx2label[label.item()]
    #             r = {"label": label_name, "probability": float(prob)}
    #             data['predictions'].append(r)

    #         # Indicate that the request was a success.
            data["success"] = True
            # data['instances'] = outputs['instances']
            # torch.where(outputs['instances'].pred_masks > 0).numpy()
            idx = np.where(outputs['instances'].pred_masks.cpu().numpy() > 0)
            data['instances_id'] = idx[0].tolist()
            data['instances_row'] = idx[1].tolist()
            data['instances_col'] = idx[2].tolist()
            data['pred_boxes']  = outputs['instances'].get_fields()['pred_boxes'].tensor.tolist()
            data['pred_classes']  = outputs['instances'].get_fields()['pred_classes'].cpu().tolist()


    # # Return the data dictionary as a JSON response.
    return flask.jsonify(data) 

if __name__== "__main__":
    load_model()
    app.run(host="49.52.10.229", port = 5000)