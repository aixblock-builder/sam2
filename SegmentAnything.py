from argparse import ArgumentParser
import re
import time
import datetime
import io
import json
import os
import logging
import torch
import cv2
import numpy as np
import hashlib
import urllib
import requests

# import boto3
# from botocore.exceptions import ClientError
# from urllib.parse import urlparse
# from wow_ai_ml.model import LabelStudioMLBase
# from wow_ai_ml.utils import get_single_tag_keys, get_image_size, is_skipped, DATA_UNDEFINED_NAME, os.getenv
# from label_studio_tools.core.utils.io import get_data_dir
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import os
# from pycocotools import mask as cocomask
# from pycocotools import coco as cocoapi
# from flask import render_template
# import redis

# import lightning as L
# import segmentation_models_pytorch as smp
import torch
# import torch.nn.functional as F
# from box import Box
from config import cfg
# from dataset import load_datasets
# from lightning.fabric.fabric import _FabricOptimizer
# from lightning.fabric.loggers import TensorBoardLogger
# from losses import DiceLoss
# from losses import FocalLoss
# from model import Model
# from torch.utils.data import DataLoader
# from utils import AverageMeter
# from utils import calc_iou


torch.set_float32_matmul_precision('high')

REDIS_HOST = os.getenv('REDIS_HOST', 'segment_anything_annotation_redis')
REDIS_PORT = os.getenv('REDIS_PORT', 6380)
REDIS_DB = os.getenv('REDIS_DB', 5)
# https://github.com/pytorch/pytorch/issues/40403


AWS_ACCESS_KEY = os.getenv('AWS_ACCESS_KEY_ID', '476d381a1d4513b8b0dbbc2d7064f96c')
AWS_SECRET_KEY = os.getenv('AWS_ACCESS_SECRET_KEY', '126b46018eb0b7b12eeb70b9113c7849')
AWS_HOST = os.getenv('AWS_HOST', "https://sin1.contabostorage.com")
IN_DOCKER = bool(os.getenv('IN_DOCKER', True))
DEBUG = bool(os.getenv('DEBUG', True))
logger = logging.getLogger(__name__)
# staging server
# HOSTNAME = os.getenv('HOSTNAME', 'http://208.51.60.130')
# API_KEY = os.getenv('API_KEY', "1b1232488c28e609b16040ea2f2d6bf7dd7c941e")
# live server
HOSTNAME = os.getenv('HOSTNAME', 'http://208.51.60.130')
API_KEY = os.getenv('API_KEY', "fb5b650c7b92ddb5150b7965b58ba3854c87d94b")
from huggingface_hub.hf_api import HfFolder; HfFolder.save_token('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')

if not API_KEY:
    print('=> WARNING! API_KEY is not set')
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

css = """
@import "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/4.7.0/css/font-awesome.min.css";

.aixblock__title .md p {
    display: block;
    text-align: center;
    font-size: 2em;
    font-weight: 700;
}

.aixblock__tabs .tab-nav {
    justify-content: center;
    gap: 8px;
    padding-bottom: 1rem;
    border-bottom: none !important;
}

.aixblock__tabs .tab-nav > button {
    border-radius: 8px;
    border: 1px solid #DEDEEC;
    height: 32px;
    padding: 8px 10px;
    text-align: center;
    line-height: 1em;
}

.aixblock__tabs .tab-nav > button.selected {
    background-color: #5050FF;
    border-color: #5050FF;
    color: #FFFFFF;
}

.aixblock__tabs .tabitem {
    padding: 0;
    border: none;
}

.aixblock__tabs .tabitem .gap.panel {
    background: none;
    padding: 0;
}

.aixblock__input-image,
.aixblock__output-image {
    border: solid 2px #DEDEEC !important;
}

.aixblock__input-image {
    border-style: dashed !important;
}

footer {
    display: none !important;
}

button.secondary,
button.primary {
    border: none !important;
    background-image: none !important;
    color: white !important;
    box-shadow: none !important;
    border-radius: 8px !important;
}

button.primary {
    background-color: #5050FF !important;
}

button.secondary {
    background-color: #F5A !important;
}

.aixblock__input-buttons {
    justify-content: flex-end;
}

.aixblock__input-buttons > button {
    flex: 0 !important;
}

.aixblock__trial-train {
    text-align: center;
    margin-top: 2rem;
}
"""

js = """
window.addEventListener("DOMContentLoaded", function() {
    function process() {
        let buttonsContainer = document.querySelector('.aixblock__input-image')?.parentElement?.nextElementSibling;
        
        if (!buttonsContainer) {
            setTimeout(function() {
                process();
            }, 100);
            return;
        }
        
        document.querySelectorAll('.aixblock__input-image').forEach(function(ele) {
            ele.parentElement.nextElementSibling.classList.add('aixblock__input-buttons');
        });
    }
    
    process();
});
"""
from logging_class import start_queue, write_log
from gradio_webrtc import WebRTC
from twilio.rest import Client
try:
    account_sid = "AC10d8927d012a31a1f5b1696d2e323a4b"
    auth_token = "7ca8d300f9850ddba5919c2ef4df2648"

    if account_sid and auth_token:
        client = Client(account_sid, auth_token)

        token = client.tokens.create()

        rtc_configuration = {
            "iceServers": token.ice_servers,
            "iceTransportPolicy": "relay",
        }
    else:
        rtc_configuration = None
except:
    rtc_configuration = None
# import torch
# DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import os
# GROUNDING_DINO_CONFIG_PATH = os.path.join("/app/GroundingDINO/", "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py")

# MODEL_FILE = os.getenv(
#     'MODEL_FILE', '/app/models/model_final_beta.pth')
# CONFIG_FILE = os.getenv(
#     "CONFIG_FILE", '/app/configs/fcos/fcos_imprv_R_101_FPN_cpu.yaml')
# ===== dev ====
# MODEL_FILE = os.getenv(
#     'MODEL_FILE', '/app/models/model_final_beta.pth')
# CONFIG_FILE = os.getenv(
#     "CONFIG_FILE", '/app/configs/fcos/fcos_imprv_R_101_FPN_cpu.yaml')


CONFIG_FILE = os.getenv(
    "CONFIG_FILE", '/app/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml')
MODEL_FILE = os.getenv(
    'MODEL_FILE', '/app/models/model_final_f10217.pkl')

DEFAULT_MODEL_NAME = os.getenv('DEFAULT_MODEL_NAME', 'model_final_f10217.pkl')
# LABELS_FILE = os.getenv(
# 'LABELS_FILE', '/app/ms_coco_classnames.txt')
LABELS_FILE = os.getenv(
    'LABELS_FILE', '/app/ms_coco_classnames.txt')
SCORE_THRESHOLD = .5


DEVICE = os.getenv('DEVICE', 'cuda')
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="0"
# _redis = redis.Redis(host=REDIS_HOST, port=REDIS_PORT,
#                      db=REDIS_DB, decode_responses=True)

# # delete all keys with prefix detectron2::
# for key in _redis.scan_iter(f"detectron2::*"):
#     _redis.delete(key)


# def set_preview_cache(project_id, local_url, share_url, port):
#     _redis.set(f"detectron2::{project_id}", json.dumps({
#         'local_url': local_url,
#         'share_url': share_url,
#         'port': port
#     }), ex=60*60*24)


# def get_preview_cache(project_id):
#     _redis = redis.Redis(host=REDIS_HOST, port=REDIS_PORT,
#                          db=REDIS_DB, decode_responses=True)

#     if not _redis.exists(f"detectron2::{project_id}"):
#         return None
#     return json.loads(_redis.get(f"detectron2::{project_id}"))


def setup_cfg(config_file):
    print(config_file)
    with open(config_file) as f:
        print(f.read())
    # Load default config
    cfg = get_cfg()
    cfg.MODEL.DEVICE = DEVICE
    # Merge the default with customized one
    cfg.merge_from_file(config_file)
    # Set score_threshold for builtin models
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = SCORE_THRESHOLD
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = SCORE_THRESHOLD
    # cfg.MODEL.FCOS.INFERENCE_TH = SCORE_THRESHOLD
    # cfg.freeze()
    return cfg

image_size = 224
image_transforms = transforms.Compose([
    transforms.Resize(image_size),
    transforms.CenterCrop(image_size),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
image_cache_dir = os.path.join(os.path.dirname(__file__), 'image-cache')
os.makedirs(image_cache_dir, exist_ok=True)

def get_transformed_image(url, transform=True):
    is_local_file = url.startswith('/data')
    if is_local_file:
        filename, dir_path = url.split('/data/')[1].split('?d=')
        dir_path = str(urllib.parse.unquote(dir_path))
        filepath = os.path.join(dir_path, filename)
        if not os.path.exists(filepath):
            raise FileNotFoundError(filepath)
        with open(filepath, mode='rb') as f:
            image = Image.open(f).convert('RGB')
    else:
        cached_file = os.path.join(
            image_cache_dir, hashlib.md5(url.encode()).hexdigest())
        if os.path.exists(cached_file):
            with open(cached_file, mode='rb') as f:
                image = Image.open(f).convert('RGB')
        else:
            r = requests.get(url, stream=True)
            r.raise_for_status()
            with io.BytesIO(r.content) as f:
                image = Image.open(f).convert('RGB')
            with io.open(cached_file, mode='wb') as fout:
                fout.write(r.content)
    if transform:
        return image_transforms(image)
    else:
        return image

from typing import List, Dict, Optional
from aixblock_ml.model import AIxBlockMLBase
import torch.distributed as dist
import os
import torch
import os
import subprocess
import random
import asyncio
import logging
import logging
import base64
import hmac
import json
import hashlib
import zipfile
import subprocess
import shutil
import threading
import requests
import time
# from centrifuge import CentrifugeError, Client, ClientEventHandler, SubscriptionEventHandler

HOST_NAME = os.environ.get('HOST_NAME',"http://127.0.0.1:8080")
from function_ml import connecet_project, download_dataset, upload_checkpoint

def download_checkpoint(weight_zip_path, project_id, checkpoint_id, token):
    url = f"{HOST_NAME}/api/checkpoint_model_marketplace/download/{checkpoint_id}?project_id={project_id}"
    payload = {}
    headers = {
        'accept': 'application/json',
        # 'Authorization': 'Token 5d3604c4c57def9a192950ef7b90d7f1e0bb05c1'
        'Authorization': f'Token {token}'
    }
    response = requests.request("GET", url, headers=headers, data=payload) 
    checkpoint_name = response.headers.get('X-Checkpoint-Name')

    if response.status_code == 200:
        with open(weight_zip_path, 'wb') as f:
            f.write(response.content)
        return checkpoint_name
    
    else: 
        return None

def read_dataset(file_path):
    # Kiểm tra xem thư mục /content/ có tồn tại không
    if os.path.isdir(file_path):
        files = os.listdir(file_path)
        # Kiểm tra xem có file json nào không
        for file in files:
            if file.endswith(".json"):
            # Đọc file json
                with open(os.path.join(file_path, file), "r") as f:
                    data = json.load(f)

                return data
    return None

def is_correct_format(data_json):
    try:
        for item in data_json:
            if not all(key in item for key in ['instruction', 'input', 'output']):
                return False
        return True
    except Exception as e:
        return False
    
def conver_to_hf_dataset(data_json):
    formatted_data = []
    for item in data_json:
        for annotation in item['annotations']:
            question = None
            answer = None
            for result in annotation['result']:
                if result['from_name'] == 'question':
                    question = result['value']['text'][0]
                elif result['from_name'] == 'answer':
                    answer = result['value']['text'][0]
            if question and answer:
                formatted_data.append({
                    'instruction': item['data']['text'],
                    'input': question,
                    'output': answer
                })
    return formatted_data

class SegmentationModel(AIxBlockMLBase):
    def __init__(self,
                #  config_file=CONFIG_FILE,
                #  checkpoint=MODEL_FILE,
                 image_dir=None,
                 labels_file=LABELS_FILE,
                #  score_threshold=SCORE_THRESHOLD,
                 device='cpu',
                 instance_mode=None,
                 **kwargs):

        super(SegmentationModel, self).__init__(**kwargs)
        # Config file and checkpoint for our model
        self.model_path = None
        self.predictor = None
        self.device = device
        # self.checkpoint = checkpoint
        # default AIxBlock image upload folder
        # upload_dir = os.path.join(get_data_dir(), 'media', 'upload')
        # self.image_dir = image_dir or upload_dir

        # print('Load new model from: ', config_file, checkpoint)
        # loading model from checkpoint
        self.device = torch.device(
            'cuda') if torch.cuda.is_available() else torch.device(device)
        # self.device = torch.device(device)
        # self.cfg = setup_cfg(config_file)
        # self.cfg.defrost()
        # self.cfg.MODEL.WEIGHTS = checkpoint

        # self.metadata = MetadataCatalog.get(
        #     self.cfg.DATASETS.TEST[0] if len(
        #         self.cfg.DATASETS.TEST) else "__unused"
        # )
        self.instance_mode = instance_mode
        # self.predictor = DefaultPredictor(self.cfg)
       
        if self.parsed_label_config:
            from_name, schema = list(self.parsed_label_config.items())[0]
            self.from_name = from_name
            self.to_name = schema['to_name'][0]
            self.labels_in_config = [l.lower() for l in schema['labels']]
            print(f"Labels in config: {self.labels_in_config}")
        else:
            self.from_name = ''
            self.to_name = ''
            self.labels_in_config = []
        if os.path.exists(labels_file):
            self.labels_map = json.load(open(labels_file))
        else:
            self.labels_map = {}

    def _get_image_url(self, task):
        pass
        # data_values = list(task['data'].values())
        # image_url = data_values[0]
        # if image_url.startswith('s3://'):
        #     # presign s3 url
        #     r = urlparse(image_url, allow_fragments=False)
        #     bucket_name = r.netloc
        #     key = r.path.lstrip('/')
        #     client = boto3.client('s3', aws_access_key_id=AWS_ACCESS_KEY,
        #           aws_secret_access_key=AWS_SECRET_KEY,
        #           endpoint_url=AWS_HOST)
        #     try:
        #         image_url = client.generate_presigned_url(
        #             ClientMethod='get_object',
        #             Params={'Bucket': bucket_name, 'Key': key}
        #         )
        #         print(image_url)
        #     except ClientError as exc:
        #         logger.warning(
        #             f'Can\'t generate presigned URL for {image_url}. Reason: {exc}')
        # return image_url
    
    def mask_to_polygons(self, mask, max_width, max_height, simplification=0.001):
        # cv2.RETR_CCOMP flag retrieves all the contours and arranges them to a 2-level
        # hierarchy. External contours (boundary) of the object are placed in hierarchy-1.
        # Internal contours (holes) are placed in hierarchy-2.
        # cv2.CHAIN_APPROX_NONE flag gets vertices of polygons from contours.
        # some versions of cv2 does not support incontiguous arr
        mask = np.ascontiguousarray(mask)
        contours, hierarchy = cv2.findContours(mask.astype(
            "uint8"), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
        if hierarchy is None:  # empty mask
            return [], False
        has_holes = (hierarchy.reshape(-1, 4)[:, 3] >= 0).sum() > 0
        res = contours[0]
        print(f"Befor approxPolyDP: {len(res)}")
        epsilon = simplification * cv2.arcLength(res, True) #simplification * 
        res = cv2.approxPolyDP(res, epsilon, True)
        print(f"After approxPolyDP: {len(res)}")
        res = res.reshape(-1, 2)

        # # These coordinates from OpenCV are integers in range [0, W-1 or H-1].
        # # We add 0.5 to turn them into real-value coordinate space. A better solution
        # # would be to first +0.5 and then dilate the returned polygon by 0.5.
        # # res = [list(x + 0.5) for x in res if len(x) >= 6]
        res = list(res+0.25) if len(res) >= 6 else []
        # # res = [[res[i]/max_width*100, res[i+1]/max_height*100]
        # #        for i in range(0, len(res), 2)]
        if res:
            x, y = zip(*res)
            x = [i/max_width*100 for i in x]
            y = [i/max_height*100 for i in y]
            res = list(zip(x, y))

        return res, has_holes

    def get_last_dir(self, path, condition):
        # there exp, exp1, exp2, ..., exp10, ..., exp100
        # get the latest one
        def sort_key(x):
            try:
                return int(x.split('exp')[1])
            except:
                return 0

        # list all dirs containing exp, abs path
        dirs = [os.path.join(path, d) for d in os.listdir(path) if 'exp' in d]

        if dirs:
            dirs = sorted(dirs, key=sort_key, reverse=True)
            print(f"Dirs: {dirs}")
            for d in dirs:
                if condition(d):
                    return d
        else:
            return ''

    def get_training_location(self, project_id):
        if IN_DOCKER:
            return os.path.join("/data", str(project_id), "training")
        else:
            return os.path.join(os.getcwd(), str(project_id), "training")

    def get_latest_model(self, project, default=None):
        if default:
            # cfg = get_cfg()
            # cfg.merge_from_file(CONFIG_FILE)
            # cfg.MODEL.RETINANET.SCORE_THRESH_TEST = SCORE_THRESHOLD
            # cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = SCORE_THRESHOLD
            # cfg.MODEL.WEIGHTS = MODEL_FILE
            # # cfg.MODEL.DEVICE = DEVICE
            # self.cfg = cfg
            # self.predictor = DefaultPredictor(cfg)
            return None, ''

        training_location = self.get_training_location(project)

        def condition(x):
            # check if success.txt exists
            success_file = os.path.join(x, 'model_final.pth')
            if not os.path.exists(success_file):
                return False
            else:
                return True

        try:
            last_dir = self.get_last_dir(training_location, condition)
            if last_dir:
                print("last_dir: ", last_dir)
                # cfg = get_cfg()
                # cfg.merge_from_file(os.path.join(
                #     last_dir, 'config.yaml'))
                # cfg.MODEL.WEIGHTS = os.path.join(
                #     last_dir, 'model_final.pth')
                # cfg.MODEL.RETINANET.SCORE_THRESH_TEST = SCORE_THRESHOLD
                # cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = SCORE_THRESHOLD
                # self.predictor = DefaultPredictor(cfg)
                # self.cfg = cfg
                # self.model_path = last_dir
                return None, last_dir
            else:
                return self.get_latest_model(project, default=True)
        except Exception as e:
            print(f"Get latest model error: {e}")
            # if there is no model, return default model
            return self.get_latest_model(project, default=True)
    
    def transform_output(self, masks, size):
            if self.keep_input_size == True:
                return masks
            else:
                h,w = size
                N = masks.shape[0]
                new_masks = np.zeros((N,h,w), dtype=np.uint8)
                for idx in range(N):
                    new_masks[idx] = cv2.resize(masks[idx], (w,h))
                return new_masks
    
    def predict(self, tasks, **kwargs):
        results = []
        self.hostname = HOSTNAME
        self.access_token = API_KEY


        from segment_anything_2 import SamPredictor, sam_model_registry
        if not self.predictor:
            self.get_latest_model(tasks[0]['project'], default=DEBUG)

        for task in tasks:
            image_url = self._get_image_url(task)

            img = get_transformed_image(HOSTNAME+image_url, transform=False)
            # image_path = self.get_local_path(image_url)
            # load image to PIL
            img_width, img_height = img.size
            # convert img to np array
            img = np.array(img)
            input_point = np.array([[500, 375]])
            input_label = np.array(list("car,person"))
            input_box = np.array([0, 0, img_width, img_height])


            sam = sam_model_registry["vit_h"](checkpoint="./sam_vit_h_4b8939.pth")
            predictor = SamPredictor(sam)
            predictor.set_image(img)
            masks, iou_prediction, _ = predictor.predict(
                point_coords=None,
                point_labels=input_label,
                box=input_box[None, :],
                multimask_output=True,
            )
            print(iou_prediction) 
            scores = None
            points = []
           
            for i, (mask,prediction) in enumerate(zip(masks, iou_prediction)):
                output_label = "car"#self.labels_map[str(classes[i]+1)]
                
                if output_label not in self.labels_in_config:
                    print(output_label + ' label not found in project config.')
                    continue
                if output_label not in self.labels_in_config:
                    self.from_name = 'from_name'
                    self.to_name = 'to_name'
                if len(mask) < 8:
                    continue
                points, has_hole = self.mask_to_polygons(
                    mask=mask, max_width=img_width, max_height=img_height)
                
                if len(points)>0:
                    results.append({
                        "points":points,
                        "regionType": "polygon",
                        "classification": "Car",
                        "from_name": self.from_name,
                        "to_name": self.to_name,
                        "regionType": "polygon",
                        "classification": output_label.capitalize(),
                        "original_width": img_width,
                        "original_height": img_height,
                        # "from_name": self.from_name,
                        # "to_name": self.to_name,
                        # "type": "polygonlabels",
                        # "value": {
                        #     "points": points,
                        #     "polygonlabels": [
                        #         "car"
                        #     ]
                        # },
                        # "original_width": img_width,
                        # "original_height": img_height,
                        # "score": 10,
                        # "id": int(time.time()*1000)
                    })
        if scores is not None:
            avg_score = sum(scores)/max(len(scores), 1)
        else:
            avg_score = 1
        return [{
            'result': results,
            'score': avg_score
        }]

    def __get_annotation__(self, image, polygon):
        def flatten(l):
            # flatten list of dictionaries. Take dict['x'] value first, then dict['y'] value.
            flat = []
            for i in l:
                flat.append(i['x'])
                flat.append(i['y'])
            return flat

        if isinstance(polygon, list):
            segmentation = polygon
            if len(segmentation) == 0:
                return None, None, None
            #     Split the polygon points into x and y coordinate lists:
            x_coords = polygon[0::2]
            y_coords = polygon[1::2]
            min_x, max_x = min(x_coords), max(x_coords)
            min_y, max_y = min(y_coords), max(y_coords)
        else:
            points = polygon['points']
            if len(points) == 0:
                return None, None, None
            segmentation = [flatten(points)]
            x_coords = [point['x'] for point in points]
            y_coords = [point['y'] for point in points]
            min_x, max_x = min(x_coords), max(x_coords)
            min_y, max_y = min(y_coords), max(y_coords)
        bbox = [min_x, min_y, max_x, max_y]

        width = max_x - min_x
        height = max_y - min_y

        area = width * height

        return segmentation, bbox, area

    def generate_dataset(self, annotations, workdir=None, **kwargs):
        def is_skipped(completion):
            if len(completion['annotations']) != 1:
                return False
            completion = completion['annotations'][0]
            return completion.get('skipped', False) or completion.get('was_cancelled', False)
        
        workdir = os.path.abspath(workdir)
        image_urls, image_classes = [], []
        project_id = None
        image_classes = []
        polygons = []
        for annotation in annotations:
            if is_skipped(annotation):
                continue
            if project_id is None:
                project_id = annotation['project']
            if not annotation.get('annotations'):
                print(
                    f"Image {annotation['data']['image']} has no annotations.")
                continue
            polygon = annotation['annotations'][0]['result']
            for p in polygon:
                if p.get('classification') is None:
                    continue
                if p.get('regionType', '') != 'polygon':
                    continue
                else:
                    if self.__get_annotation__(annotation['data']['image'], p)[0] is None:
                        print(
                            f"Image {annotation['data']['image']} has no annotations.")
                        continue
                    image_classes.append(p['classification'].lower())
                    polygons.append(p)
                    image_urls.append(HOSTNAME+annotation['data']['image'])
        if not polygons or not image_urls:
            return '', project_id
        image_classes = list(set(image_classes))

        print(f"Polygons: {len(polygons)}")
        print(f"Images: {len(image_urls)}")

        print(f'Creating dataset with {len(image_urls)} images...')
        data_location = os.path.abspath(os.path.join(
            workdir, "datasets"))

        if not os.path.exists(data_location):
            os.makedirs(data_location)

        # date_created in this format "2023-02-07T16:34:15+00:00" (timezone)
        annotation_json = {
            "info": {
                "year": str(datetime.datetime.now().year),
                "version": "1",
                "description": f"Dataset generated from project {project_id}",
                "contributor": "",
                "url": "",
                "date_created": datetime.datetime.now().isoformat()
            },
            "licenses": [
                {
                    "id": 1,
                    "url": "https://creativecommons.org/licenses/by/4.0/",
                    "name": "CC BY 4.0"
                }
            ],
            "categories": [
                {"id": 0, "name": "WowAI", "supercategory": "none"},
            ],
        }

        annotation_json['categories'] += [
            {"id": i+1, "name": image_classes[i].lower(),
                "supercategory": "WowAI"}
            for i in range(0, len(image_classes))
        ]

        def image_class_to_index(img_class):
            # given image_class as value of annotation_json['categories'][i]["names"], return index
            for i in range(len(annotation_json['categories'])):
                if annotation_json['categories'][i]["name"] == img_class.lower() and annotation_json['categories'][i]["supercategory"] != "none":
                    return i
            print(img_class)
            print(annotation_json['categories'])
            raise Exception("Image class not found")

        # create train, val, test directories
        train_location = os.path.join(data_location, "train")
        val_location = os.path.join(data_location, "val")
        if not os.path.exists(train_location):
            os.makedirs(train_location)
        if not os.path.exists(val_location):
            os.makedirs(val_location)

        # download images to train, val directories. 80% train, 20% val. Each directory has images and _annotations.coco.json file contains the images and annotations
        train_image_urls = image_urls[:int(len(image_urls)*0.8)]
        val_image_urls = image_urls[int(len(image_urls)*0.8):]
        train_polygons = polygons[:int(len(polygons)*0.8)]
        val_polygons = polygons[int(len(polygons)*0.8):]

        # create train dataset
        train_annotations_json = annotation_json.copy()
        train_images = []
        for i, image_url in enumerate(train_image_urls):
            image_path = os.path.join(train_location, f'{i}.jpg')
            # save image
            print(image_url)
            image = requests.get(
                image_url).content
            img = Image.open(io.BytesIO(image))
            # Convert the RGBA image to RGB
            img = img.convert('RGB')

            # resize image to 224x224
            img = img.resize((224, 224))

            # save image as jpg
            img.save(image_path, "JPEG", quality=100,
                     optimize=True, progressive=True)

            train_images.append({
                "id": i,
                "width": img.size[0],
                "height": img.size[1],
                "file_name": f'{i}.jpg',
                "license": 1,
                "date_captured": datetime.datetime.now().isoformat()
            })
        train_annotations_json['images'] = train_images

        train_annotations = []
        for i, polygon in enumerate(train_polygons):
            image_path = os.path.join(train_location, f'{i}.jpg')
            # read image as numpy array
            image = cv2.imread(image_path)
            segmentation, bbox, area = self.__get_annotation__(image, polygon)
            train_annotations.append({
                "id": i,
                "image_id": i,
                "category_id": image_class_to_index(polygon['classification']),
                "segmentation":  segmentation,
                "area": area,
                "bbox": bbox,
                # "bbox_mode": BoxMode.XYXY_ABS,
                "iscrowd": 0
            })

        train_annotations_json['annotations'] = train_annotations

        # write train annotations to file _annotations.coco.json inside train directory
        with open(os.path.join(train_location, '_annotations.coco.json'), 'w') as f:
            train_annotations_json = eval(str(train_annotations_json))
            json.dump(train_annotations_json, f)

        # create val dataset
        val_annotations_json = annotation_json.copy()
        val_images = []
        for i, image_url in enumerate(val_image_urls):
            image_path = os.path.join(val_location, f'{i}.jpg')
            image = requests.get(
                image_url).content
            img = Image.open(io.BytesIO(image))
            # save image as jpg
            img = img.convert('RGB')
            img = img.resize((224, 224))
            img.save(image_path, "JPEG", quality=100,
                     optimize=True, progressive=True)
            val_images.append({
                "id": i,
                "width": img.size[0],
                "height": img.size[1],
                "file_name": f'{i}.jpg',
                "license": 1,
                "date_captured": datetime.datetime.now().isoformat()
            })
        val_annotations_json['images'] = val_images

        val_annotations = []
        for i, polygon in enumerate(val_polygons):
            image_path = os.path.join(val_location, f'{i}.jpg')
            # read image as numpy array
            image = cv2.imread(image_path)
            segmentation, bbox, area = self.__get_annotation__(image, polygon)
            val_annotations.append({
                "id": i,
                "image_id": i,
                "category_id": image_class_to_index(polygon['classification']),
                "segmentation": segmentation,  # [flatten(polygon['points'])],
                "area": area,
                "bbox": bbox,
                # "bbox_mode": BoxMode.XYXY_ABS,
                "iscrowd": 0
            })

        val_annotations_json['annotations'] = val_annotations

        # write val_annotations to file _annotations.coco.json inside val directory
        with open(os.path.join(val_location, '_annotations.coco.json'), 'w') as f:
            val_annotations_json = eval(str(val_annotations_json))
            json.dump(val_annotations_json, f)

        print(data_location, project_id)

        return data_location, project_id
    
    # def configure_opt(cfg: Box, model: Model):
    #     def lr_lambda(step):
    #         if step < cfg.opt.warmup_steps:
    #             return step / cfg.opt.warmup_steps
    #         elif step < cfg.opt.steps[0]:
    #             return 1.0
    #         elif step < cfg.opt.steps[1]:
    #             return 1 / cfg.opt.decay_factor
    #         else:
    #             return 1 / (cfg.opt.decay_factor**2)

        optimizer = torch.optim.Adam(model.model.parameters(), lr=cfg.opt.learning_rate, weight_decay=cfg.opt.weight_decay)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        return optimizer, scheduler
    
    # def validate(fabric: L.Fabric, model: Model, val_dataloader: DataLoader, epoch: int = 0):
    #     model.eval()
    #     # ious = AverageMeter()
    #     # f1_scores = AverageMeter()

    #     # with torch.no_grad():
    #     #     for iter, data in enumerate(val_dataloader):
    #     #         images, bboxes, gt_masks = data
    #     #         num_images = images.size(0)
    #     #         pred_masks, _ = model(images, bboxes)
    #     #         for pred_mask, gt_mask in zip(pred_masks, gt_masks):
    #     #             batch_stats = smp.metrics.get_stats(
    #     #                 pred_mask,
    #     #                 gt_mask.int(),
    #     #                 mode='binary',
    #     #                 threshold=0.5,
    #     #             )
    #     #             batch_iou = smp.metrics.iou_score(*batch_stats, reduction="micro-imagewise")
    #     #             batch_f1 = smp.metrics.f1_score(*batch_stats, reduction="micro-imagewise")
    #     #             ious.update(batch_iou, num_images)
    #     #             f1_scores.update(batch_f1, num_images)
    #     #         fabric.print(
    #     #             f'Val: [{epoch}] - [{iter}/{len(val_dataloader)}]: Mean IoU: [{ious.avg:.4f}] -- Mean F1: [{f1_scores.avg:.4f}]'
    #     #         )

    #     # fabric.print(f'Validation [{epoch}]: Mean IoU: [{ious.avg:.4f}] -- Mean F1: [{f1_scores.avg:.4f}]')

    #     # fabric.print(f"Saving checkpoint to {cfg.out_dir}")
    #     # state_dict = model.model.state_dict()
    #     # if fabric.global_rank == 0:
    #     #     torch.save(state_dict, os.path.join(cfg.out_dir, f"epoch-{epoch:06d}-f1{f1_scores.avg:.2f}-ckpt.pth"))
    #     # model.train()

    # def train_sam(
    #     cfg: Box,
    #     fabric: L.Fabric,
    #     model: Model,
    #     optimizer: _FabricOptimizer,
    #     scheduler: _FabricOptimizer,
    #     train_dataloader: DataLoader,
    #     val_dataloader: DataLoader,
    # ):
    #     """The SAM training loop."""

    #     focal_loss = FocalLoss()
    #     dice_loss = DiceLoss()

    #     for epoch in range(1, cfg.num_epochs):
    #         batch_time = AverageMeter()
    #         data_time = AverageMeter()
    #         focal_losses = AverageMeter()
    #         dice_losses = AverageMeter()
    #         iou_losses = AverageMeter()
    #         total_losses = AverageMeter()
    #         end = time.time()
    #         validated = False

    #         for iter, data in enumerate(train_dataloader):
    #             if epoch > 1 and epoch % cfg.eval_interval == 0 and not validated:
    #                 validate(fabric, model, val_dataloader, epoch)
    #                 validated = True

    #             data_time.update(time.time() - end)
    #             images, bboxes, gt_masks = data
    #             batch_size = images.size(0)
    #             pred_masks, iou_predictions = model(images, bboxes)
    #             num_masks = sum(len(pred_mask) for pred_mask in pred_masks)
    #             loss_focal = torch.tensor(0., device=fabric.device)
    #             loss_dice = torch.tensor(0., device=fabric.device)
    #             loss_iou = torch.tensor(0., device=fabric.device)
    #             for pred_mask, gt_mask, iou_prediction in zip(pred_masks, gt_masks, iou_predictions):
    #                 batch_iou = calc_iou(pred_mask, gt_mask)
    #                 loss_focal += focal_loss(pred_mask, gt_mask, num_masks)
    #                 loss_dice += dice_loss(pred_mask, gt_mask, num_masks)
    #                 loss_iou += F.mse_loss(iou_prediction, batch_iou, reduction='sum') / num_masks

    #             loss_total = 20. * loss_focal + loss_dice + loss_iou
    #             optimizer.zero_grad()
    #             fabric.backward(loss_total)
    #             optimizer.step()
    #             scheduler.step()
    #             batch_time.update(time.time() - end)
    #             end = time.time()

    #             focal_losses.update(loss_focal.item(), batch_size)
    #             dice_losses.update(loss_dice.item(), batch_size)
    #             iou_losses.update(loss_iou.item(), batch_size)
    #             total_losses.update(loss_total.item(), batch_size)

    #             fabric.print(f'Epoch: [{epoch}][{iter+1}/{len(train_dataloader)}]'
    #                         f' | Time [{batch_time.val:.3f}s ({batch_time.avg:.3f}s)]'
    #                         f' | Data [{data_time.val:.3f}s ({data_time.avg:.3f}s)]'
    #                         f' | Focal Loss [{focal_losses.val:.4f} ({focal_losses.avg:.4f})]'
    #                         f' | Dice Loss [{dice_losses.val:.4f} ({dice_losses.avg:.4f})]'
    #                         f' | IoU Loss [{iou_losses.val:.4f} ({iou_losses.avg:.4f})]'
    #                         f' | Total Loss [{total_losses.val:.4f} ({total_losses.avg:.4f})]')
    
    def fit(self, annotations, workdir=None, batch_size=1, num_epochs=10, **kwargs):
        # self.cfg = setup_cfg(CONFIG_FILE)
        data_location, project_id = self.generate_dataset(
            annotations, workdir, **kwargs)

        if not data_location or not project_id:
            print(
                f"No data found for project. Skipping training. Workdir: {workdir}\nAnnotations: {annotations}")
            return {'model_path': '', 'classes': []}

        training_location = self.get_training_location(project_id)

        # create training folder if it doesn't exist
        if not os.path.exists(training_location):
            os.makedirs(training_location)

        def condition(x):
            # check if success.txt exists
            success_file = os.path.join(x, 'model_final.pth')
            if not os.path.exists(success_file):
                return False
            else:
                return True

        latest_exp_dir = self.get_last_dir(training_location, condition)
        if not latest_exp_dir:
            # create first experiment folder
            current_exp_numb = 0
            current_exp_location = os.path.join(
                training_location, f"exp{current_exp_numb}")
            os.makedirs(current_exp_location)
            cfg, model_path = self.get_latest_model(project_id, default=True)
        else:
            current_exp_numb = int(latest_exp_dir.split(os.sep)[-1][3:]) + 1
            cfg, model_path = self.get_latest_model(project_id, default=False)
            current_exp_location = os.path.join(
                training_location, f"exp{current_exp_numb}")
            os.makedirs(current_exp_location)

        # get second last part of data_location path
        data_location_name = data_location.split(os.sep)[-2]

        train_dataset_name = f"dataset_train_{data_location_name}_exp{current_exp_numb}"
        val_dataset_name = f"dataset_val_{data_location_name}_exp{current_exp_numb}"
        
        # register_coco_instances(train_dataset_name, {
        # }, os.path.join(data_location, "train/_annotations.coco.json"), os.path.join(data_location, "train"))
        # register_coco_instances(val_dataset_name, {
        # }, os.path.join(data_location, "val/_annotations.coco.json"), os.path.join(data_location, "val"))

        # read annotations.coco.json from train dir
        with open(os.path.join(data_location, "train/_annotations.coco.json"), 'r') as f:
            annotations_json = json.load(f)
            categories = annotations_json['categories']

        # free CUDA memory
        torch.cuda.empty_cache()
        # fabric = L.Fabric(accelerator="auto",
        #               devices=cfg.num_devices,
        #               strategy="auto",
        #               loggers=[TensorBoardLogger(cfg.out_dir, name="lightning-sam")])
        # fabric.launch()
        # fabric.seed_everything(1337 + fabric.global_rank)

        # if fabric.global_rank == 0:
        #     os.makedirs(cfg.out_dir, exist_ok=True)

        # with fabric.device:
        #     model = Model(cfg)
        #     model.setup()

        # train_data, val_data = load_datasets(root_dir=data_location,annotation_file= os.path.join(data_location, "train/_annotations.coco.json"),val_annotation_file= os.path.join(data_location, "val/_annotations.coco.json"), img_size=model.model.image_encoder.img_size,batch_size=8,num_workers=4) #load_datasets(cfg, model.model.image_encoder.img_size)
        # train_data = fabric._setup_dataloader(train_data)
        # val_data = fabric._setup_dataloader(val_data)

        # optimizer, scheduler = self.configure_opt(cfg, model)
        # model, optimizer = fabric.setup(model, optimizer)
        # self.train_sam(cfg, fabric, model, optimizer, scheduler, train_data, val_data)
        # self.validate(fabric, model, val_data, epoch=0)

        # write success.txt file contain current time format
        with open(os.path.join(current_exp_location, 'success.txt'), 'w') as f:
            f.write(str(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))

        # export cfg to yaml file
        with open(os.path.join(current_exp_location, 'config.yaml'), 'w') as f:
            f.write(cfg.dump())

        # clear cuda cache
        torch.cuda.empty_cache()

        return {'model_path': os.path.join(
            current_exp_location, 'model_final.pth'), 'classes': [i['name'] for i in categories]}
    
    def action(self, project, command, collection, **kwargs):
        if command.lower() == "train":
            try:
                # checkpoint = kwargs.get("checkpoint")
                # aixblock 
                #
                args = ('dummy', )

                clone_dir = os.path.join(os.getcwd())
                epochs = kwargs.get("num_epochs", 10)
                imgsz = kwargs.get("imgsz", 224)
                project_id = kwargs.get("project_id")
                token = kwargs.get("token")
                checkpoint_version = kwargs.get("checkpoint_version")
                checkpoint_id = kwargs.get("checkpoint")
                dataset_version = kwargs.get("dataset_version")
                dataset_id = kwargs.get("dataset")
                channel_log = kwargs.get("channel_log", "training_logs")
                world_size = kwargs.get("world_size", "1")
                rank = kwargs.get("rank", "0")
                master_add = kwargs.get("master_add")
                master_port = kwargs.get("master_port", "12345")
                # entry_file = kwargs.get("entry_file")
                configs = kwargs.get("configs")

                log_queue, logging_thread = start_queue(channel_log)
                write_log(log_queue)

                def func_train_model(clone_dir, project_id, epochs, token, checkpoint_version, checkpoint_id, dataset_version, dataset_id):
                    os.makedirs(f'{clone_dir}/data_zip', exist_ok=True)
                    print(HOST_NAME, token, project_id)
                    project = connecet_project(HOST_NAME, token, project_id)

                    weight_path = os.path.join(clone_dir, f"models")
                    dataset_path = "data"
                    datasets_train = "alpaca_dataset"
                    models_train = "stas/tiny-random-llama-2"

                    if checkpoint_version and checkpoint_id:
                        weight_path = os.path.join(clone_dir, f"models/{checkpoint_version}")
                        if not os.path.exists(weight_path):
                            weight_zip_path = os.path.join(clone_dir, "data_zip/weights.zip")
                            checkpoint_name = download_checkpoint(weight_zip_path, project_id, checkpoint_id, token)
                            if checkpoint_name:
                                print(weight_zip_path)
                                with zipfile.ZipFile(weight_zip_path, 'r') as zip_ref:
                                    zip_ref.extractall(weight_path)

                    if dataset_version and dataset_id:
                        dataset_path = os.path.join(clone_dir, f"datasets/{dataset_version}")

                        if not os.path.exists(dataset_path):
                            data_path = os.path.join(clone_dir, "data_zip")
                            os.makedirs(data_path, exist_ok=True)
                            # data_zip_dir = os.path.join(clone_dir, "data_zip/data.zip")
                            # dataset_name = download_dataset(data_zip_dir, project_id, dataset_id, token)
                            dataset_name = download_dataset(project, dataset_id, data_path)
                            print(dataset_name)
                            if dataset_name: 
                                data_zip_dir = os.path.join(data_path, dataset_name)
                                # if not os.path.exists(dataset_path):
                                with zipfile.ZipFile(data_zip_dir, 'r') as zip_ref:
                                    zip_ref.extractall(dataset_path)
                    # files = [os.path.join(weight_path, filename) for filename in os.listdir(weight_path) if os.path.isfile(os.path.join(weight_path, filename))]
                    # train_dir = os.path.join(os.getcwd(),f"yolov9/runs/train")

                    # script_path = os.path.join(os.getcwd(),f"yolov9/train.py")

                    train_dir = os.path.join(os.getcwd(), "models")
                    log_dir = os.path.join(os.getcwd(), "logs")
                    os.makedirs(train_dir, exist_ok=True)
                    os.makedirs(log_dir, exist_ok=True)

                    print("log_dir")
                    if dataset_version:
                        log_profile =  os.path.join(log_dir, "dataset_version")
                    else:
                        log_profile =  os.path.join(log_dir, "profiler")

                    print(log_profile)

                    # train_dir = os.path.join(os.getcwd(), "models")

                    os.environ["LOGLEVEL"] = "ERROR"
                    from datasets import load_dataset
                    from torch.optim import Adam
                    import monai
                    import torch
                    import numpy as np
                    from torch.utils.data import DataLoader
                    from transformers import SamProcessor, SamModel
                    from torch.utils.data import Dataset

                    def get_bounding_box(ground_truth_map):
                        # get bounding box from mask
                        y_indices, x_indices = np.where(ground_truth_map > 0)
                        x_min, x_max = np.min(x_indices), np.max(x_indices)
                        y_min, y_max = np.min(y_indices), np.max(y_indices)
                        # add perturbation to bounding box coordinates
                        H, W = ground_truth_map.shape
                        x_min = max(0, x_min - np.random.randint(0, 20))
                        x_max = min(W, x_max + np.random.randint(0, 20))
                        y_min = max(0, y_min - np.random.randint(0, 20))
                        y_max = min(H, y_max + np.random.randint(0, 20))
                        bbox = [x_min, y_min, x_max, y_max]

                        return bbox
                    class SAMDataset(Dataset):
                        def __init__(self, dataset, processor):
                            self.dataset = dataset
                            self.processor = processor

                        def __len__(self):
                            return len(self.dataset)

                        def __getitem__(self, idx):
                            item = self.dataset[idx]
                            image = item["image"]
                            ground_truth_mask = np.array(item["label"])

                            # get bounding box prompt
                            prompt = get_bounding_box(ground_truth_mask)

                            # prepare image and prompt for the model
                            inputs = self.processor(image, input_boxes=[[prompt]], return_tensors="pt")

                            # remove batch dimension which the processor adds by default
                            inputs = {k:v.squeeze(0) for k,v in inputs.items()}

                            # add ground truth segmentation
                            inputs["ground_truth_mask"] = ground_truth_mask

                            return inputs
                    
                    # device = "cuda" if torch.cuda, is available() else "cpu"
                    # model = SamModel.from_pretrained("facebook/san-vit-huge")
                    model = SamModel.from_pretrained("facebook/sam-vit-base")
                    device = "cuda" if torch.cuda.is_available() else "cpu"
                    model.to(device)

                    dataset = load_dataset("nielsr/breast-cancer", split="train")
                    processor = SamProcessor.from_pretrained("facebook/sam-vit-base")
                    train_dataset = SAMDataset(dataset=dataset, processor=processor)
                    train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True)

                    # Note: Hyperparameter tuning could improve performance here
                    optimizer = Adam(model.mask_decoder.parameters(), lr=1e-5, weight_decay=0)

                    seg_loss = monai.losses.DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')
                    from tqdm import tqdm
                    from statistics import mean
                    from torch.nn.functional import threshold, normalize

                    num_epochs = epochs
                    
                    model.train()
                    for epoch in range(num_epochs):
                        epoch_losses = []
                        for batch in tqdm(train_dataloader):
                            # forward pass
                            outputs = model(pixel_values=batch["pixel_values"].to(device),
                                            input_boxes=batch["input_boxes"].to(device),
                                            multimask_output=False)

                        # compute loss
                        predicted_masks = outputs.pred_masks.squeeze(1)
                        ground_truth_masks = batch["ground_truth_mask"].float().to(device)
                        loss = seg_loss(predicted_masks, ground_truth_masks.unsqueeze(1))

                        # backward pass (compute gradients of parameters w.r.t. loss)
                        optimizer.zero_grad()
                        loss.backward()

                        # optimize
                        optimizer.step()
                        epoch_losses.append(loss.item())

                        print(f'EPOCH: {epoch}')
                        print(f'Mean loss: {mean(epoch_losses)}')

                    model.push_to_hub( repo_id="segment_anything_v2",
                        commit_message="SAM2",
                        private=True,
                        branch="main",
                        create_pr=False,
                        token="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")
                    # free the memory again

                    del model
                    torch.cuda.empty_cache()


                    # if configs and configs["entry_file"] != "":
                    #     command = [
                    #         "torchrun",
                    #         "--nproc_per_node", "1", #< count gpu card in compute
                    #         "--rdzv-backend", "c10d",
                    #         "--node-rank", f'{rank}',
                    #         "--nnodes", f'{world_size}',
                    #         "--rdzv-endpoint", f'{master_add}:{master_port}',
                    #         "--master-addr", f'{master_add}',
                    #         "--master-port", f'{master_port}',
                    #         f'{configs["entry_file"]}'
                    #     ]

                    #     if configs["arguments"] and len(configs["arguments"])>0:
                    #         args = configs["arguments"]
                    #         for arg in args:
                    #             command.append(arg['name'])
                    #             command.append(arg['value'])

                    # else:
                    #     command = [
                    #         "torchrun",
                    #         "--nproc_per_node", "1", #< count gpu card in compute
                    #         "--rdzv-backend", "c10d",
                    #         "--node-rank", f'{rank}',
                    #         "--nnodes", f'{world_size}',
                    #         "--rdzv-endpoint", f'{master_add}:{master_port}',
                    #         "--master-addr", f'{master_add}',
                    #         "--master-port", f'{master_port}',
                    #         "llama_recipes/finetuning.py",
                    #         "--model_name", f'{models_train}',
                    #         "--use_peft", 
                    #         "--num_epochs", f'{epochs}',
                    #         "--batch_size_training", "2",
                    #         "--peft_method", "lora",
                    #         "--dataset", f'{datasets_train}',
                    #         "--save_model",
                    #         "--dist_checkpoint_root_folder", "model_checkpoints",
                    #         "--dist_checkpoint_folder", "fine-tuned",
                    #         "--pure_bf16",
                    #         "--save_metrics",
                    #         "--output_dir", "/app/models/",
                    #         "--use_profiler",
                    #         "--profiler_dir", f'{log_profile}'
                    #     ]

                    # subprocess.run(command, shell=True)
                    # run_train(command, channel_log)
                    # from sam2.training import train
                    # parser = ArgumentParser()
                    # parser.add_argument(
                    #     "-c",
                    #     "--config",
                    #     required=True,
                    #     type=str,
                    #     default="/app/sam2/configs/sam2.1_training/sam2.1_hiera_b+_MOSE_finetune.yaml",
                    #     help="path to config file (e.g. configs/sam2.1_training/sam2.1_hiera_b+_MOSE_finetune.yaml)",
                    # )
                    # parser.add_argument(
                    #     "--use-cluster",
                    #     type=int,
                    #     default=0,
                    #     help="whether to launch on a cluster, 0: run locally, 1: run on a cluster",
                    # )
                    # # parser.add_argument("--partition", type=str, default=None, help="SLURM partition")
                    # # parser.add_argument("--account", type=str, default=None, help="SLURM account")
                    # # parser.add_argument("--qos", type=str, default=None, help="SLURM qos")
                    # parser.add_argument(
                    #     "--num-gpus", type=int, default=1, help="number of GPUS per node"
                    # )
                    # # parser.add_argument("--num-nodes", type=int, default=None, help="Number of nodes")
                    # args = parser.parse_args()
                    # args.use_cluster = bool(args.use_cluster) if args.use_cluster is not None else None
                    # train.main(parser)
                    # import subprocess
                    # subprocess.run(["python3.10", "/app/sam2/training/train.py", "-c", "configs/sam2.1_training/sam2.1_hiera_b+_MOSE_finetune.yaml", "--use-cluster", "0", "--num-gpus", "1"])
                    # # python training/train.py \
                    # # -c configs/sam2.1_training/sam2.1_hiera_b+_MOSE_finetune.yaml \
                    # # --use-cluster 0 \
                    # # --num-gpus 1
                    # checkpoint_model = f'{train_dir}'
                    # checkpoint_model_zip = f'{train_dir}.zip'
                    # shutil.make_archive(checkpoint_model_zip, 'zip', checkpoint_model)

                    # if os.path.exists(checkpoint_model_zip):
                    #     # print(checkpoint_model)
                    #     checkpoint_name = upload_checkpoint(checkpoint_model_zip, project_id, token)
                    #     if checkpoint_name:
                    #         models_train = "/app/models/"
                    output_dir = "./data/checkpoint"

                    import datetime
                    output_dir = "./data/checkpoint"
                    now = datetime.datetime.now()
                    date_str = now.strftime("%Y%m%d")
                    time_str = now.strftime("%H%M%S")
                    version = f'{date_str}-{time_str}'

                    upload_checkpoint(project, version, output_dir)

                train_thread = threading.Thread(target=func_train_model, args=(clone_dir, project_id, epochs, token, checkpoint_version, checkpoint_id, dataset_version, dataset_id, ))

                train_thread.start()

                return {"message": "train completed successfully"}
                # # use cache to retrieve the data from the previous fit() runs
              
                # os.environ['MASTER_ADDR'] = 'localhost'
                # port = 29500 + random.randint(0, 500)
                # os.environ['MASTER_PORT'] = f'{port}'
                # print(f"Using localhost:{port=}")
                
                # torch.multiprocessing.spawn(self.debug_sp(0), nprocs=1, args=args)

                # return {"message": "train completed successfully"}
            except Exception as e:
                return {"message": f"train failed: {e}"}

        elif command.lower() == "tensorboard":
            def run_tensorboard():
                # train_dir = os.path.join(os.getcwd(), "{project_id}")
                # log_dir = os.path.join(os.getcwd(), "logs")
                p = subprocess.Popen(f"tensorboard --logdir ./logs --host 0.0.0.0 --port=6006", stdout=subprocess.PIPE, stderr=None, shell=True)
                out = p.communicate()
                print(out)

            tensorboard_thread = threading.Thread(target=run_tensorboard)
            tensorboard_thread.start()
            return {"message": "tensorboardx started successfully"}
        
        elif command.lower() == "dashboard":
            return {"Share_url": ""}
          
        elif command.lower() == "predict":
            import cv2
            import numpy as np
            from PIL import Image
            results = []
            imagebase64 = kwargs.get("image","")
            prompt = kwargs.get("prompt", None)
            model_id = kwargs.get("model_id", "")
            context = kwargs.get("text", "")
            token_length = kwargs.get("token_lenght", "")
            tasks = kwargs.get("task", "")
            voice = kwargs.get("voice", None)

            # image_64 = kwargs.get("image")
            image_64 = kwargs.get("image",None)
            if not image_64:
                image_64 = kwargs.get('data', {}).get('image')

            data = kwargs.get("data", None)
            confidence_threshold = kwargs.get("confidence_threshold", 0.2)
            iou_threshold  = kwargs.get("iou_threshold", 0.2)

            results = []
            # read base64 image to cv2
            image = image_64 #data["image"]
            image = image.replace('data:image/png;base64,', '')

            img = Image.open(io.BytesIO(
                base64.b64decode(image))).convert('RGB')
            img_width, img_height = img.size 

            print(f"image size: {img_width}x{img_height}")
            # img = np.array(img)
            import random
            from dataclasses import dataclass
            from typing import Any, List, Dict, Optional, Union, Tuple

            import cv2
            import torch
            import requests
            import numpy as np
            from PIL import Image
            import plotly.express as px
            import matplotlib.pyplot as plt
            import plotly.graph_objects as go
            from transformers import AutoModelForMaskGeneration, AutoProcessor, pipeline
            @dataclass
            class BoundingBox:
                xmin: int
                ymin: int
                xmax: int
                ymax: int

                @property
                def xyxy(self) -> List[float]:
                    return [self.xmin, self.ymin, self.xmax, self.ymax]

            @dataclass
            class DetectionResult:
                score: float
                label: str
                box: BoundingBox
                mask: Optional[np.array] = None

                @classmethod
                def from_dict(cls, detection_dict: Dict) -> 'DetectionResult':
                    return cls(score=detection_dict['score'],
                            label=detection_dict['label'],
                            box=BoundingBox(xmin=detection_dict['box']['xmin'],
                                            ymin=detection_dict['box']['ymin'],
                                            xmax=detection_dict['box']['xmax'],
                                            ymax=detection_dict['box']['ymax']))
            def annotate(image: Union[Image.Image, np.ndarray], detection_results: List[DetectionResult]) -> np.ndarray:
                # Convert PIL Image to OpenCV format
                image_cv2 = np.array(image) if isinstance(image, Image.Image) else image
                image_cv2 = cv2.cvtColor(image_cv2, cv2.COLOR_RGB2BGR)

                # Iterate over detections and add bounding boxes and masks
                for detection in detection_results:
                    label = detection.label
                    score = detection.score
                    box = detection.box
                    mask = detection.mask

                    # Sample a random color for each detection
                    color = np.random.randint(0, 256, size=3)

                    # Draw bounding box
                    cv2.rectangle(image_cv2, (box.xmin, box.ymin), (box.xmax, box.ymax), color.tolist(), 2)
                    cv2.putText(image_cv2, f'{label}: {score:.2f}', (box.xmin, box.ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color.tolist(), 2)

                    # If mask is available, apply it
                    if mask is not None:
                        # Convert mask to uint8
                        mask_uint8 = (mask * 255).astype(np.uint8)
                        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        cv2.drawContours(image_cv2, contours, -1, color.tolist(), 2)

                return cv2.cvtColor(image_cv2, cv2.COLOR_BGR2RGB)

            def plot_detections(
                image: Union[Image.Image, np.ndarray],
                detections: List[DetectionResult],
                save_name: Optional[str] = None
            ) -> None:
                annotated_image = annotate(image, detections)
                plt.imshow(annotated_image)
                plt.axis('off')
                if save_name:
                    plt.savefig(save_name, bbox_inches='tight')
                plt.show()
            def random_named_css_colors(num_colors: int) -> List[str]:
                """
                Returns a list of randomly selected named CSS colors.

                Args:
                - num_colors (int): Number of random colors to generate.

                Returns:
                - list: List of randomly selected named CSS colors.
                """
                # List of named CSS colors
                named_css_colors = [
                    'aliceblue', 'antiquewhite', 'aqua', 'aquamarine', 'azure', 'beige', 'bisque', 'black', 'blanchedalmond',
                    'blue', 'blueviolet', 'brown', 'burlywood', 'cadetblue', 'chartreuse', 'chocolate', 'coral', 'cornflowerblue',
                    'cornsilk', 'crimson', 'cyan', 'darkblue', 'darkcyan', 'darkgoldenrod', 'darkgray', 'darkgreen', 'darkgrey',
                    'darkkhaki', 'darkmagenta', 'darkolivegreen', 'darkorange', 'darkorchid', 'darkred', 'darksalmon', 'darkseagreen',
                    'darkslateblue', 'darkslategray', 'darkslategrey', 'darkturquoise', 'darkviolet', 'deeppink', 'deepskyblue',
                    'dimgray', 'dimgrey', 'dodgerblue', 'firebrick', 'floralwhite', 'forestgreen', 'fuchsia', 'gainsboro', 'ghostwhite',
                    'gold', 'goldenrod', 'gray', 'green', 'greenyellow', 'grey', 'honeydew', 'hotpink', 'indianred', 'indigo', 'ivory',
                    'khaki', 'lavender', 'lavenderblush', 'lawngreen', 'lemonchiffon', 'lightblue', 'lightcoral', 'lightcyan', 'lightgoldenrodyellow',
                    'lightgray', 'lightgreen', 'lightgrey', 'lightpink', 'lightsalmon', 'lightseagreen', 'lightskyblue', 'lightslategray',
                    'lightslategrey', 'lightsteelblue', 'lightyellow', 'lime', 'limegreen', 'linen', 'magenta', 'maroon', 'mediumaquamarine',
                    'mediumblue', 'mediumorchid', 'mediumpurple', 'mediumseagreen', 'mediumslateblue', 'mediumspringgreen', 'mediumturquoise',
                    'mediumvioletred', 'midnightblue', 'mintcream', 'mistyrose', 'moccasin', 'navajowhite', 'navy', 'oldlace', 'olive',
                    'olivedrab', 'orange', 'orangered', 'orchid', 'palegoldenrod', 'palegreen', 'paleturquoise', 'palevioletred', 'papayawhip',
                    'peachpuff', 'peru', 'pink', 'plum', 'powderblue', 'purple', 'rebeccapurple', 'red', 'rosybrown', 'royalblue', 'saddlebrown',
                    'salmon', 'sandybrown', 'seagreen', 'seashell', 'sienna', 'silver', 'skyblue', 'slateblue', 'slategray', 'slategrey',
                    'snow', 'springgreen', 'steelblue', 'tan', 'teal', 'thistle', 'tomato', 'turquoise', 'violet', 'wheat', 'white',
                    'whitesmoke', 'yellow', 'yellowgreen'
                ]

                # Sample random named CSS colors
                return random.sample(named_css_colors, min(num_colors, len(named_css_colors)))

            def plot_detections_plotly(
                image: np.ndarray,
                detections: List[DetectionResult],
                class_colors: Optional[Dict[str, str]] = None
            ) -> None:
                # If class_colors is not provided, generate random colors for each class
                if class_colors is None:
                    num_detections = len(detections)
                    colors = random_named_css_colors(num_detections)
                    class_colors = {}
                    for i in range(num_detections):
                        class_colors[i] = colors[i]


                fig = px.imshow(image)

                # Add bounding boxes
                shapes = []
                annotations = []
                for idx, detection in enumerate(detections):
                    label = detection.label
                    box = detection.box
                    score = detection.score
                    mask = detection.mask

                    polygon = mask_to_polygon(mask)

                    fig.add_trace(go.Scatter(
                        x=[point[0] for point in polygon] + [polygon[0][0]],
                        y=[point[1] for point in polygon] + [polygon[0][1]],
                        mode='lines',
                        line=dict(color=class_colors[idx], width=2),
                        fill='toself',
                        name=f"{label}: {score:.2f}"
                    ))

                    xmin, ymin, xmax, ymax = box.xyxy
                    shape = [
                        dict(
                            type="rect",
                            xref="x", yref="y",
                            x0=xmin, y0=ymin,
                            x1=xmax, y1=ymax,
                            line=dict(color=class_colors[idx])
                        )
                    ]
                    annotation = [
                        dict(
                            x=(xmin+xmax) // 2, y=(ymin+ymax) // 2,
                            xref="x", yref="y",
                            text=f"{label}: {score:.2f}",
                        )
                    ]

                    shapes.append(shape)
                    annotations.append(annotation)

                # Update layout
                button_shapes = [dict(label="None",method="relayout",args=["shapes", []])]
                button_shapes = button_shapes + [
                    dict(label=f"Detection {idx+1}",method="relayout",args=["shapes", shape]) for idx, shape in enumerate(shapes)
                ]
                button_shapes = button_shapes + [dict(label="All", method="relayout", args=["shapes", sum(shapes, [])])]

                fig.update_layout(
                    xaxis=dict(visible=False),
                    yaxis=dict(visible=False),
                    # margin=dict(l=0, r=0, t=0, b=0),
                    showlegend=True,
                    updatemenus=[
                        dict(
                            type="buttons",
                            direction="up",
                            buttons=button_shapes
                        )
                    ],
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1
                    )
                )

                # Show plot
                fig.show()
            def mask_to_polygon(mask: np.ndarray) -> List[List[int]]:
                # Find contours in the binary mask
                contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                # Find the contour with the largest area
                largest_contour = max(contours, key=cv2.contourArea)

                # Extract the vertices of the contour
                polygon = largest_contour.reshape(-1, 2).tolist()

                return polygon

            def polygon_to_mask(polygon: List[Tuple[int, int]], image_shape: Tuple[int, int]) -> np.ndarray:
                """
                Convert a polygon to a segmentation mask.

                Args:
                - polygon (list): List of (x, y) coordinates representing the vertices of the polygon.
                - image_shape (tuple): Shape of the image (height, width) for the mask.

                Returns:
                - np.ndarray: Segmentation mask with the polygon filled.
                """
                # Create an empty mask
                mask = np.zeros(image_shape, dtype=np.uint8)

                # Convert polygon to an array of points
                pts = np.array(polygon, dtype=np.int32)

                # Fill the polygon with white color (255)
                cv2.fillPoly(mask, [pts], color=(255,))

                return mask

            def load_image(image_str: str) -> Image.Image:
                if image_str.startswith("http"):
                    image = Image.open(requests.get(image_str, stream=True).raw).convert("RGB")
                else:
                    image = Image.open(image_str).convert("RGB")

                return image

            def get_boxes(results: DetectionResult) -> List[List[List[float]]]:
                boxes = []
                for result in results:
                    xyxy = result.box.xyxy
                    boxes.append(xyxy)

                return [boxes]

            def refine_masks(masks: torch.BoolTensor, polygon_refinement: bool = False) -> List[np.ndarray]:
                masks = masks.cpu().float()
                masks = masks.permute(0, 2, 3, 1)
                masks = masks.mean(axis=-1)
                masks = (masks > 0).int()
                masks = masks.numpy().astype(np.uint8)
                masks = list(masks)

                if polygon_refinement:
                    for idx, mask in enumerate(masks):
                        shape = mask.shape
                        polygon = mask_to_polygon(mask)
                        mask = polygon_to_mask(polygon, shape)
                        masks[idx] = mask

                return masks
            
            def nms(detections: List[Dict[str, Any]], iou_threshold: float = 0.02) -> List[Dict[str, Any]]:
                """
                Apply Non-Maximum Suppression to filter overlapping boxes.
                """
                boxes = torch.tensor([[
                    d['box']['xmin'],
                    d['box']['ymin'],
                    d['box']['xmax'],
                    d['box']['ymax']
                ] for d in detections], dtype=torch.float32)

                scores = torch.tensor([d['score'] for d in detections])
                indices = torch.ops.torchvision.nms(boxes, scores, iou_threshold)
                return [detections[i] for i in indices]

            def detect(
                    image: Image.Image,
                    labels: List[str],
                    threshold: float = 0.3,
                    detector_id: Optional[str] = None
                ) -> List[Dict[str, Any]]:
                    """
                    Use Grounding DINO to detect a set of labels in an image in a zero-shot fashion.
                    """
                    device = "cuda" if torch.cuda.is_available() else "cpu"
                    detector_id = detector_id if detector_id is not None else "IDEA-Research/grounding-dino-tiny"
                    object_detector = pipeline(model=detector_id, task="zero-shot-object-detection", device=device)

                    labels = [label if label.endswith(".") else label+"." for label in labels]

                    results = object_detector(image,  candidate_labels=labels, threshold=threshold)
                    try:
                        # Áp dụng NMS để lọc kết quả
                        filtered_results = nms(results, iou_threshold=iou_threshold)
                        filtered_results = [DetectionResult.from_dict(result) for result in filtered_results]
                        # results = [DetectionResult.from_dict(result) for result in results]
                    except:
                        filtered_results = []

                    return filtered_results

            def segment(
                image: Image.Image,
                detection_results: List[Dict[str, Any]],
                polygon_refinement: bool = False,
                segmenter_id: Optional[str] = None
            ) -> List[DetectionResult]:
                """
                Use Segment Anything (SAM) to generate masks given an image + a set of bounding boxes.
                """
                device = "cuda" if torch.cuda.is_available() else "cpu"
                segmenter_id = segmenter_id if segmenter_id is not None else "facebook/sam-vit-base"

                segmentator = AutoModelForMaskGeneration.from_pretrained(segmenter_id).to(device)
                processor = AutoProcessor.from_pretrained(segmenter_id)

                try:
                    boxes = get_boxes(detection_results)
                    inputs = processor(images=image, input_boxes=boxes, return_tensors="pt").to(device)

                    outputs = segmentator(**inputs)
                    masks = processor.post_process_masks(
                        masks=outputs.pred_masks,
                        original_sizes=inputs.original_sizes,
                        reshaped_input_sizes=inputs.reshaped_input_sizes
                    )[0]

                    masks = refine_masks(masks, polygon_refinement)

                    for detection_result, mask in zip(detection_results, masks):
                        detection_result.mask = mask
                
                except:
                    detection_results = []

                return detection_results

            def grounded_segmentation(
                image: Union[Image.Image, str],
                labels: List[str],
                threshold: float = 0.3,
                polygon_refinement: bool = False,
                detector_id: Optional[str] = None,
                segmenter_id: Optional[str] = None
            ) -> Tuple[np.ndarray, List[DetectionResult]]:
                if isinstance(image, str):
                    image = load_image(image)

                detections = detect(image, labels, threshold, detector_id)
                detections = segment(image, detections, polygon_refinement, segmenter_id)

                return np.array(image), detections
            # image_url = "https://img1.oto.com.vn/2024/01/18/hai-mau-sac-moi-tren-bmw-x4-9088-94c0_wm.webp"
            _prompt = kwargs.get("prompt", "")
            labels = ["car", "truck", "bus", "motorcycle", "bicycle"]

            if _prompt and len(_prompt)>0:
                labels =  _prompt.split(",")

            threshold = iou_threshold

            detector_id = "IDEA-Research/grounding-dino-tiny"
            segmenter_id = "facebook/sam-vit-base"
            image_array, detections = grounded_segmentation(
                image=img,
                labels=labels,
                threshold=threshold,
                polygon_refinement=True,
                detector_id=detector_id,
                segmenter_id=segmenter_id
            )
            for index, _item in enumerate(detections):
                print(_item.box.xyxy)
                print(_item.label)
                print(_item.score)
                print(_item.mask)
                # bbox = _item.box.xyxy
                xyxy = _item.box.xyxy
                x_min, y_min, x_max, y_max = xyxy
                width = x_max - x_min
                height = y_max - y_min
                xywh = [x_min, y_min, width, height]

                if _item.score and _item.score < float(confidence_threshold):
                    break
                # points = []
                # for i, item in enumerate(_item.mask):
                polygon, _ = self.mask_to_polygons(_item.mask,img_width,img_height)
                # points.append(polygon)
                # Tạo đối tượng phát hiện
                results.append({
                    "type": 'polygonlabels',
                    "value": {
                        "points": polygon,
                        "polygonlabels": [
                            _item.label.rstrip('.').capitalize()
                        ]
                    },
                    "original_width": img_width,
                    "original_height": img_height,
                    "score": _item.score,
                    "image_rotation": 0,
                    "xywh": xywh
                    # "id": int(time.time()*1000)
                })
                # break

            return {"message": "predict completed successfully",
                        "result": results}
        
    def create_gradio_UI(self, cfg, **kwargs):
        import gradio as gr
        from PIL import Image
        import tqdm
        # from predictor import VisualizationDemo
        import tempfile
        WINDOW_NAME = "AIxBlock Segment Anything"
        # predictor = DefaultPredictor(cfg)
        # demo = VisualizationDemo(cfg)

        model_path = kwargs.get('model_path')

        def test_opencv_video_format(codec, file_ext):
            with tempfile.TemporaryDirectory(prefix="video_format_test") as dir:
                filename = os.path.join(dir, "test_file" + file_ext)
                writer = cv2.VideoWriter(
                    filename=filename,
                    fourcc=cv2.VideoWriter_fourcc(*codec),
                    fps=float(30),
                    frameSize=(10, 10),
                    isColor=True,
                )
                [writer.write(np.zeros((10, 10, 3), np.uint8))
                 for _ in range(30)]
                writer.release()
                if os.path.isfile(filename):
                    return True
                return False

        def numpy_to_base64(np_array, format="JPEG"):
            from io import BytesIO
            """
            Chuyển đổi một mảng NumPy (hình ảnh) sang chuỗi base64.
            
            Parameters:
            - np_array: Mảng NumPy chứa dữ liệu hình ảnh.
            - format: Định dạng tệp hình ảnh (ví dụ: JPEG, PNG).
            
            Returns:
            - Chuỗi base64 đại diện cho hình ảnh.
            """
            # Chuyển đổi mảng NumPy thành đối tượng hình ảnh PIL
            image = Image.fromarray(np_array.astype("uint8"))
            
            # Lưu hình ảnh vào một buffer bằng BytesIO
            buffered = BytesIO()
            image.save(buffered, format=format)
            
            # Mã hóa nội dung buffer thành base64
            img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
            
            return img_base64
        
        def detectron_2(input_img, size=image_size):
            img_64 = numpy_to_base64(input_img)
            result = self.action(1, "predict", "", data={"image": img_64})
            if result['result']:
                for res in result['result']:  
                    original_width = res['original_width']
                    original_height = res['original_height']
                    points = np.array(res['value']['points'], np.int32)  # Chuyển các điểm về dạng numpy array
                    points = points * [original_width / 100, original_height / 100]
                    label = res['value']['polygonlabels'][0] 

                    score = res['score']  
                    points = points.astype(np.int32)

                    input_img = cv2.fillPoly(input_img, [points], color=(0, 255, 0))
                    input_img = cv2.polylines(input_img, [points], isClosed=True, color=(0, 255, 0), thickness=2)

                    input_img = cv2.putText(input_img, label, (points[0][0], points[0][1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                    # score_text = f"Score: {score:.2f}"
                    # input_img = cv2.putText(input_img, score_text, (points[0][0], points[0][1] + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                    

                    x_left_pct, y_top_pct, width_pct, height_pct = res['xywh']

                    # Tính toán lại tọa độ và kích thước bounding box trong ảnh gốc
                    x_left = x_left_pct #(x_left_pct / 100) * original_width
                    y_top = y_top_pct #(y_top_pct / 100) * original_height
                    width = width_pct #(width_pct / 100) * original_width
                    height = height_pct #(height_pct / 100) * original_height

                    # Chuyển đổi tọa độ thành kiểu int nếu cần
                    x_left, y_top, x_right, y_bottom = int(x_left), int(y_top), int(x_left + width), int(y_top + height)

                    # Vẽ hình chữ nhật lên ảnh
                    input_img = cv2.rectangle(input_img, (x_left, y_top), (x_right, y_bottom), color=(255, 0, 0), thickness=2)
                    input_img = cv2.putText(input_img, label, (x_left, y_top), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    score_text = f"Score: {score:.2f}"
                    input_img = cv2.putText(input_img, score_text, (x_left, y_top + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

            return input_img
            # outputs = predictor(img)

            # v = Visualizer(
            #     im, MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1)
            # out = v.draw_instance_predictions(outputs["instances"].to('cpu'))
            # return Image.fromarray(np.uint8(out.get_image())).convert('RGB')

        def video_process(video_input, file_name=f"video_output_{time.time()}.mp4"):
            # predict on video
            video_output = os.path.join(
                os.getcwd(), 'outputs', file_name)
            if not os.path.exists(os.path.dirname(video_output)):
                os.makedirs(os.path.dirname(video_output))
            video = cv2.VideoCapture(video_input)
            width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
            frames_per_second = video.get(cv2.CAP_PROP_FPS)
            num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
            basename = os.path.basename(video_input)
            codec, file_ext = (
                ("x264", ".mkv") if test_opencv_video_format(
                    "x264", ".mkv") else ("mp4v", ".mp4")
            )

            print(codec, file_ext)

            if codec == ".mp4v":
                print("x264 codec not available, switching to mp4v")
            output_file = cv2.VideoWriter(
                filename=video_output,
                # some installation of opencv may not support x264 (due to its license),
                # you can try other format (e.g. MPEG)
                fourcc=cv2.VideoWriter_fourcc(*codec),
                fps=float(frames_per_second),
                frameSize=(width, height),
                isColor=True,
            )
            for vis_frame in tqdm.tqdm(demo.run_on_video(video), total=num_frames):
                if video_output:
                    output_file.write(vis_frame)
                else:
                    cv2.namedWindow(basename, cv2.WINDOW_NORMAL)
                    cv2.imshow(basename, vis_frame)
                    if cv2.waitKey(1) == 27:
                        break  # esc to quit
            video.release()
            output_file.release()
            return video_output

        def webcam_process(video_input):
            return video_process(video_input, file_name=f"webcam_output_{time.time()}.mp4")

        def live_webcam_process(im):
            return detectron_2(im)

        def download(option):
            if option == "pytorch":
                return os.path.join(model_path, 'model_final.pth')
            else:
                return MODEL_FILE

        image_interface = gr.Interface(
            fn=detectron_2,
            inputs=[gr.Image(type="pil", label="Input Image")],
            outputs=[gr.Image(type="pil", label="Output Image")],
            title="AIxBlock - Detectron2 Image",
            description="Upload an image or click an example image to use.",
            theme="huggingface",
        )

        def capture_and_process_image(image):
            # Ảnh từ webcam sẽ được truyền vào đây dưới dạng PIL Image
            # Chúng ta chỉ cần trả về ảnh này nếu không có xử lý gì thêm
            return image

        webcam_interface = gr.Interface(
            fn=detectron_2,  # Hàm này nhận và trả về ảnh giống như image_interface
            inputs=[gr.Image(sources="webcam", type="pil", label="Capture Image from Webcam")],  # Capture ảnh từ webcam
            outputs=[gr.Image(type="pil", label="Captured Image")],  # Trả về ảnh giống format
            title="AIxBlock - Webcam Capture",
            description="Capture an image from the webcam and return it.",
            theme="huggingface",
        )

        video_interface = gr.Interface(
            fn=video_process,
            inputs=[gr.Video(label="Input Video", format="mp4")],
            outputs=[gr.Video(label="Output Video", format="mp4")],
            title="AIxBlock - Detectron2 Video",
            description="Upload a video or click an example video to use.",
            theme="huggingface",
        )
        # live_webcam_interface = gr.Interface(
        #     fn=live_webcam_process,
        #     inputs=[gr.Image(type="pil", label="Input Webcam",
        #                      sources="webcam", streaming=True)],
        #     outputs=[gr.Image(type="pil", label="Output Webcam")],
        #     live=True,
        #     title="AIxBlock - Detectron2 Webcam",
        #     description="Real-time object detection with webcam.",
        #     theme="huggingface",
        # )
        
        live_webcam_interface = gr.Interface(
            fn=live_webcam_process,  # Hàm xử lý video
            inputs=[WebRTC(label="Stream", rtc_configuration=rtc_configuration)],  # Webcam stream input
            # outputs=[gr.Image(type="pil", label="Output Webcam")],  # Output sau khi xử lý
            outputs=[gr.Image(type="pil", label="Output Webcam")],  # Output sau khi xử lý
            title="AIxBlock - Detectron2 Video",
            description="Upload a video or click an example video to use.",
            theme="huggingface",
        )

        # if model_path:
        #     model_type_choices = ['pytorch', 'baseline']  # onnx, tfjs
        # else:
        #     model_type_choices = ['baseline']

        # download models
        # download_interface = gr.Interface(
        #     fn=download,
        #     inputs=[gr.inputs.Dropdown(
        #         choices=model_type_choices, label="Model type")],
        #     outputs=[
        #         gr.File(label="Download Model", type="file")
        #     ],
        #     title="AIxBlock - Download",
        #     description="Download models.",
        #     theme="huggingface",
        # )

        # gradio_app, local_url, share_url = gr.TabbedInterface(
        #     [image_interface, webcam_interface, live_webcam_interface],
        #     # [image_interface, video_interface,
        #     #     webcam_interface, live_webcam_interface, download_interface],
        #     # tab_names=['Image', 'Video', 'Webcam',
        #     #            'Live Webcam', 'Download Model'],
        #     tab_names=['Image', 'Webcam', 'Live Webcam'],
        # ).launch(share=True, quiet=True, prevent_thread_lock=True, server_name='0.0.0.0')

        def process_video(video_path):
                # Mở video tải lên từ đường dẫn
                cap = cv2.VideoCapture(video_path)

                # Kiểm tra nếu video được mở thành công
                if not cap.isOpened():
                    return "Error: Could not open video"

                # Lấy thông tin về video
                fps = cap.get(cv2.CAP_PROP_FPS)
                frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

                # Tạo đối tượng để ghi video đầu ra
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter('processed_video.mp4', fourcc, fps, (frame_width, frame_height))

                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break  # Dừng nếu không còn frame nào

                    # Xử lý frame
                    processed_frame = detectron_2(frame)

                    processed_frame = np.array(processed_frame)

                    # Đảm bảo kích thước của processed_frame khớp với kích thước video
                    processed_frame = cv2.resize(processed_frame, (frame_width, frame_height))

                    # Viết frame đã xử lý vào video output
                    out.write(processed_frame)

                # Giải phóng các tài nguyên
                cap.release()
                out.release()

                return 'processed_video.mp4'
        
        with gr.Blocks(css=css) as demo2:
            with gr.Row():
                with gr.Column(scale=10):
                    gr.Markdown(
                        """
                        # Theme preview: `AIxBlock`
                        """
                    )

            def clear():
                """Hàm xóa đầu vào và đầu ra."""
                return None, None
            
            with gr.Tabs(elem_classes=["feedback"]) as parent_tabs:
                with gr.TabItem("Image", id=0):
                    with gr.Tabs() as demo_sub_tabs:
                            # Tab con cho Upload Image
                            with gr.TabItem("Upload Image"):
                                with gr.Row():
                                    gr.Markdown("## Input", elem_classes=["title1"])
                                    gr.Markdown("## Output", elem_classes=["title1"])

                                gr.Interface(detectron_2,
                                                gr.Image(elem_classes=["upload_image"], sources="upload", container=False, height=345,
                                                        show_label=False),
                                                gr.Image(elem_classes=["upload_image"], container=False, height=345, show_label=False),
                                                # allow_flagging=False
                                                )

                            with gr.TabItem("Webcam Capture"):
                                with gr.Row():
                                    gr.Markdown("## Input", elem_classes=["title1"])
                                    gr.Markdown("## Output", elem_classes=["title1"])

                                # Row for Input and Output columns
                                with gr.Row():
                                    # Column for Input (Webcam and Buttons)
                                    with gr.Column():
                                        # Webcam stream input
                                        webcam_feed = gr.Image(
                                            sources="webcam",
                                            elem_classes=["upload_image"],
                                            container=False,
                                            height=345,
                                            show_label=False
                                        )

                                        # Buttons under the Input column
                                        with gr.Row():
                                            clear_button = gr.Button("Clear")
                                            submit_button = gr.Button("Submit", elem_classes=["primary"])


                                    # Column for Output
                                    with gr.Column():
                                        # Processed Output
                                        processed_output = gr.Image(
                                            elem_classes=["upload_image"],
                                            container=False,
                                            height=345,
                                            show_label=False
                                        )

                                # Button functionality
                                submit_button.click(
                                    fn=detectron_2,  # Gọi hàm xử lý
                                    inputs=webcam_feed,  # Đầu vào là webcam
                                    outputs=processed_output  # Hiển thị kết quả
                                )

                                clear_button.click(
                                    fn=clear,  # Gọi hàm xóa
                                    inputs=None,  # Không cần đầu vào
                                    outputs=[webcam_feed, processed_output]  # Xóa cả webcam feed và kết quả
                                )

                with gr.TabItem("Video", id=1):
                    # gr.Image(elem_classes=["upload_image"], sources="clipboard", height=345, container=False,
                    #           show_label=False)
                    with gr.Tabs() as demo_sub_tabs:
                        with gr.TabItem("Live cam"):
                            with gr.Column(elem_classes=["my-column"]):
                                with gr.Group(elem_classes=["my-group"]):
                                    image = WebRTC(label="Stream", rtc_configuration=rtc_configuration)

                                image.stream(
                                    fn=live_webcam_process, inputs=[image], outputs=[image], time_limit=10
                                )
                        with gr.TabItem("Upload Video"):
                            with gr.Row():
                                gr.Markdown("## Input", elem_classes=["title1"])
                                gr.Markdown("## Output", elem_classes=["title1"])

                            with gr.Row():
                                # Cột đầu tiên cho video input
                                with gr.Column():
                                    video_input = gr.Video(label="Upload Video", format="mp4")
                                    submit_button = gr.Button("Process Video")

                                # Cột thứ hai cho video output và nút submit
                                with gr.Column():
                                    video_output = gr.Video(label="Processed Video")
                                
                                submit_button.click(
                                        fn=process_video, inputs=[video_input], outputs=[video_output]
                                    )

        gradio_app, local_url, share_url = demo2.launch(share=True, quiet=True, prevent_thread_lock=True, server_name='0.0.0.0',show_error=True)

        print(f"Gradio app: {gradio_app}")
        print(f"Local URL: {local_url}")
        print(f"Public URL: {share_url}")

        return gradio_app, local_url, share_url

    def preview(self, project):

        print(f"start launch gradio app for project {project}")
        # gradio.app
        # 3 main UI: file upload, link, video (webcam)
        # to predict
        if not self.predictor:
            cfg, model_path = self.get_latest_model(project, default=False)
        # set threshold
        # cfg.MODEL.RETINANET.SCORE_THRESH_TEST = SCORE_THRESHOLD
        # cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = SCORE_THRESHOLD
        print(f"model path: {model_path}")

        _, local_url, share_url = self.create_gradio_UI(cfg=cfg)
        return {"share_url": share_url, 'local_url': local_url}
    
    def model(self, project, **kwargs):
        if not os.path.exists(os.path.join('app', 'segment_anything')):
            os.makedirs(os.path.join('app', 'segment_anything'))
        if not os.path.exists(os.path.join('app', 'segment_anything', '{project}')):
            os.makedirs(os.path.join(
                'app', 'segment_anything', '{project}'))
        if not os.path.exists(os.path.join('app', 'segment_anything', '{project}', 'training')):
            os.makedirs(os.path.join(
                'app', 'segment_anything', '{project}', 'training'))
        if not self.predictor:
            self.get_latest_model(project, True)

        if self.model_path:
            # read and parse datetime from success.txt
            with open(os.path.join(self.model_path, 'success.txt'), 'r') as f:
                model_build_date = f.read()

            # open metrics.json file
            metrics = []
            with open(os.path.join(self.model_path, 'metrics.json'), 'r') as f:
                # read each line and append to list as dict
                for line in f:
                    metrics.append(json.loads(line))
        else:
            model_build_date = None
            metrics = []

        f = kwargs.get('f', 'preview')  # format
        share_url, model_url = '', ''
        if f == 'onnx':
            pass
        elif f == 'preview':
            _, local_url, share_url = self.create_gradio_UI(
                None, model_path=self.model_path)
        else:
            model_url = f"{HOSTNAME}/download?project={project}"

        response = {
            'model_build_date': model_build_date,
            # 'model_url': model_url,
            'share_url': share_url,
            'metrics': metrics,
            'model_type': 'preview',  # ONNX,TorchScript,PyTorch,TFJS, ...
            'model_version': '1.0.0'
        }
        return response
    
    def toolbar_predict(self, image, clickpoint, polygons, vertices, project, **kwargs):
        print("toolbar_predict")
        pass
    
    def yolov8_detection(self,image):
        from ultralytics import YOLO
        model=YOLO('yolov8m.pt')
        # image = cv2.imread(image_path)
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = model(image, stream=True)  # generator of Results objects
        bbox=[]
        names = model.names
        for result in results:
            boxes = result.boxes  # Boxes object for bbox outputs
            # masks = result.masks  # Masks object for segmentation masks outputs
            # probs = result.probs  # Class probabilities for classification outputs
            # for c in boxes.cls:
            #     print(names[int(c)])
            print("Object type:", boxes.cls)
            print("Coordinates:", boxes.xyxy)
            print("Probability:", boxes.conf)
        if len(boxes.xyxy.tolist())>0:
            bbox=boxes.xyxy.tolist()[0]
        return bbox,image,boxes.cls
    
    def toolbar_predict_sam(self, image):
        # return polygonlabels list
        # image: base64 image
        # voice: base64 audio
        # prompt: text
        # draw polygons: [[x1, y1, x2, y2, x3, y3, x4, y4], ...]
        # clickpoint: [x, y]
        # polygons: [[x1, y1, x2, y2, x3, y3, x4, y4], ...]
        # vertices: horizontal or vertical
        # return all predicted labels in selected region
        print("toolbar_predict_sam")
        
        import base64

        # if not self.predictor:
        #     self.get_latest_model(
        #         project, default=DEBUG)

        print(f"load model from {self.model_path}")

        results = []

        try:
            # print(image)
            # read base64 image to cv2
            image = image.replace('data:image/png;base64,', '')
            image = image.replace('data:image/jpeg;base64,', '')
            img = Image.open(io.BytesIO(
                base64.b64decode(image))).convert('RGB')
            img_width, img_height = img.size  # PIL, shape is for Tensor
            print(f"image size: {img_width}x{img_height}")
            image = img
            self.hostname = HOSTNAME
            self.access_token = API_KEY
            prompts = prompt.split(",")
           
            # print("==========bounding box====================")
           
            # w, h = (box.xyxy[2]-box.xyxy[0]), (box.xyxy[3]-box.xyxy[1])
            # centerX = (box.xyxy[0] + w / 2) / img_width
            # centerY = (box.xyxy[1] + h / 2) / img_height
            # width = w / img_width
            # height = h / img_height
            # for i, (box) in enumerate(zip(yolov8_boxex)):
            #     print(box)
            #     results.append({
            #             'classification':  box.cls,
            #             'centerX':centerX,
            #             'centerY':centerY,
            #             'width': width,
            #             'height': height,
            #             'regionType': 'bounding-box',
            #             "score":box.conf,
            #             "original_width": img_width,
            #             "original_height": img_height,
            #             'type': 'rectanglelabels',
            #             'value': {
            #                 'rectanglelabels': [
            #                      box.cls],
            #                 'x': box.xyxy[0] / img_width * 100,
            #                 'y': box.xyxy[1] / img_height * 100,
            #                 'width': (box.xyxy[2] - box.xyxy[0] ) / img_width * 100,
            #                 'height': (box.xyxy[3] - box.xyxy[1]) / img_height * 100,
            #             }
            #         })
                    
            # print("==========")

            # convert img to np array
            import numpy as np
            _img = np.array(img)
            # input_point = np.array([[]])
            # input_label = np.array(classes)
            input_box = np.array([0, 0, img_width, img_height])
            
            from groundingdino.util.inference import Model
            from groundingdino.config import GroundingDINO_SwinT_OGC

            GROUNDING_DINO_CHECKPOINT_PATH = os.path.join("groundingdino_swint_ogc.pth")
            GROUNDING_DINO_CONFIG_PATH = GroundingDINO_SwinT_OGC

            grounding_dino_model = Model(model_config_path=GROUNDING_DINO_CONFIG_PATH, model_checkpoint_path=GROUNDING_DINO_CHECKPOINT_PATH)

            from segment_anything_2 import sam_model_registry, SamPredictor
            sam = sam_model_registry["vit_h"](checkpoint="./sam_vit_h_4b8939.pth")
            sam_predictor = SamPredictor(sam)

            prompts = prompt.split(",")
            CLASSES =  prompts #['car', 'dog', 'person', 'nose', 'chair', 'shoe', 'ear']
            if len(prompts)==0:
                 CLASSES = self.labels_in_config
            BOX_TRESHOLD = 0.35
            TEXT_TRESHOLD = 0.25

            from typing import List

            def enhance_class_name(class_names: List[str]) -> List[str]:
                return [
                    f"all {class_name}s"
                    for class_name
                    in class_names
                ]
            import cv2
            import wow_ai_cv as sv
            import numpy as np

            # load image
            image = _img

            print(f"CLASSES:{CLASSES}")
            # detect objects
            detections = grounding_dino_model.predict_with_classes(
                image=image,
                classes=enhance_class_name(class_names=CLASSES),
                box_threshold=BOX_TRESHOLD,
                text_threshold=TEXT_TRESHOLD
            )

            def segment(sam_predictor: SamPredictor, image: np.ndarray, xyxy: np.ndarray) -> np.ndarray:
                sam_predictor.set_image(image)
                result_masks = []
                polygons_array=[]
                for box in xyxy:
                    masks, scores, logits = sam_predictor.predict(
                        box=box,
                        multimask_output=True
                    )
                    index = np.argmax(scores)
                    result_masks.append(masks[index])
                    polygons_array.append(sv.mask_to_polygons(masks[index]))
                    # print(sv.mask_to_polygons(masks[index]))
                return np.array(result_masks),polygons_array
            
            detections.mask,polygons_array = segment(
                sam_predictor=sam_predictor,
                image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
                xyxy=detections.xyxy
            )

            labels = [
                f"{CLASSES[class_id]}"
                for _, _, confidence, class_id, _
                in detections]
            scores = [
                f"{confidence:0.2f}"
                for _, _, confidence, class_id, _
                in detections]
            
            for j, (label,polygon,score) in enumerate(zip(labels,polygons_array,scores)):
                print(f"label:{label}")
                print(f"score:{score}")
                _points=[]
                for j, item in enumerate(polygon):
                    x, y = zip(*item)
                    x = [l/img_width*100 for l in x]
                    y = [l/img_height*100 for l in y]
                    res = list(zip(x, y))
                    _points.append(res)
                _points = json.dumps(_points[0], indent=2)
                _points = _points.replace("(","[")
                _points = _points.replace(")","]")
                
                results.append({
                    "type": 'polygonlabels',
                    "original_width": img_width,
                    "original_height": img_height,
                    "image_rotation": 0,
                    "value": {
                        "polygonlabels": [label],
                        "points": json.loads(_points)
                    },
                    "score": score
                })
           
        
            #     # if "RectangleLabels" in self.labels_in_config:
                   
            #         results.append({
            #             'classification': "car",
            #             'centerX': w/2,
            #             'centerY': h/2,
            #             'width': img_width,
            #             'height': img_height,
            #             'regionType': 'bounding-box',
            #             "score": 0.0,
            #             "original_width": img_width,
            #             "original_height": img_height,
            #             'type': 'rectanglelabels',
            #             'value': {
            #                 'rectanglelabels': [
            #                     "car"],
            #                 'x': x / img_width * 100,
            #                 'y': y / img_height * 100,
            #                 'width': w / img_width * 100,
            #                 'height': h / img_height * 100
            #             }
            #         })
                # elif  "PolygonLabels" in self.labels_in_config:
                #     points, has_hole = self.mask_to_polygons(
                #         mask=mask, max_width=img_width, max_height=img_height)
                #     if len(points)>0:
                #         results.append({
                #             # "from_name": self.from_name,
                #             # "to_name": self.to_name,
                #             "type": 'polygonlabels',
                #             "original_width": img_width,
                #             "original_height": img_height,
                #             "image_rotation": 0,
                #             "value": {
                #                 "polygonlabels": [output_label],
                #                 "points": points,
                #             },
                #             "id": i,
                #             "score": 0
                #         })
            # print(results)
            return results
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"Toobar predict error: {e}")
            return []

    def model_trial(self, project, **kwargs):
        import gradio as gr 


        css = """
        .feedback .tab-nav {
            justify-content: center;
        }

        .feedback button.selected{
            background-color:rgb(115,0,254); !important;
            color: #ffff !important;
        }

        .feedback button{
            font-size: 16px !important;
            color: black !important;
            border-radius: 12px !important;
            display: block !important;
            margin-right: 17px !important;
            border: 1px solid var(--border-color-primary);
        }

        .feedback div {
            border: none !important;
            justify-content: center;
            margin-bottom: 5px;
        }

        .feedback .panel{
            background: none !important;
        }


        .feedback .unpadded_box{
            border-style: groove !important;
            width: 500px;
            height: 345px;
            margin: auto;
        }

        .feedback .secondary{
            background: rgb(225,0,170);
            color: #ffff !important;
        }

        .feedback .primary{
            background: rgb(115,0,254);
            color: #ffff !important;
        }

        .upload_image button{
            border: 1px var(--border-color-primary) !important;
        }
        .upload_image {
            align-items: center !important;
            justify-content: center !important;
            border-style: dashed !important;
            width: 500px;
            height: 345px;
            padding: 10px 10px 10px 10px
        }
        .upload_image .wrap{
            align-items: center !important;
            justify-content: center !important;
            border-style: dashed !important;
            width: 500px;
            height: 345px;
            padding: 10px 10px 10px 10px
        }

        .webcam_style .wrap{
            border: none !important;
            align-items: center !important;
            justify-content: center !important;
            height: 345px;
        }

        .webcam_style .feedback button{
            border: none !important;
            height: 345px;
        }

        .webcam_style .unpadded_box {
            all: unset !important;
        }

        .btn-custom {
            background: rgb(0,0,0) !important;
            color: #ffff !important;
            width: 200px;
        }

        .title1 {
            margin-right: 90px !important;
        }

        .title1 block{
            margin-right: 90px !important;
        }

        """

        with gr.Blocks(css=css) as demo:
            with gr.Row():
                with gr.Column(scale=10):
                    gr.Markdown(
                        """
                        # Theme preview: `AIxBlock`
                        """
                    )

            import numpy as np
            def predict(input_img):
                import cv2
                result = self.action(project, "predict",collection="",data={"img":input_img})
                print(result)
                if result['result']:
                    # boxes = result['result']['boxes']
                    # names = result['result']['names']
                    # labels = result['result']['labels']

                    for res in result['result']:
                        original_width = res['original_width']
                        original_height = res['original_height']
                        points = np.array(res['value']['points'], np.int32)  # Chuyển các điểm về dạng numpy array
                        points = points * [original_width / 100, original_height / 100]
                        label = res['value']['polygonlabels'][0] 

                        score = res['score']  
                        points = points.astype(np.int32)

                        input_img = cv2.fillPoly(input_img, [points], color=(0, 255, 0))
                        input_img = cv2.polylines(input_img, [points], isClosed=True, color=(0, 255, 0), thickness=2)

                        input_img = cv2.putText(input_img, label, (points[0][0], points[0][1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                        score_text = f"Score: {score:.2f}"
                        input_img = cv2.putText(input_img, score_text, (points[0][0], points[0][1] + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

                    
                    # for box, label in zip(boxes, labels):
                    #     box = [int(i) for i in box]
                    #     label = int(label)
                    #     input_img = cv2.rectangle(input_img, box, color=(255, 0, 0), thickness=2)
                    #     # input_img = cv2.(input_img, names[label], (box[0], box[1]), color=(255, 0, 0), size=1)
                    #     input_img = cv2.putText(input_img, names[label], (box[0], box[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        
                return input_img
            
            def download_btn(evt: gr.SelectData):
                print(f"Downloading {dataset_choosen}")
                return f'<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/4.7.0/css/font-awesome.min.css"><a href="/my_ml_backend/datasets/{evt.value}" style="font-size:50px"> <i class="fa fa-download"></i> Download this dataset</a>'
                
            def trial_training(dataset_choosen):
                print(f"Training with {dataset_choosen}")
                result = self.action(project, "train",collection="",data=dataset_choosen)
                return result['message']

            def get_checkpoint_list(project):
                print("GETTING CHECKPOINT LIST")
                print(f"Proejct: {project}")
                import os
                checkpoint_list = [i for i in os.listdir("my_ml_backend/models") if i.endswith(".pt")]
                checkpoint_list = [f"<a href='./my_ml_backend/checkpoints/{i}' download>{i}</a>" for i in checkpoint_list]
                if os.path.exists(f"my_ml_backend/{project}"):
                    for folder in os.listdir(f"my_ml_backend/{project}"):
                        if "train" in folder:
                            project_checkpoint_list = [i for i in os.listdir(f"my_ml_backend/{project}/{folder}/weights") if i.endswith(".pt")]
                            project_checkpoint_list = [f"<a href='./my_ml_backend/{project}/{folder}/weights/{i}' download>{folder}-{i}</a>" for i in project_checkpoint_list]
                            checkpoint_list.extend(project_checkpoint_list)
                
                return "<br>".join(checkpoint_list)

            def tab_changed(tab):
                if tab == "Download":
                    get_checkpoint_list(project=project)
            
            def upload_file(file):
                return "File uploaded!"
            
            with gr.Tabs(elem_classes=["feedback"]) as parent_tabs:
                with gr.TabItem("Image", id=0):   
                    with gr.Row():
                        gr.Markdown("## Input", elem_classes=["title1"])
                        gr.Markdown("## Output", elem_classes=["title1"])
                    
                    gr.Interface(predict, gr.Image(elem_classes=["upload_image"], sources="upload", container = False, height = 345,show_label = False), 
                                gr.Image(elem_classes=["upload_image"],container = False, height = 345,show_label = False), allow_flagging = False             
                    )


                # with gr.TabItem("Webcam", id=1):    
                #     gr.Image(elem_classes=["webcam_style"], sources="webcam", container = False, show_label = False, height = 450)

                # with gr.TabItem("Video", id=2):    
                #     gr.Image(elem_classes=["upload_image"], sources="clipboard", height = 345,container = False, show_label = False)

                # with gr.TabItem("About", id=3):  
                #     gr.Label("About Page")

                with gr.TabItem("Trial Train", id=2):
                    gr.Markdown("# Trial Train")
                    with gr.Column():
                        with gr.Column():
                            gr.Markdown("## Dataset template to prepare your own and initiate training")
                            with gr.Row():
                                #get all filename in datasets folder
                                datasets = [(f"dataset{i}", name) for i, name in enumerate(os.listdir('./my_ml_backend/datasets'))]
                                
                                dataset_choosen = gr.Dropdown(datasets, label="Choose dataset", show_label=False, interactive=True, type="value")
                                # gr.Button("Download this dataset", variant="primary").click(download_btn, dataset_choosen, gr.HTML())
                                download_link = gr.HTML("""
                                        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/4.7.0/css/font-awesome.min.css">
                                        <a href='' style="font-size:24px"><i class="fa fa-download" ></i> Download this dataset</a>""")
                                
                                dataset_choosen.select(download_btn, None, download_link)
                                
                                #when the button is clicked, download the dataset from dropdown
                                # download_btn
                            gr.Markdown("## Upload your sample dataset to have a trial training")
                            # gr.File(file_types=['tar','zip'])
                            gr.Interface(predict, gr.File(elem_classes=["upload_image"],file_types=['tar','zip']), 
                                gr.Label(elem_classes=["upload_image"],container = False), allow_flagging = False             
                    )
                            with gr.Row():
                                gr.Markdown(f"## You can attemp up to {2} FLOps")
                                gr.Button("Trial Train", variant="primary").click(trial_training, dataset_choosen, None)
                
                # with gr.TabItem("Download"):
                #     with gr.Column():
                #         gr.Markdown("## Download")
                #         with gr.Column():
                #             gr.HTML(get_checkpoint_list(project))

        gradio_app, local_url, share_url = demo.launch(share=True, quiet=True, prevent_thread_lock=True, server_name='0.0.0.0',show_error=True)
   
        return {"share_url": share_url, 'local_url': local_url}
    
    def download(self, project, **kwargs):
        return super().download(project, **kwargs)
