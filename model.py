import base64
import gc
import io
import os
import subprocess
import sys
import threading
import time
import uuid
from typing import Iterator

import gradio as gr
import numpy as np
import supervision as sv
import torch
from aixblock_ml.model import AIxBlockMLBase
from loguru import logger
from mcp.server.fastmcp import FastMCP
from PIL import Image

from logging_class import start_queue, write_log
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from sam2.build_sam import build_sam2

# ------------------------------------------------------------------


HOST_NAME = os.environ.get("HOST_NAME", "https://dev-us-west-1.aixblock.io")
TYPE_ENV = os.environ.get("TYPE_ENV", "DETECTION")


mcp = FastMCP("aixblock-mcp")

CHANNEL_STATUS = {}

sam2_base = None
mask_generator_base = None


class MyModel(AIxBlockMLBase):

    @mcp.tool()
    def action(self, command, **kwargs):
        print(
            f"""
                command: {command}
            """
        )
        if command.lower() == "execute":
            _command = kwargs.get("shell", None)
            logger.info(f"Executing command: {_command}")
            subprocess.Popen(
                _command,
                shell=True,
                stdout=sys.stdout,
                stderr=sys.stderr,
                text=True,
            )
            return {"message": "command completed successfully"}
        # region Train
        elif command.lower() == "train":
            model_id = kwargs.get("model_id", "segment-anything-2.1")
            roboflow_key = kwargs.get("roboflow_key", "z6IkNNY2MUovMbOOeYaZ")
            roboflow_worksapce = kwargs.get("roboflow_worksapce", "brad-dwyer")
            roboflow_project = kwargs.get("roboflow_project", "car-parts-pgo19")
            roboflow_version = kwargs.get("roboflow_version", 6)
            roboflow_dataset = kwargs.get("roboflow_dataset", "sam2")
            num_epochs = kwargs.get("num_epochs", 1)
            batch_size = kwargs.get("batch_size", 1)
            yaml_path = kwargs.get(
                "yaml_path",
                "./sam2/configs/train_example.yaml",
            )
            model = kwargs.get("model", "sam2.1_hiera_base_plus.pt")
            hf_token = kwargs.get(
                "hf_token",
                "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",
            )

            project_id = kwargs.get("project_id", 0)
            channel_log = kwargs.get("channel_log", "training_logs")
            world_size = kwargs.get("world_size", 1)
            rank = kwargs.get("rank", 0)
            master_add = kwargs.get("master_add", "127.0.0.1")
            master_port = kwargs.get("master_port", "23456")
            host_name = kwargs.get("host_name", HOST_NAME)
            framework = kwargs.get("framework", "huggingface")

            log_queue, logging_thread = start_queue(channel_log)
            write_log(log_queue)

            hf_model_id = kwargs.get("hf_model_id", "segment_anything")
            channel_name = f"{hf_model_id}_{str(uuid.uuid4())[:8]}"

            CHANNEL_STATUS[channel_name] = {
                "status": "training",
                "hf_model_id": hf_model_id,
                "command": command,
                "created_at": time.time(),
            }
            print(f"🚀 Đã bắt đầu training kênh: {channel_name}")
            subprocess.Popen(
                f"venv/bin/python train.py --roboflow_key {roboflow_key} --roboflow_worksapce {roboflow_worksapce} --roboflow_project {roboflow_project} --roboflow_version {roboflow_version} --roboflow_dataset {roboflow_dataset} --num_epochs 2 --hf_model_id {hf_model_id} --model {model} --hf_token {hf_token}",
                stdout=sys.stdout,
                stderr=sys.stderr,
                text=True,
                shell=True,
            )
            return {
                "message": "train completed successfully",
                "channel_name": channel_name,
            }
        # region Predict
        elif command.lower() == "predict":
            image_64 = kwargs.get("image", None)
            if not image_64:
                image_64 = kwargs.get("data", {}).get("image")
            model = kwargs.get("model", "sam2.1_hiera_base_plus.pt")
            config_dict = {
                "sam2.1_hiera_base_plus.pt": "configs/sam2.1/sam2.1_hiera_b+.yaml",
                "sam2.1_hiera_tiny.pt": "configs/sam2.1/sam2.1_hiera_t.yaml",
                "sam2.1_hiera_small.pt": "configs/sam2.1/sam2.1_hiera_s.yaml",
                "sam2.1_hiera_large.pt": "configs/sam2.1/sam2.1_hiera_l.yaml",
            }
            config = config_dict.get(model, "configs/sam2.1/sam2.1_hiera_b+.yaml")
            if image_64.startswith("data:image"):
                image_64 = image_64.split(",")[1]
            image_bytes = base64.b64decode(image_64)
            image = Image.open(io.BytesIO(image_bytes))
            # load model
            try:
                torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
                if torch.cuda.get_device_properties(0).major >= 8:
                    torch.backends.cuda.matmul.allow_tf32 = True
                    torch.backends.cudnn.allow_tf32 = True
            except Exception as e:
                print(f"Torch setup warning: {e}")
            sam2_base = build_sam2(config, f"./{model}", device="cuda")
            mask_generator_base = SAM2AutomaticMaskGenerator(sam2_base)
            # process image
            if isinstance(image, Image.Image):
                opened_image = np.array(image.convert("RGB"))
            else:
                opened_image = np.array(Image.fromarray(image).convert("RGB"))
            base_result = mask_generator_base.generate(opened_image)
            base_detections = sv.Detections.from_sam(sam_result=base_result)
            base_annotator = sv.MaskAnnotator(color_lookup=sv.ColorLookup.INDEX)
            base_annotated_image = opened_image.copy()
            base_annotated_image = base_annotator.annotate(
                base_annotated_image, detections=base_detections
            )
            if isinstance(base_annotated_image, np.ndarray):
                pil_img = Image.fromarray(base_annotated_image)
            else:
                pil_img = base_annotated_image

            buffered = io.BytesIO()
            pil_img.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

            def serialize_base_result_base64(base_result):
                serialized = []
                for item in base_result:
                    new_item = item.copy()
                    if isinstance(new_item["segmentation"], np.ndarray):
                        buffer = io.BytesIO()
                        np.save(buffer, new_item["segmentation"], allow_pickle=False)
                        buffer.seek(0)
                        b64_str = base64.b64encode(buffer.read()).decode("utf-8")
                        new_item["segmentation"] = b64_str
                    serialized.append(new_item)
                return serialized

            result_serialized = serialize_base_result_base64(base_result)
            # clear cache
            del sam2_base, mask_generator_base, base_annotator, base_detections
            gc.collect()
            torch.cuda.empty_cache()
            return {
                "message": "predict completed successfully",
                "result": result_serialized,
                "image": img_str,
            }
        elif command.lower() == "prompt_sample":
            task = kwargs.get("task", "")
            if task == "question-answering":
                prompt_text = f"""
                    Here is the context: 
                    {{context}}

                    Based on the above context, provide an answer to the following question: 
                    {{question}}

                    Answer:
                    """
            elif task == "text-classification":
                prompt_text = f"""
                    Summarize the following text into a single, concise paragraph focusing on the key ideas and important points:

                    Text: 
                    {{context}}

                    Summary:
                    """

            elif task == "summarization":
                prompt_text = f"""
                    Summarize the following text into a single, concise paragraph focusing on the key ideas and important points:

                    Text: 
                    {{context}}

                    Summary:
                    """
            return {
                "message": "prompt_sample completed successfully",
                "result": prompt_text,
            }

        elif command.lower() == "action-example":
            return {"message": "Done", "result": "Done"}

        elif command == "status":
            channel = kwargs.get("channel", None)

            if channel:
                # Nếu có truyền kênh cụ thể
                status_info = CHANNEL_STATUS.get(channel)
                if status_info is None:
                    return {"channel": channel, "status": "not_found"}
                elif isinstance(status_info, dict):
                    return {"channel": channel, **status_info}
                else:
                    return {"channel": channel, "status": status_info}
            else:
                # Lấy tất cả kênh
                if not CHANNEL_STATUS:
                    return {"message": "No channels available"}

                channels = []
                for ch, info in CHANNEL_STATUS.items():
                    if isinstance(info, dict):
                        channels.append({"channel": ch, **info})
                    else:
                        channels.append({"channel": ch, "status": info})

                return {"channels": channels}
    
        elif command.lower() == "tensorboard":
            project_id = kwargs.get("project_id")
            clone_dir = os.path.join(os.getcwd())
            def run_tensorboard():
                # train_dir = os.path.join(os.getcwd(), "{project_id}")
                # log_dir = os.path.join(os.getcwd(), "logs")
                if project_id:
                    p = subprocess.Popen(f"tensorboard --logdir ./sam2_logs/ --host 0.0.0.0 --port=6006", stdout=subprocess.PIPE, stderr=None, shell=True)
                else:
                    p = subprocess.Popen(f"tensorboard --logdir ./sam2_logs/ --host 0.0.0.0 --port=6006", stdout=subprocess.PIPE, stderr=None, shell=True)
                out = p.communicate()
                print(out)

            tensorboard_thread = threading.Thread(target=run_tensorboard)
            tensorboard_thread.start()
            return {"message": "tensorboardx started successfully"}
        else:
            return {"message": "command not supported", "result": None}
        
    @mcp.tool()
    def model(self, **kwargs):
        def load_model():
            global sam2_base, mask_generator_base
            subprocess.run("venv/bin/pip install -e .[dev]", shell=True)
            checkpoint_base = "./sam2.1_hiera_base_plus.pt"
            model_cfg_base = "configs/sam2.1/sam2.1_hiera_b+.yaml"
            try:
                torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
                if torch.cuda.get_device_properties(0).major >= 8:
                    torch.backends.cuda.matmul.allow_tf32 = True
                    torch.backends.cudnn.allow_tf32 = True
            except Exception as e:
                print(f"Torch setup warning: {e}")
            sam2_base = build_sam2(model_cfg_base, checkpoint_base, device="cuda")
            mask_generator_base = SAM2AutomaticMaskGenerator(sam2_base)
            return "Base model loaded successfully!"
        def segment_and_annotate(input_image):
            global mask_generator_base
            if mask_generator_base is None:
                return (
                    None,
                    "Model has not been loaded! Please click the Load Model button before.",
                )
            if isinstance(input_image, Image.Image):
                opened_image = np.array(input_image.convert("RGB"))
            else:
                opened_image = np.array(Image.fromarray(input_image).convert("RGB"))
            base_result = mask_generator_base.generate(opened_image)
            base_detections = sv.Detections.from_sam(sam_result=base_result)
            base_annotator = sv.MaskAnnotator(color_lookup=sv.ColorLookup.INDEX)
            base_annotated_image = opened_image.copy()
            base_annotated_image = base_annotator.annotate(
                base_annotated_image, detections=base_detections1
            )
            return base_annotated_image, "Completed!"
        with gr.Blocks() as demo:
            gr.Markdown("# Segment Anything 2.1 Base Demo")
            load_btn = gr.Button("Load Base Model")
            load_status = gr.Textbox(
                label="Model status",
                value="Model has not been loaded",
                interactive=False,
            )
            with gr.Row():
                with gr.Column():
                    input_image = gr.Image(
                        type="pil", label="Upload image", sources=["upload"]
                    )
                    segment_btn = gr.Button("Segment")
                with gr.Column():
                    output_base = gr.Image(label="Base SAM-2.1")
            output_status = gr.Textbox(label="Processing status", interactive=False)
            load_btn.click(fn=load_model, outputs=load_status)
            segment_btn.click(
                fn=segment_and_annotate,
                inputs=input_image,
                outputs=[output_base, output_status],
            )
        gradio_app, local_url, share_url = demo.launch(
            share=True,
            quiet=True,
            prevent_thread_lock=True,
            server_name="0.0.0.0",
            show_error=True,
        )
        return {"share_url": share_url, "local_url": local_url}

    @mcp.tool()
    def model_trial(self, project, **kwargs):
        import gradio as gr

        return {"message": "Done", "result": "Done"}

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

                result = self.action(
                    project, "predict", collection="", data={"img": input_img}
                )
                print(result)
                if result["result"]:
                    boxes = result["result"]["boxes"]
                    names = result["result"]["names"]
                    labels = result["result"]["labels"]

                    for box, label in zip(boxes, labels):
                        box = [int(i) for i in box]
                        label = int(label)
                        input_img = cv2.rectangle(
                            input_img, box, color=(255, 0, 0), thickness=2
                        )
                        # input_img = cv2.(input_img, names[label], (box[0], box[1]), color=(255, 0, 0), size=1)
                        input_img = cv2.putText(
                            input_img,
                            names[label],
                            (box[0], box[1]),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (0, 255, 0),
                            2,
                        )

                return input_img

            def download_btn(evt: gr.SelectData):
                print(f"Downloading {dataset_choosen}")
                return f'<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/4.7.0/css/font-awesome.min.css"><a href="/my_ml_backend/datasets/{evt.value}" style="font-size:50px"> <i class="fa fa-download"></i> Download this dataset</a>'

            def trial_training(dataset_choosen):
                print(f"Training with {dataset_choosen}")
                result = self.action(
                    project, "train", collection="", data=dataset_choosen
                )
                return result["message"]

            def get_checkpoint_list(project):
                print("GETTING CHECKPOINT LIST")
                print(f"Proejct: {project}")
                import os

                checkpoint_list = [
                    i for i in os.listdir("my_ml_backend/models") if i.endswith(".pt")
                ]
                checkpoint_list = [
                    f"<a href='./my_ml_backend/checkpoints/{i}' download>{i}</a>"
                    for i in checkpoint_list
                ]
                if os.path.exists(f"my_ml_backend/{project}"):
                    for folder in os.listdir(f"my_ml_backend/{project}"):
                        if "train" in folder:
                            project_checkpoint_list = [
                                i
                                for i in os.listdir(
                                    f"my_ml_backend/{project}/{folder}/weights"
                                )
                                if i.endswith(".pt")
                            ]
                            project_checkpoint_list = [
                                f"<a href='./my_ml_backend/{project}/{folder}/weights/{i}' download>{folder}-{i}</a>"
                                for i in project_checkpoint_list
                            ]
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

                    gr.Interface(
                        predict,
                        gr.Image(
                            elem_classes=["upload_image"],
                            sources="upload",
                            container=False,
                            height=345,
                            show_label=False,
                        ),
                        gr.Image(
                            elem_classes=["upload_image"],
                            container=False,
                            height=345,
                            show_label=False,
                        ),
                        allow_flagging=False,
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
                            gr.Markdown(
                                "## Dataset template to prepare your own and initiate training"
                            )
                            with gr.Row():
                                # get all filename in datasets folder
                                if not os.path.exists(f"./datasets"):
                                    os.makedirs(f"./datasets")
                                datasets = [
                                    (f"dataset{i}", name)
                                    for i, name in enumerate(os.listdir("./datasets"))
                                ]

                                dataset_choosen = gr.Dropdown(
                                    datasets,
                                    label="Choose dataset",
                                    show_label=False,
                                    interactive=True,
                                    type="value",
                                )
                                # gr.Button("Download this dataset", variant="primary").click(download_btn, dataset_choosen, gr.HTML())
                                download_link = gr.HTML(
                                    """
                                        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/4.7.0/css/font-awesome.min.css">
                                        <a href='' style="font-size:24px"><i class="fa fa-download" ></i> Download this dataset</a>"""
                                )

                                dataset_choosen.select(
                                    download_btn, None, download_link
                                )

                                # when the button is clicked, download the dataset from dropdown
                                # download_btn
                            gr.Markdown(
                                "## Upload your sample dataset to have a trial training"
                            )
                            # gr.File(file_types=['tar','zip'])
                            gr.Interface(
                                predict,
                                gr.File(
                                    elem_classes=["upload_image"],
                                    file_types=["tar", "zip"],
                                ),
                                gr.Label(
                                    elem_classes=["upload_image"], container=False
                                ),
                                allow_flagging=False,
                            )
                            with gr.Row():
                                gr.Markdown(f"## You can attemp up to {2} FLOps")
                                gr.Button("Trial Train", variant="primary").click(
                                    trial_training, dataset_choosen, None
                                )

                # with gr.TabItem("Download"):
                #     with gr.Column():
                #         gr.Markdown("## Download")
                #         with gr.Column():
                #             gr.HTML(get_checkpoint_list(project))

        gradio_app, local_url, share_url = demo.launch(
            share=True,
            quiet=True,
            prevent_thread_lock=True,
            server_name="0.0.0.0",
            show_error=True,
        )

        return {"share_url": share_url, "local_url": local_url}

    @mcp.tool()
    def download(self, project, **kwargs):
        from flask import request, send_from_directory

        file_path = request.args.get("path")
        print(request.args)
        return send_from_directory(os.getcwd(), file_path)
