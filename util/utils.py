# from ultralytics import YOLO
import os
import io
import base64
import time
import tempfile
import subprocess
_RUNTIME_CACHE_DIR = os.path.join(tempfile.gettempdir(), "omniparser-runtime")
os.makedirs(_RUNTIME_CACHE_DIR, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", os.path.join(_RUNTIME_CACHE_DIR, "matplotlib"))
os.environ.setdefault("YOLO_CONFIG_DIR", os.path.join(_RUNTIME_CACHE_DIR, "ultralytics"))
os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
from PIL import Image, ImageDraw, ImageFont
import json
import requests
# utility function
import os
from openai import AzureOpenAI

import json
import sys
import os
import cv2
import numpy as np
# %matplotlib inline
from matplotlib import pyplot as plt
import easyocr
from paddleocr import PaddleOCR
HF_CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub")
DEFAULT_ICON_DESCRIPTION = "icon"
DEFAULT_OCR_LANGS = ('ch_sim', 'en')
DEFAULT_OCR_BACKEND = 'easyocr'
VISION_OCR_SOURCE_PATH = os.path.join(os.path.dirname(__file__), 'vision_ocr.swift')
VISION_OCR_BINARY_PATH = os.path.join(_RUNTIME_CACHE_DIR, 'vision_ocr')
easyocr_readers = {}

def _build_paddle_ocr():
    kwargs = {
        'lang': 'en',
        'use_angle_cls': False,
        'use_gpu': False,  # using cuda will conflict with pytorch in the same process
        'show_log': False,
        'use_dilation': True,  # improves accuracy
        'det_db_score_mode': 'slow',  # improves accuracy
        'max_batch_size': 1024,
        'rec_batch_num': 1024,
    }

    while True:
        try:
            return PaddleOCR(**kwargs)
        except TypeError:
            unsupported = ['max_batch_size', 'rec_batch_num', 'use_dilation', 'det_db_score_mode']
            trimmed = {k: v for k, v in kwargs.items() if k not in unsupported}
            if trimmed == kwargs:
                raise
            kwargs = trimmed
        except ValueError as error:
            message = str(error)
            if 'Unknown argument:' in message:
                unknown_arg = message.split('Unknown argument:', 1)[1].strip()
                if unknown_arg in kwargs:
                    kwargs.pop(unknown_arg, None)
                    continue
            raise

paddle_ocr = None

def get_paddle_ocr():
    global paddle_ocr
    if paddle_ocr is None:
        paddle_ocr = _build_paddle_ocr()
    return paddle_ocr

def normalize_ocr_langs(raw_langs):
    if isinstance(raw_langs, (list, tuple)):
        candidates = [str(lang).strip() for lang in raw_langs if str(lang).strip()]
    else:
        candidates = [part.strip() for part in str(raw_langs or '').split(',') if part.strip()]
    return tuple(dict.fromkeys(candidates)) or DEFAULT_OCR_LANGS

def _build_easyocr_reader(langs):
    normalized_langs = normalize_ocr_langs(langs)
    fallback_candidates = []
    if normalized_langs:
        fallback_candidates.append(normalized_langs)
    if ('en',) not in fallback_candidates:
        fallback_candidates.append(('en',))

    resolved_device = resolve_torch_device('auto')
    easyocr_gpu = resolved_device if resolved_device in {'cuda', 'mps'} else False
    print(f'EasyOCR requested device={resolved_device}')

    last_error = None
    for candidate_langs in fallback_candidates:
        try:
            return easyocr.Reader(list(candidate_langs), gpu=easyocr_gpu, download_enabled=False), candidate_langs
        except Exception as error:
            last_error = error
            continue
    raise last_error or RuntimeError('failed to initialize EasyOCR reader')

def get_easyocr_reader(langs=None):
    normalized_langs = normalize_ocr_langs(langs)
    cache_key = ','.join(normalized_langs)
    if cache_key not in easyocr_readers:
        reader, resolved_langs = _build_easyocr_reader(normalized_langs)
        easyocr_readers[cache_key] = (reader, resolved_langs)
        print(f'EasyOCR initialized with langs={list(resolved_langs)}')
    return easyocr_readers[cache_key]


def ensure_vision_ocr_binary():
    if sys.platform != 'darwin':
        raise RuntimeError('Vision OCR is only supported on macOS')
    if not os.path.exists(VISION_OCR_SOURCE_PATH):
        raise RuntimeError(f'Vision OCR source not found: {VISION_OCR_SOURCE_PATH}')

    source_mtime = os.path.getmtime(VISION_OCR_SOURCE_PATH)
    binary_ready = os.path.exists(VISION_OCR_BINARY_PATH) and os.path.getmtime(VISION_OCR_BINARY_PATH) >= source_mtime
    if binary_ready:
        return VISION_OCR_BINARY_PATH

    compile_cmd = ['/usr/bin/swiftc', '-O', VISION_OCR_SOURCE_PATH, '-o', VISION_OCR_BINARY_PATH]
    result = subprocess.run(compile_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f'Failed to compile Vision OCR helper: {result.stderr.strip() or result.stdout.strip()}')
    return VISION_OCR_BINARY_PATH


def run_vision_ocr(image_source, ocr_langs=None):
    binary_path = ensure_vision_ocr_binary()
    normalized_langs = normalize_ocr_langs(ocr_langs)

    with tempfile.NamedTemporaryFile(prefix='vision-ocr-', suffix='.jpg', dir=_RUNTIME_CACHE_DIR, delete=False) as tmp:
        tmp_path = tmp.name
    try:
        image_source.save(tmp_path, format='JPEG', quality=85)
        result = subprocess.run(
            [binary_path, tmp_path, ','.join(normalized_langs)],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(result.stderr.strip() or result.stdout.strip() or 'Vision OCR failed')
        payload = json.loads(result.stdout or '{}')
        items = payload.get('items') or []
        text = [str(item.get('text') or '').strip() for item in items if str(item.get('text') or '').strip()]
        boxes = []
        for item in items:
            bbox = item.get('bbox') or []
            if len(bbox) < 4:
                continue
            x1, y1, x2, y2 = [int(round(float(v))) for v in bbox[:4]]
            boxes.append([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])
        return text, boxes, normalized_langs
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
import time
import base64

import os
import ast
import torch
from typing import Tuple, List, Union
from torchvision.ops import box_convert
import re
from torchvision.transforms import ToPILImage
import supervision as sv
import torchvision.transforms as T
from util.box_annotator import BoxAnnotator 

def resolve_cached_snapshot_path(model_id):
    normalized_id = str(model_id or '').strip()
    if not normalized_id:
        return ''

    cache_root = os.path.join(HF_CACHE_DIR, f"models--{normalized_id.replace('/', '--')}")
    refs_main_path = os.path.join(cache_root, 'refs', 'main')
    try:
        if os.path.exists(refs_main_path):
            snapshot_id = open(refs_main_path, 'r', encoding='utf-8').read().strip()
            snapshot_path = os.path.join(cache_root, 'snapshots', snapshot_id)
            if os.path.exists(snapshot_path):
                return snapshot_path
    except Exception:
        pass
    return ''

def has_florence_processor_files(model_path):
    if not model_path:
        return False

    required_files = [
        'preprocessor_config.json',
        'tokenizer.json',
        'tokenizer_config.json',
        'special_tokens_map.json',
        'vocab.json',
        'merges.txt',
    ]
    return all(os.path.exists(os.path.join(model_path, filename)) for filename in required_files)

def resolve_florence_processor_source(model_name_or_path):
    candidates = [
        model_name_or_path,
        resolve_cached_snapshot_path('microsoft/Florence-2-base'),
        resolve_cached_snapshot_path('microsoft/Florence-2-base-ft'),
    ]
    for candidate in candidates:
        if has_florence_processor_files(candidate):
            return candidate, True
    return "microsoft/Florence-2-base", False


def resolve_torch_device(requested_device=None):
    requested = str(requested_device or 'auto').strip().lower()
    mps_backend = getattr(torch.backends, 'mps', None)
    mps_available = bool(mps_backend and mps_backend.is_available())

    if requested in {'', 'auto'}:
        if torch.cuda.is_available():
            return 'cuda'
        if mps_available:
            return 'mps'
        return 'cpu'

    if requested == 'cuda':
        if torch.cuda.is_available():
            return 'cuda'
        print('CUDA requested but unavailable, falling back to CPU')
        return 'cpu'

    if requested == 'mps':
        if mps_available:
            return 'mps'
        print('MPS requested but unavailable, falling back to CPU')
        return 'cpu'

    if requested == 'cpu':
        return 'cpu'

    print(f'Unknown device "{requested}", falling back to auto selection')
    return resolve_torch_device('auto')


def get_caption_model_processor(model_name, model_name_or_path="Salesforce/blip2-opt-2.7b", device=None):
    device = resolve_torch_device(device)
    dtype = torch.float32 if device == 'cpu' else torch.float16
    if model_name == "blip2":
        from transformers import Blip2Processor, Blip2ForConditionalGeneration
        processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
        model = Blip2ForConditionalGeneration.from_pretrained(
            model_name_or_path, device_map=None, torch_dtype=dtype
        )
    elif model_name == "florence2":
        from transformers import Florence2Processor, Florence2ForConditionalGeneration
        processor_source, local_files_only = resolve_florence_processor_source(model_name_or_path)
        processor = Florence2Processor.from_pretrained(processor_source, local_files_only=local_files_only)
        model = Florence2ForConditionalGeneration.from_pretrained(model_name_or_path, torch_dtype=dtype)
    else:
        raise ValueError(f'Unsupported caption model: {model_name}')
    return {'model': model.to(device), 'processor': processor}


def get_yolo_model(model_path, device=None):
    from ultralytics import YOLO
    model = YOLO(model_path)
    resolved_device = resolve_torch_device(device)
    try:
        model.to(resolved_device)
    except Exception as error:
        print(f'Failed to move YOLO model to {resolved_device}: {error}')
        resolved_device = 'cpu'
        model.to(resolved_device)
    setattr(model, '_omniparser_device', resolved_device)
    return model


@torch.inference_mode()
def get_parsed_content_icon(filtered_boxes, starting_idx, image_source, caption_model_processor, prompt=None, batch_size=128):
    # Number of samples per batch, --> 128 roughly takes 4 GB of GPU memory for florence v2 model
    to_pil = ToPILImage()
    if starting_idx:
        non_ocr_boxes = filtered_boxes[starting_idx:]
    else:
        non_ocr_boxes = filtered_boxes
    croped_pil_image = []
    for i, coord in enumerate(non_ocr_boxes):
        try:
            xmin, xmax = int(coord[0]*image_source.shape[1]), int(coord[2]*image_source.shape[1])
            ymin, ymax = int(coord[1]*image_source.shape[0]), int(coord[3]*image_source.shape[0])
            cropped_image = image_source[ymin:ymax, xmin:xmax, :]
            cropped_image = cv2.resize(cropped_image, (64, 64))
            croped_pil_image.append(to_pil(cropped_image))
        except:
            continue

    model, processor = caption_model_processor['model'], caption_model_processor['processor']
    if not prompt:
        if 'florence' in model.config.name_or_path:
            prompt = "<CAPTION>"
        else:
            prompt = "The image shows"
    
    generated_texts = []
    device = model.device
    for i in range(0, len(croped_pil_image), batch_size):
        start = time.time()
        batch = croped_pil_image[i:i+batch_size]
        t1 = time.time()
        input_kwargs = {'device': device}
        if model.device.type != 'cpu' and model.dtype in (torch.float16, torch.bfloat16):
            input_kwargs['dtype'] = model.dtype
        if model.device.type == 'cuda':
            inputs = processor(images=batch, text=[prompt]*len(batch), return_tensors="pt", do_resize=False).to(**input_kwargs)
        else:
            inputs = processor(images=batch, text=[prompt]*len(batch), return_tensors="pt").to(**input_kwargs)
        if 'florence' in model.config.name_or_path:
            generated_ids = model.generate(input_ids=inputs["input_ids"],pixel_values=inputs["pixel_values"],max_new_tokens=20,num_beams=1, do_sample=False)
        else:
            generated_ids = model.generate(**inputs, max_length=100, num_beams=5, no_repeat_ngram_size=2, early_stopping=True, num_return_sequences=1) # temperature=0.01, do_sample=True,
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)
        generated_text = [gen.strip() for gen in generated_text]
        generated_texts.extend(generated_text)
    
    return generated_texts



def get_parsed_content_icon_phi3v(filtered_boxes, ocr_bbox, image_source, caption_model_processor):
    to_pil = ToPILImage()
    if ocr_bbox:
        non_ocr_boxes = filtered_boxes[len(ocr_bbox):]
    else:
        non_ocr_boxes = filtered_boxes
    croped_pil_image = []
    for i, coord in enumerate(non_ocr_boxes):
        xmin, xmax = int(coord[0]*image_source.shape[1]), int(coord[2]*image_source.shape[1])
        ymin, ymax = int(coord[1]*image_source.shape[0]), int(coord[3]*image_source.shape[0])
        cropped_image = image_source[ymin:ymax, xmin:xmax, :]
        croped_pil_image.append(to_pil(cropped_image))

    model, processor = caption_model_processor['model'], caption_model_processor['processor']
    device = model.device
    messages = [{"role": "user", "content": "<|image_1|>\ndescribe the icon in one sentence"}] 
    prompt = processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    batch_size = 5  # Number of samples per batch
    generated_texts = []

    for i in range(0, len(croped_pil_image), batch_size):
        images = croped_pil_image[i:i+batch_size]
        image_inputs = [processor.image_processor(x, return_tensors="pt") for x in images]
        inputs ={'input_ids': [], 'attention_mask': [], 'pixel_values': [], 'image_sizes': []}
        texts = [prompt] * len(images)
        for i, txt in enumerate(texts):
            input = processor._convert_images_texts_to_inputs(image_inputs[i], txt, return_tensors="pt")
            inputs['input_ids'].append(input['input_ids'])
            inputs['attention_mask'].append(input['attention_mask'])
            inputs['pixel_values'].append(input['pixel_values'])
            inputs['image_sizes'].append(input['image_sizes'])
        max_len = max([x.shape[1] for x in inputs['input_ids']])
        for i, v in enumerate(inputs['input_ids']):
            inputs['input_ids'][i] = torch.cat([processor.tokenizer.pad_token_id * torch.ones(1, max_len - v.shape[1], dtype=torch.long), v], dim=1)
            inputs['attention_mask'][i] = torch.cat([torch.zeros(1, max_len - v.shape[1], dtype=torch.long), inputs['attention_mask'][i]], dim=1)
        inputs_cat = {k: torch.concatenate(v).to(device) for k, v in inputs.items()}

        generation_args = { 
            "max_new_tokens": 25, 
            "temperature": 0.01, 
            "do_sample": False, 
        } 
        generate_ids = model.generate(**inputs_cat, eos_token_id=processor.tokenizer.eos_token_id, **generation_args) 
        # # remove input tokens 
        generate_ids = generate_ids[:, inputs_cat['input_ids'].shape[1]:]
        response = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        response = [res.strip('\n').strip() for res in response]
        generated_texts.extend(response)

    return generated_texts

def remove_overlap(boxes, iou_threshold, ocr_bbox=None):
    assert ocr_bbox is None or isinstance(ocr_bbox, List)

    def box_area(box):
        return (box[2] - box[0]) * (box[3] - box[1])

    def intersection_area(box1, box2):
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        return max(0, x2 - x1) * max(0, y2 - y1)

    def IoU(box1, box2):
        intersection = intersection_area(box1, box2)
        union = box_area(box1) + box_area(box2) - intersection + 1e-6
        if box_area(box1) > 0 and box_area(box2) > 0:
            ratio1 = intersection / box_area(box1)
            ratio2 = intersection / box_area(box2)
        else:
            ratio1, ratio2 = 0, 0
        return max(intersection / union, ratio1, ratio2)

    def is_inside(box1, box2):
        # return box1[0] >= box2[0] and box1[1] >= box2[1] and box1[2] <= box2[2] and box1[3] <= box2[3]
        intersection = intersection_area(box1, box2)
        ratio1 = intersection / box_area(box1)
        return ratio1 > 0.95

    boxes = boxes.tolist()
    filtered_boxes = []
    if ocr_bbox:
        filtered_boxes.extend(ocr_bbox)
    # print('ocr_bbox!!!', ocr_bbox)
    for i, box1 in enumerate(boxes):
        # if not any(IoU(box1, box2) > iou_threshold and box_area(box1) > box_area(box2) for j, box2 in enumerate(boxes) if i != j):
        is_valid_box = True
        for j, box2 in enumerate(boxes):
            # keep the smaller box
            if i != j and IoU(box1, box2) > iou_threshold and box_area(box1) > box_area(box2):
                is_valid_box = False
                break
        if is_valid_box:
            # add the following 2 lines to include ocr bbox
            if ocr_bbox:
                # only add the box if it does not overlap with any ocr bbox
                if not any(IoU(box1, box3) > iou_threshold and not is_inside(box1, box3) for k, box3 in enumerate(ocr_bbox)):
                    filtered_boxes.append(box1)
            else:
                filtered_boxes.append(box1)
    return torch.tensor(filtered_boxes)


def remove_overlap_new(boxes, iou_threshold, ocr_bbox=None):
    '''
    ocr_bbox format: [{'type': 'text', 'bbox':[x,y], 'interactivity':False, 'content':str }, ...]
    boxes format: [{'type': 'icon', 'bbox':[x,y], 'interactivity':True, 'content':None }, ...]

    '''
    assert ocr_bbox is None or isinstance(ocr_bbox, List)

    def box_area(box):
        return (box[2] - box[0]) * (box[3] - box[1])

    def intersection_area(box1, box2):
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        return max(0, x2 - x1) * max(0, y2 - y1)

    def IoU(box1, box2):
        intersection = intersection_area(box1, box2)
        union = box_area(box1) + box_area(box2) - intersection + 1e-6
        if box_area(box1) > 0 and box_area(box2) > 0:
            ratio1 = intersection / box_area(box1)
            ratio2 = intersection / box_area(box2)
        else:
            ratio1, ratio2 = 0, 0
        return max(intersection / union, ratio1, ratio2)

    def is_inside(box1, box2):
        # return box1[0] >= box2[0] and box1[1] >= box2[1] and box1[2] <= box2[2] and box1[3] <= box2[3]
        intersection = intersection_area(box1, box2)
        ratio1 = intersection / box_area(box1)
        return ratio1 > 0.80

    # boxes = boxes.tolist()
    filtered_boxes = []
    if ocr_bbox:
        filtered_boxes.extend(ocr_bbox)
    # print('ocr_bbox!!!', ocr_bbox)
    for i, box1_elem in enumerate(boxes):
        box1 = box1_elem['bbox']
        is_valid_box = True
        for j, box2_elem in enumerate(boxes):
            # keep the smaller box
            box2 = box2_elem['bbox']
            if i != j and IoU(box1, box2) > iou_threshold and box_area(box1) > box_area(box2):
                is_valid_box = False
                break
        if is_valid_box:
            if ocr_bbox:
                # keep yolo boxes + prioritize ocr label
                box_added = False
                ocr_labels = ''
                for box3_elem in ocr_bbox:
                    if not box_added:
                        box3 = box3_elem['bbox']
                        if is_inside(box3, box1): # ocr inside icon
                            # box_added = True
                            # delete the box3_elem from ocr_bbox
                            try:
                                # gather all ocr labels
                                ocr_labels += box3_elem['content'] + ' '
                                filtered_boxes.remove(box3_elem)
                            except:
                                continue
                            # break
                        elif is_inside(box1, box3): # icon inside ocr, don't added this icon box, no need to check other ocr bbox bc no overlap between ocr bbox, icon can only be in one ocr box
                            box_added = True
                            break
                        else:
                            continue
                if not box_added:
                    if ocr_labels:
                        filtered_boxes.append({'type': 'icon', 'bbox': box1_elem['bbox'], 'interactivity': True, 'content': ocr_labels, 'source':'box_yolo_content_ocr'})
                    else:
                        filtered_boxes.append({'type': 'icon', 'bbox': box1_elem['bbox'], 'interactivity': True, 'content': None, 'source':'box_yolo_content_yolo'})
            else:
                filtered_boxes.append(box1_elem)
    return filtered_boxes # torch.tensor(filtered_boxes)


def load_image(image_path: str) -> Tuple[np.array, torch.Tensor]:
    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image_source = Image.open(image_path).convert("RGB")
    image = np.asarray(image_source)
    image_transformed, _ = transform(image_source, None)
    return image, image_transformed


def annotate(image_source: np.ndarray, boxes: torch.Tensor, logits: torch.Tensor, phrases: List[str], text_scale: float, 
             text_padding=5, text_thickness=2, thickness=3) -> np.ndarray:
    """    
    This function annotates an image with bounding boxes and labels.

    Parameters:
    image_source (np.ndarray): The source image to be annotated.
    boxes (torch.Tensor): A tensor containing bounding box coordinates. in cxcywh format, pixel scale
    logits (torch.Tensor): A tensor containing confidence scores for each bounding box.
    phrases (List[str]): A list of labels for each bounding box.
    text_scale (float): The scale of the text to be displayed. 0.8 for mobile/web, 0.3 for desktop # 0.4 for mind2web

    Returns:
    np.ndarray: The annotated image.
    """
    h, w, _ = image_source.shape
    boxes = boxes * torch.Tensor([w, h, w, h])
    xyxy = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()
    xywh = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xywh").numpy()
    detections = sv.Detections(xyxy=xyxy)

    labels = [f"{phrase}" for phrase in range(boxes.shape[0])]

    box_annotator = BoxAnnotator(text_scale=text_scale, text_padding=text_padding,text_thickness=text_thickness,thickness=thickness) # 0.8 for mobile/web, 0.3 for desktop # 0.4 for mind2web
    annotated_frame = image_source.copy()
    annotated_frame = box_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels, image_size=(w,h))

    label_coordinates = {f"{phrase}": v for phrase, v in zip(phrases, xywh)}
    return annotated_frame, label_coordinates


def predict(model, image, caption, box_threshold, text_threshold):
    """ Use huggingface model to replace the original model
    """
    model, processor = model['model'], model['processor']
    device = model.device

    inputs = processor(images=image, text=caption, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)

    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        box_threshold=box_threshold, # 0.4,
        text_threshold=text_threshold, # 0.3,
        target_sizes=[image.size[::-1]]
    )[0]
    boxes, logits, phrases = results["boxes"], results["scores"], results["labels"]
    return boxes, logits, phrases


def predict_yolo(model, image, box_threshold, imgsz, scale_img, iou_threshold=0.7):
    """ Use huggingface model to replace the original model
    """
    predict_kwargs = {
        'source': image,
        'conf': box_threshold,
        'iou': iou_threshold,
    }
    if scale_img:
        predict_kwargs['imgsz'] = imgsz
    model_device = getattr(model, '_omniparser_device', None)
    if model_device:
        predict_kwargs['device'] = model_device
    result = model.predict(**predict_kwargs)
    boxes = result[0].boxes.xyxy#.tolist() # in pixel space
    conf = result[0].boxes.conf
    phrases = [str(i) for i in range(len(boxes))]

    return boxes, conf, phrases

def int_box_area(box, w, h):
    x1, y1, x2, y2 = box
    int_box = [int(x1*w), int(y1*h), int(x2*w), int(y2*h)]
    area = (int_box[2] - int_box[0]) * (int_box[3] - int_box[1])
    return area

def get_som_labeled_img(image_source: Union[str, Image.Image], model=None, BOX_TRESHOLD=0.01, output_coord_in_ratio=False, ocr_bbox=None, text_scale=0.4, text_padding=5, draw_bbox_config=None, caption_model_processor=None, ocr_text=[], use_local_semantics=True, iou_threshold=0.9,prompt=None, scale_img=False, imgsz=None, batch_size=128):
    """Process either an image path or Image object
    
    Args:
        image_source: Either a file path (str) or PIL Image object
        ...
    """
    if isinstance(image_source, str):
        image_source = Image.open(image_source)
    image_source = image_source.convert("RGB") # for CLIP
    w, h = image_source.size
    if not imgsz:
        imgsz = (h, w)
    # print('image size:', w, h)
    xyxy, logits, phrases = predict_yolo(model=model, image=image_source, box_threshold=BOX_TRESHOLD, imgsz=imgsz, scale_img=scale_img, iou_threshold=0.1)
    xyxy = xyxy / torch.Tensor([w, h, w, h]).to(xyxy.device)
    image_source = np.asarray(image_source)
    phrases = [str(i) for i in range(len(phrases))]

    # annotate the image with labels
    if ocr_bbox:
        ocr_bbox = torch.tensor(ocr_bbox) / torch.Tensor([w, h, w, h])
        ocr_bbox=ocr_bbox.tolist()
    else:
        print('no ocr bbox!!!')
        ocr_bbox = None

    ocr_bbox_elem = []
    if ocr_bbox:
        ocr_bbox_elem = [
            {'type': 'text', 'bbox': box, 'interactivity': False, 'content': txt, 'source': 'box_ocr_content_ocr'}
            for box, txt in zip(ocr_bbox, ocr_text)
            if int_box_area(box, w, h) > 0
        ]
    xyxy_elem = [{'type': 'icon', 'bbox':box, 'interactivity':True, 'content':None} for box in xyxy.tolist() if int_box_area(box, w, h) > 0]
    filtered_boxes = remove_overlap_new(boxes=xyxy_elem, iou_threshold=iou_threshold, ocr_bbox=ocr_bbox_elem)
    
    # sort the filtered_boxes so that the one with 'content': None is at the end, and get the index of the first 'content': None
    filtered_boxes_elem = sorted(filtered_boxes, key=lambda x: x['content'] is None)
    # get the index of the first 'content': None
    starting_idx = next((i for i, box in enumerate(filtered_boxes_elem) if box['content'] is None), -1)
    filtered_boxes = torch.tensor([box['bbox'] for box in filtered_boxes_elem])
    print('len(filtered_boxes):', len(filtered_boxes), starting_idx)

    # get parsed icon local semantics
    time1 = time.time()
    if use_local_semantics and caption_model_processor:
        caption_model = caption_model_processor['model']
        if 'phi3_v' in caption_model.config.model_type: 
            parsed_content_icon = get_parsed_content_icon_phi3v(filtered_boxes, ocr_bbox, image_source, caption_model_processor)
        else:
            parsed_content_icon = get_parsed_content_icon(filtered_boxes, starting_idx, image_source, caption_model_processor, prompt=prompt,batch_size=batch_size)
        ocr_text = [f"Text Box ID {i}: {txt}" for i, txt in enumerate(ocr_text)]
        icon_start = len(ocr_text)
        parsed_content_icon_ls = []
        # fill the filtered_boxes_elem None content with parsed_content_icon in order
        for i, box in enumerate(filtered_boxes_elem):
            if box['content'] is None:
                box['content'] = parsed_content_icon.pop(0)
        for i, txt in enumerate(parsed_content_icon):
            parsed_content_icon_ls.append(f"Icon Box ID {str(i+icon_start)}: {txt}")
        parsed_content_merged = ocr_text + parsed_content_icon_ls
    else:
        for box in filtered_boxes_elem:
            if box.get('type') == 'icon' and not box.get('content'):
                box['content'] = DEFAULT_ICON_DESCRIPTION
        ocr_text = [f"Text Box ID {i}: {txt}" for i, txt in enumerate(ocr_text)]
        parsed_content_merged = ocr_text
    print('time to get parsed content:', time.time()-time1)

    filtered_boxes = box_convert(boxes=filtered_boxes, in_fmt="xyxy", out_fmt="cxcywh")

    phrases = [i for i in range(len(filtered_boxes))]
    
    # draw boxes
    if draw_bbox_config:
        annotated_frame, label_coordinates = annotate(image_source=image_source, boxes=filtered_boxes, logits=logits, phrases=phrases, **draw_bbox_config)
    else:
        annotated_frame, label_coordinates = annotate(image_source=image_source, boxes=filtered_boxes, logits=logits, phrases=phrases, text_scale=text_scale, text_padding=text_padding)
    
    pil_img = Image.fromarray(annotated_frame)
    buffered = io.BytesIO()
    pil_img.save(buffered, format="PNG")
    encoded_image = base64.b64encode(buffered.getvalue()).decode('ascii')
    if output_coord_in_ratio:
        label_coordinates = {k: [v[0]/w, v[1]/h, v[2]/w, v[3]/h] for k, v in label_coordinates.items()}
        assert w == annotated_frame.shape[1] and h == annotated_frame.shape[0]

    return encoded_image, label_coordinates, filtered_boxes_elem


def get_xywh(input):
    x, y, w, h = input[0][0], input[0][1], input[2][0] - input[0][0], input[2][1] - input[0][1]
    x, y, w, h = int(x), int(y), int(w), int(h)
    return x, y, w, h

def get_xyxy(input):
    x, y, xp, yp = input[0][0], input[0][1], input[2][0], input[2][1]
    x, y, xp, yp = int(x), int(y), int(xp), int(yp)
    return x, y, xp, yp

def get_xywh_yolo(input):
    x, y, w, h = input[0], input[1], input[2] - input[0], input[3] - input[1]
    x, y, w, h = int(x), int(y), int(w), int(h)
    return x, y, w, h

def check_ocr_box(image_source: Union[str, Image.Image], display_img = True, output_bb_format='xywh', goal_filtering=None, easyocr_args=None, use_paddleocr=False, ocr_langs=None):
    if isinstance(image_source, str):
        image_source = Image.open(image_source)
    if image_source.mode == 'RGBA':
        # Convert RGBA to RGB to avoid alpha channel issues
        image_source = image_source.convert('RGB')
    image_np = np.array(image_source)
    w, h = image_source.size
    if use_paddleocr is True:
        if easyocr_args is None:
            text_threshold = 0.5
        else:
            text_threshold = easyocr_args['text_threshold']
        result = get_paddle_ocr().ocr(image_np, cls=False)[0]
        coord = [item[0] for item in result if item[1][1] > text_threshold]
        text = [item[1][0] for item in result if item[1][1] > text_threshold]
    elif use_paddleocr == 'vision':
        text, bb, resolved_langs = run_vision_ocr(image_source, ocr_langs)
        coord = bb
        print(f'Vision OCR initialized with langs={list(resolved_langs)}')
    else:  # EasyOCR
        if easyocr_args is None:
            easyocr_args = {}
        easyocr_reader, resolved_langs = get_easyocr_reader(ocr_langs)
        result = easyocr_reader.readtext(image_np, **easyocr_args)
        coord = [item[0] for item in result]
        text = [item[1] for item in result]
    if display_img:
        opencv_img = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        bb = []
        for item in coord:
            x, y, a, b = get_xywh(item)
            bb.append((x, y, a, b))
            cv2.rectangle(opencv_img, (x, y), (x+a, y+b), (0, 255, 0), 2)
        #  matplotlib expects RGB
        plt.imshow(cv2.cvtColor(opencv_img, cv2.COLOR_BGR2RGB))
    else:
        if output_bb_format == 'xywh':
            bb = [get_xywh(item) for item in coord]
        elif output_bb_format == 'xyxy':
            bb = [get_xyxy(item) for item in coord]
    return (text, bb), goal_filtering
