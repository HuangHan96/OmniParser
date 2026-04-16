from util.utils import get_som_labeled_img, get_caption_model_processor, get_yolo_model, check_ocr_box, resolve_torch_device
import torch
from PIL import Image
import io
import base64
import time
from typing import Dict


class Omniparser(object):
    def __init__(self, config: Dict):
        self.config = config
        device = resolve_torch_device(config.get('device'))
        self.device = device

        self.som_model = get_yolo_model(model_path=config['som_model_path'], device=device)
        self.ocr_backend = str(config.get('ocr_backend') or 'easyocr').strip().lower()
        self.ocr_langs = str(config.get('ocr_langs') or 'ch_sim,en').strip()
        self.caption_model_processor = None
        self.use_local_semantics = False
        try:
            self.caption_model_processor = get_caption_model_processor(
                model_name=config['caption_model_name'],
                model_name_or_path=config['caption_model_path'],
                device=device,
            )
            self.use_local_semantics = self.caption_model_processor is not None
        except Exception as error:
            print(f'Caption model init failed, continuing without local semantics: {error}')
        print(f'Omniparser initialized on device={self.device}!!!')

    def parse(self, image_base64: str):
        parse_started_at = time.time()
        image_bytes = base64.b64decode(image_base64)
        image = Image.open(io.BytesIO(image_bytes))
        print('image size:', image.size)
        
        box_overlay_ratio = max(image.size) / 3200
        draw_bbox_config = {
            'text_scale': 0.8 * box_overlay_ratio,
            'text_thickness': max(int(2 * box_overlay_ratio), 1),
            'text_padding': max(int(3 * box_overlay_ratio), 1),
            'thickness': max(int(3 * box_overlay_ratio), 1),
        }

        ocr_started_at = time.time()
        (text, ocr_bbox), _ = check_ocr_box(
            image,
            display_img=False,
            output_bb_format='xyxy',
            easyocr_args={
                'text_threshold': 0.8,
                'batch_size': 64,
                'workers': 0,
                'canvas_size': 1280,
                'min_size': 24,
                'mag_ratio': 1.0,
                'paragraph': False,
                'decoder': 'greedy'
            },
            use_paddleocr='vision' if self.ocr_backend == 'vision' else (self.ocr_backend == 'paddleocr'),
            ocr_langs=self.ocr_langs,
        )
        print(f'ocr time: {time.time() - ocr_started_at:.3f}s, text_boxes={len(ocr_bbox) if ocr_bbox else 0}')

        som_started_at = time.time()
        dino_labled_img, label_coordinates, parsed_content_list = get_som_labeled_img(
            image,
            self.som_model,
            BOX_TRESHOLD=self.config['BOX_TRESHOLD'],
            output_coord_in_ratio=True,
            ocr_bbox=ocr_bbox,
            draw_bbox_config=draw_bbox_config,
            caption_model_processor=self.caption_model_processor,
            ocr_text=text,
            use_local_semantics=self.use_local_semantics,
            iou_threshold=0.7,
            scale_img=False,
            batch_size=128,
        )
        print(f'som time: {time.time() - som_started_at:.3f}s, total parse: {time.time() - parse_started_at:.3f}s')

        return dino_labled_img, parsed_content_list
