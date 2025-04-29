import ast
import torch
import numpy as np
import sys, os, re
from PIL import Image
import cv2
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from sam2.build_sam import build_sam2_camera_predictor
from llm.qwen2_modeling import Qwen2

# Set device and precision
device = "cuda" if torch.cuda.is_available() else "cpu"
torch_type = torch.bfloat16  # use bfloat16 for efficiency
torch.autocast(device_type="cuda", dtype=torch_type).__enter__()
if torch.cuda.is_available() and torch.cuda.get_device_properties(0).major >= 8:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


class GroundedSam2Processor:
    """
    A class to run GroundingDINO + SAM2 on a sequence of RGB images.

    Attributes:
        grounding_processor: HuggingFace processor for GroundingDINO.
        grounding_model: HuggingFace model for zero-shot object detection.
        predictor: SAM2 camera predictor for mask tracking.
        llm: Optional LLM model for extracting objects from a text query.
    """

    def __init__(self,
                 model_name: str = "Qwen/Qwen2.5-7B-Instruct",
                 sam_checkpoint: str = "checkpoints/sam2_hiera_large.pt",
                 sam_cfg: str = "sam2_hiera_l.yaml",
                 grounding_model_id: str = "gdino_checkpoints/grounding-dino-base"):
        """
        Initializes and loads all models.

        Args:
            model_name: Name/path of the LLM checkpoint if use_llm=True.
            use_llm: Whether to use LLM for extracting object text.
            sam_checkpoint: Path to SAM2 checkpoint.
            sam_cfg: Path to SAM2 config file.
            grounding_model_id: HuggingFace ID or path for GroundingDINO.
        """
        # # Load Grounding DINO
        self.grounding_processor = AutoProcessor.from_pretrained(os.path.join(sys.path[0], grounding_model_id))
        self.grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(os.path.join(sys.path[0], grounding_model_id)).to(device)

        # Load SAM2 predictor
        self.predictor = build_sam2_camera_predictor(sam_cfg, os.path.join(sys.path[0], sam_checkpoint), device=device)

        # Load LLM
        self.llm = Qwen2(os.path.join(sys.path[0], f"llm_checkpoints/{model_name}"), device=device)

    def extract_objects(self, query: str, robot_detail="a black robot gripper. ") -> str:
        """
        Uses the LLM to extract object phrases from a high-level query.

        Args:
            query: Instruction text, e.g. "put the banana to the yellow plate".

        Returns:
            Extracted object text, e.g. "a banana. a yellow plate."
        """
        template = open(os.path.join(sys.path[0], "llm/openie.txt"), "r").read()
        raw = self.llm.generate(template.format_map({"query": query}))
        parsed = ast.literal_eval(raw)
        
        self.obj_texts = robot_detail + parsed.get("query", "")
        self.obj_texts_list = self.obj_texts.split('.')[:-1]
        self.obj_texts_len = len(self.obj_texts_list)
        print("[DEBUG] LLM object extraction:", self.obj_texts, self.obj_texts_list, self.obj_texts_len)
        
    def manual_bbox_annotation(self, rgb_img: np.ndarray, obj_names: list[str]) -> tuple[np.ndarray, list[str]]:
        """
        Interactive drawing of one axis-aligned bounding box per object.

        Parameters
        ----------
        rgb_img   : HxWx3 uint8 RGB image
        obj_names : list of object labels in the order you want the user to draw

        Returns
        -------
        boxes  : (N, 4) float32   in (x1, y1, x2, y2) pixel coordinates
        labels : list[str]        same length as boxes
        """
        bgr = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)
        boxes   = []
        winname = "Draw object bounding box – press ENTER when done"

        for obj in obj_names:
            tmp = bgr.copy()
            print(winname)
            cv2.putText(tmp, f"Draw box for:  {obj}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            roi = cv2.selectROI(tmp)
            cv2.destroyAllWindows()

            if roi == (0, 0, 0, 0):          # user pressed ESC
                raise RuntimeError("Annotation aborted by user")

            x, y, w, h = roi                 # convert to x1,y1,x2,y2
            boxes.append((x, y, x + w, y + h))

        return torch.as_tensor(boxes, dtype=torch.bfloat16, device="cuda"), obj_names

    def segment_objects(self,
                         img,
                         idx: int,
                         box_threshold: float = 0.5,
                         text_threshold: float = 0.5):
        """
        Runs GroundingDINO to detect boxes on the first frame and SAM2 to track masks.

        Args:
            img: List of HxWx3 RGB numpy array.
            box_threshold: Minimum score for boxes.
            text_threshold: Minimum score for text alignment.

        Returns:
            List of dicts for each frame:
            {
                "boxes": numpy.ndarray of shape (N, 4),
                "masks": numpy.ndarray of shape (H, W, N)  # binary masks
            }
        """

        # Ensure img is HxWx3 RGB numpy array
        if idx == 0:
            object_all_found=False
            for i in range(1, 10):
                # GroundingDINO detection
                inputs = self.grounding_processor(
                    images=Image.fromarray(img),
                    text=self.obj_texts,
                    return_tensors="pt").to(device)
                with torch.no_grad():
                    outputs = self.grounding_model(**inputs)
                processed = self.grounding_processor.post_process_grounded_object_detection(
                    outputs,
                    inputs.input_ids,
                    box_threshold=box_threshold,
                    text_threshold=text_threshold,
                    target_sizes=[img.shape[:2]],
                )
                boxes = processed[0]["boxes"]
                labels = processed[0]["labels"]
                scores = processed[0]["scores"].cpu().numpy()

                # ────────────────────────────────────────────────────────────────────
                # helper ― string → token set
                # ────────────────────────────────────────────────────────────────────
                _token_re = re.compile(r"[^a-z0-9]+")

                def toks(s: str) -> set[str]:
                    return set(filter(None, _token_re.sub(" ", s.lower()).split()))

                obj_tokens  = [toks(o) for o in self.obj_texts_list]   # list[set]
                pred_tokens = [toks(p) for p in labels]                # list[set]

                # ────────────────────────────────────────────────────────────────────
                # 1. build all candidate matches  (obj_i, pred_j, overlap_size, score)
                # ────────────────────────────────────────────────────────────────────
                candidates = []
                for oi, o_set in enumerate(obj_tokens):
                    for pj, p_set in enumerate(pred_tokens):
                        overlap = o_set & p_set
                        if overlap:                                    # at least 1 common word
                            candidates.append((oi, pj,
                                                len(overlap),          # primary sort key
                                                scores[pj]))           # secondary  (float)

                # nothing overlaps at all → drop thresholds immediately
                if not candidates:
                    text_threshold -= 0.05
                    box_threshold  -= 0.05
                    print(f"[WARNING] no overlaps at all ⇒ lowering thresholds "
                        f"to {text_threshold:.3f}, preds: {labels}")
                    continue

                # sort: largest token-overlap first, then higher score
                candidates.sort(key=lambda t: (-t[2], -t[3]))

                # ────────────────────────────────────────────────────────────────────
                # 2. greedy assignment to guarantee one-to-one
                # ────────────────────────────────────────────────────────────────────
                chosen_for_obj = [-1] * len(obj_tokens)    # index of pred, or -1
                used_preds     = set()

                for oi, pj, _, _ in candidates:
                    if chosen_for_obj[oi] == -1 and pj not in used_preds:
                        chosen_for_obj[oi] = pj
                        used_preds.add(pj)

                missing = [i for i, pj in enumerate(chosen_for_obj) if pj == -1]

                # ────────────────────────────────────────────────────────────────────
                # 3. thresholds or accept
                # ────────────────────────────────────────────────────────────────────
                if missing:                                     # some object still unmatched
                    text_threshold -= 0.05
                    box_threshold  -= 0.05
                    misses = [self.obj_texts_list[i] for i in missing]
                    print(f"[WARNING] objects missing {misses} ⇒ lowering thresholds "
                        f"to {text_threshold:.3f}, preds: {labels}")
                    continue            # ← retry Grounding-DINO

                # every object has **exactly one** prediction now
                keep_idx = chosen_for_obj                       # same order as obj list

                # ────────────────────────────────────────────────────────────────────
                # 4. perfect → keep only the selected boxes, initialise SAM-2
                # ────────────────────────────────────────────────────────────────────
                # Initialize SAM2 predictor
                self.predictor.load_first_frame(img)

                for obj_label, pred_i in zip(self.obj_texts_list, keep_idx):
                    canon = "robot" if "robot" in obj_label.lower() else obj_label
                    self.predictor.add_new_points(frame_idx=idx,
                                                obj_id=canon,
                                                box=boxes[pred_i])

                obj_ids, mask_logits = self.predictor.track(img)
                object_all_found = True
                print("[DEBUG] Grounded-SAM2 obj pred:", obj_ids)
                break
            
            if not object_all_found:
                # return None
                # raise ValueError("[CRITICAL] Object can't be detected. Try another text prompt!")
                if not object_all_found:
                    print("[INFO] Automatic detection incomplete – "
                        "entering manual annotation mode …")

                    try:
                        boxes, labels = self.manual_bbox_annotation(img, self.obj_texts_list)
                    except RuntimeError:
                        print("[CRITICAL] Annotation cancelled")
                        return None

                    # 1) initialise SAM-2
                    self.predictor.load_first_frame(img)

                    # 2) add every hand-drawn box
                    for obj_label, box in zip(labels, boxes):
                        canon = "robot" if "robot" in obj_label.lower() else obj_label
                        self.predictor.add_new_points(frame_idx=idx,
                                                    obj_id=canon,
                                                    box=box)

                    # 3) track once on the first frame
                    obj_ids, mask_logits = self.predictor.track(img)
                    print("[DEBUG] SAM-2 obj ids (manual):", obj_ids)

        else:
            # Track masks on subsequent frames
            obj_ids, mask_logits = self.predictor.track(img)

        # Convert logits to binary masks
        logits = mask_logits.squeeze(1).cpu().numpy()  # shape (N, H, W)
        h, w = logits.shape[1], logits.shape[2]
        masks = (logits > 0)
        
        return {"masks": masks, "obj_labels":obj_ids}
