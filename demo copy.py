import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
import imageio
from PIL import Image
import ast
import asyncio
import time
import collections
import argparse

from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection 

from sam2.build_sam import build_sam2_camera_predictor
# from llm.gpt4o_modeling import GPT4o
from llm.qwen2_modeling import Qwen2
from utils import add_text_with_background


torch_type = torch.bfloat16 #bfloat16

# use bfloat16 for the entire notebook
torch.autocast(device_type="cuda", dtype=torch_type).__enter__()
if torch.cuda.get_device_properties(0).major >= 8:
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
device = "cuda" if torch.cuda.is_available() else "cpu"
print("device", device)

async def extract(query, model):
    with open("llm/openie.txt", "r") as file:
        ie_prompt = file.read()
    text = await model.generate(ie_prompt.format_map({"query": query}))
    text = ast.literal_eval(text)["query"]
    
    return text

async def extract_handler(query, queue, model):
    text = await extract(query, model)
    queue.append(text)

def load_model(model, use_llm=False):
    # init grounding dino model from huggingface
    # model_id = "IDEA-Research/grounding-dino-tiny"
    model_id = "gdino_checkpoints/grounding-dino-base"
    grounding_processor = AutoProcessor.from_pretrained(model_id)
    grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)

    # build sam2
    sam2_checkpoint = "checkpoints/sam2_hiera_base_plus.pt"
    # model_cfg = "sam2_configs/sam2_hiera_b+.yaml"
    model_cfg = "sam2_hiera_b+.yaml"
    predictor = build_sam2_camera_predictor(model_cfg, sam2_checkpoint, device=device)

    if use_llm:
        if 'gpt' in model.lower(): # "gpt-4o-2024-05-13"
            llm = GPT4o(model)
        elif 'qwen' in model.lower(): # "llm_checkpoints/Qwen2-7B-Instruct-AWQ"
            llm = Qwen2(f"llm_checkpoints/{model}", device=device)
        else:
            raise NotImplementedError("INVALID MODEL NAME")
    else:
        llm = None

    return grounding_processor, grounding_model, predictor, llm

async def main(model="gpt-4o-2024-05-13", use_llm=False):
    
    # load model
    grounding_processor, grounding_model, predictor, llm = load_model(model, use_llm)
    
    # load video
    # cap = cv2.VideoCapture(4) # camera
    # # Warm up camera
    # for _ in range(80):
    #     ret, frame = cap.read()
        
    cap = cv2.VideoCapture("/home/jokim/projects/openvla-oft/rollouts/2025_04_14/2025_04_14-21_20_57--openvla--episode=4--success=True--task=put_the_banana_to_the_yellow_plate.mp4")
    # cap = cv2.VideoCapture("notebooks/videos/case.mp4")
    # cap = cv2.VideoCapture("notebooks/videos/children_tracking_demo.mp4")
    
    # init
    query_queue = collections.deque([])
    response_queue = collections.deque([])
    if_init = False
    frame_list = [] # for visualization
    query = "put the banana to the yellow plate"
    text = "" #"a banana. a yellow plate."
    results = None

    box_color = (0, 255, 0)
    seg_color = (0, 0, 255)
    
    idx = 0
    # fps_cut = 2 # skip every fps_cut step to save time
    while True:  
        print(f"\rframe captured: {idx}, text: {text}", end='', flush=True)
        ret, frame = cap.read() # (480, 640, 3)
        if not ret:
            break
        # if idx % fps_cut == 0: continue
        
        if use_llm:
            # simulate query
            if idx == 0:
                query_queue.append(query)

            if query_queue:
                query = query_queue.popleft()
                asyncio.create_task(extract_handler(query, response_queue, llm))
            if response_queue:
                text = response_queue.popleft()
                print(f"LLM Response: {text}")
                if_init = False

        if text:
            width, height = frame.shape[:2][::-1]    
            if not if_init:
                predictor.load_first_frame(frame)
                cv2.imshow("first frame", frame)
                

                ann_frame_idx = 0  # the frame index we interact with
                ann_obj_id = 0  # give a unique id to each object we interact with (it can be any integers)
                
                # box from groundingDINO
                inputs = grounding_processor(images=Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)), text=text, return_tensors="pt").to(device)
                with torch.no_grad():
                    outputs = grounding_model(**inputs)
                results = grounding_processor.post_process_grounded_object_detection(
                    outputs,
                    inputs.input_ids,
                    box_threshold=0.5,
                    text_threshold=0.5,
                    target_sizes=[frame.shape[:2]]
                )
                print(results)
                
                # multiple box
                boxes = results[0]["boxes"]
                if boxes.shape[0] != 0:
                    for bbox in boxes:
                        _, out_obj_ids, out_mask_logits = predictor.add_new_points(
                            frame_idx=ann_frame_idx,
                            obj_id=ann_obj_id,
                            box=bbox,
                        )
                        # ann_frame_idx += 1
                        ann_obj_id += 1

                    if_init = True
                else:
                    if_init = False
                
                
                # continue

            else:
                out_obj_ids, out_mask_logits = predictor.track(frame)
                print(out_obj_ids)

                all_mask = np.zeros((height, width, 1), dtype=np.uint8)
                # print(all_mask.shape)
                for i in out_obj_ids:
                    out_mask = (out_mask_logits[i] > 0.0).permute(1, 2, 0).cpu().numpy().astype(
                        np.uint8
                    ) * 255

                    all_mask = cv2.bitwise_or(all_mask, out_mask)

                # print(all_mask.shape, type(all_mask))


                # all_mask = cv2.cvtColor(all_mask, cv2.COLOR_GRAY2BGR)
                all_mask = cv2.cvtColor(all_mask, cv2.COLOR_GRAY2RGB)
                frame = cv2.addWeighted(frame, 1, all_mask, 0.5, 0)
        
        if query:
            for bbox in boxes:
                x1, y1, x2, y2 = map(int, bbox)
                cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, thickness=2)

            frame = add_text_with_background(frame, query)

            cv2.imshow("frame", frame)
            cv2.imwrite(f"output/frame_{idx}.jpg", frame)
            idx += 1
            
        # Ensure tasks are running
        await asyncio.sleep(0)

        
        frame_list.append(frame)
        # result.write(frame)
        if cv2.waitKey(100) & 0xFF == ord("q"):
            break

    
    # visualization 
    frame_list = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in frame_list]
    gif = imageio.mimsave("./result.gif", frame_list, "GIF")
    # w, h = frame_list[0].shape[:2][::-1]
    # fps = cap.get(cv2.CAP_PROP_FPS)
    # # video_handler = cv2.VideoWriter("result.mp4", cv2.VideoWriter_fourcc(*"MP4V"),25,(w,h))
    # # video_handler = cv2.VideoWriter("result.mp4", cv2.VideoWriter_fourcc(*"XVID"),25,(w,h))
    # video_handler = cv2.VideoWriter("result.mp4", cv2.VideoWriter_fourcc('m', 'p', '4', 'v'),int(fps),(w,h))
    # # video_handler = cv2.VideoWriter("result.avi", cv2.VideoWriter_fourcc(*"XVID"),fps,(w,h))
    # for frame in frame_list:
    #     video_handler.write(frame)
    # video_handler.release()
    
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run the system with a specified model.')
    parser.add_argument('--model', default="Qwen2-7B-Instruct-AWQ", type=str, required=False, help='The llm to use for the system.')
    args = parser.parse_args()
    
    asyncio.run(main(args.model, use_llm=True))