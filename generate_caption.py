import sys
sys.path.append("/home/jovyan/InternLM-XComposer/projects/ShareGPT4V")
import os
import json
from share4v.model.builder import load_pretrained_model
from share4v.mm_utils import get_model_name_from_path
from share4v.eval.run_share4v import eval_model,multi_eval
from PIL import Image
import argparse

def crop_and_overwrite_image(image_path, crop_area):
    mask_list=os.listdir(image_path)
    mask_list=[os.path.join(image_path.replace("samples-Copy1","samples"),i) for i in mask_list if 'jpg' in i or 'png' in i]
    for mask in mask_list:
        with Image.open(mask) as img:
            # Perform the crop operation
            cropped_img = img.crop(crop_area)
            
            # Save the cropped image back to the original path
            print(mask)
            cropped_img.save(mask,quality=100)

def make_img_caption_pair(mask_list, json_name):
    model_path = "Lin-Chen/ShareGPT4V-7B"
    
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path=model_path,
        model_base=None,
        model_name=get_model_name_from_path(model_path)
    )
    model_path = "Lin-Chen/ShareGPT4V-7B"
    prompt = "Make detailed caption of given image. Do not comment about the black background. Also your caption must be shorter than 40 words. Starts with 'Image of ...' "
    
    args = type('Args', (), {
        "model_path": model_path,
        "model_base": None,
        "model_name": get_model_name_from_path(model_path),
        "query": prompt,
        "conv_mode": None,
        "image_file_list": mask_list,
        "sep": ",",
        "temperature": 0,
        "top_p": None,
        "num_beams": 1,
        "max_new_tokens": 512
    })()
    out=multi_eval(args)
    output_dict={key:value for i, (key, value) in enumerate(zip(mask_list, out))}
    
    with open(json_name,'w') as f:
        json.dump(output_dict, f, ensure_ascii=False, indent=4)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_dir", type=str, default=None)
    parser.add_argument("--output_json_name", type=str, default=None)
    args = parser.parse_args()
    
    file_list=os.listdir(args.file_dir)
    file_list=[os.path.join(args.file_dir,i) for i in file_list if 'jpg' in i or 'png' in i]
    make_img_caption_pair(file_list, args.output_json_name)



# python generate_caption.py --file_dir '/home/jovyan/tmpMask' --output_json_name '/home/jovyan/tmpMask/pair.json'