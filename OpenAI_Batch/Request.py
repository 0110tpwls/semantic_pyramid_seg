from openai import OpenAI
from PIL import Image
import time
import json
import os
import base64
from io import BytesIO
import webdataset as wds
import numpy as np
from tqdm import tqdm
import argparse

api_key =os.getenv("OPENAI_API_KEY") # or sk-....."

# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# Function to encode the image
def encode_PIL_image(image):
    # Create a BytesIO buffer to hold the image data
    buffer = BytesIO()
    # Save the image to the buffer
    image.save(buffer, format=image.format)
    # Get the bytes value of the image
    image_bytes = buffer.getvalue()
    # Encode the bytes to base64
    return base64.b64encode(image_bytes).decode('utf-8')

def handle_webdataset(url='../data/0_images/imagenet_1k/sharded/0.tar'):
    def load_image(image):
        img = Image.fromarray(np.uint8(image * 255)).convert('RGB')
        return img

    def identity(x):
        return x

    dataset = (
        wds.WebDataset(url)
        .decode("rgb")
        .to_tuple("jpg;png", "json")
        .map_tuple(load_image, identity)
    )  
    return dataset

def make_gpt4v_format_request_file(webdataset_url, user_prompt_list ,output_file_name, model="gpt-4-turbo", img_res_mode='low', max_tokens=1500,sys_prompt='You are a helpful assistant.'):
    assert len(user_prompt_list) > 0, "User prompt list is empty."
    assert len(user_prompt_list) == len(webdataset_url), "User prompt list and webdataset url length mismatch."

    wbdb=handle_webdataset(webdataset_url)
    
    request_template = {
        "custom_id": "request-",
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": model,
            "messages": [{"role": "system", "content": sys_prompt}, {}],
            "max_tokens": max_tokens
        }
    }

    # Write each query into the output file in the OpenAI JSONL format
    with open(output_file_name, 'w') as f:
        print(f"Writing the requests to {output_file_name}")
        for (img, row),user_prompt in tqdm(zip(wbdb, user_prompt_list), total=len(wbdb)):
            key=row['key']
            chat_message = {"role": "user", "content": [ {'type':'text', 'text': user_prompt},{'type':'image_url', 'image_url': {'url': f"data:image/jpeg;base64,{encode_image(img)}",'detail': img_res_mode}}]}
            request_template["custom_id"] = f"request-{key}"
            request_template["body"]["messages"][-1] = chat_message
            f.write(json.dumps(request_template) + '\n')

    return output_file_name

def batch_request(input_file_path, api_key, batch_description):
    # Initialize the OpenAI API
    client = OpenAI(api_key=api_key)
    
    # Create a list to store the file IDs of the uploaded images
    batch_input_file= client.files.create(file=open(input_file_path, "rb"), purpose="batch")
    
    requested_batch = client.batches.create(
            input_file_id=batch_input_file.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
            metadata={"description": batch_description}
        )
    
    print(f"Batch job posted with ID: {requested_batch.id}")
    return requested_batch, requested_batch.id

def main(args):
    # Define the path to the WebDataset file
    webdataset_url=args.webdataset_url
    
    # Define the user prompts
    # 여기는 watdatset길이에 맞게 내용 넣기
    user_prompt_list = [
        # "What is the color of the car?",
        # "What is the breed of the dog?",
        # "What is the color of the cat?",
        # "What is the color of the bird?"
    ]
    
    # Define the output file name
    request_file_name = args.output_file
    
    # Define the system prompt
    sys_prompt = args.output_file
    
    # Generate the OpenAI JSONL format request file
    make_gpt4v_format_request_file(webdataset_url, user_prompt_list, request_file_name,model=args.model, img_res_mode=args.resolution, max_tokens=args.max_token, sys_prompt=sys_prompt)
    
    # Define the batch description
    batch_description = f"Batch request at time: { time.strftime('%Y-%m-%d %H:%M:%S') }"
    
    # Send the batch request to the OpenAI API
    batch_request(request_file_name, api_key, batch_description)

if __name__ == "__main__":
    #parse system arguments
    args = argparse.ArgumentParser()
    args.add_argument('--webdataset_url', type=str, default='../data/0_images/imagenet_1k/sharded/0.tar', help='path to the webdataset file')
    args.add_argument('--output_file', type=str, default='requests.jsonl', help='output file name')
    args.add_argument('--model', type=str, default='gpt-4o', help='model name')
    args.add_argument('--resolution', type=str, default='low', help='image resolution mode')
    args.add_argument('--max_token', type=int, default=1500, help='maximum tokens')
    args = args.parse_args()
    main(args)



