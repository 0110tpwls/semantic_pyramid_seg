from openai import OpenAI
from PIL import Image
from datasets import load_dataset
import requests
import json
import os
import time
import base64
import argparse

class OpenAIBatchInference:
    """
    A class to facilitate batch inference operations using OpenAI's API.

    Attributes:
        api_key (str): The API key for accessing OpenAI services.
        client (OpenAI): The OpenAI client initialized with the provided API key.
    """

    def __init__(self, api_key: str):
        """
        Initialize an instance of OpenAIBatchInference.

        Args:
            api_key (str): The API key for accessing OpenAI services.
        """
        self.api_key = api_key
        self.client = OpenAI(api_key=self.api_key)

    def batch_upload(self, file_path: str):
        """
        Upload a file for batch processing.

        Args:
            file_path (str): The path to the input file to be uploaded.

        Returns:
            dict: The uploaded file object containing metadata like file ID.
        """
        return self.client.files.create(
            file=open(file_path, "rb"),
            purpose="batch"
        )

    def batch_create(self, batch_input_file, description: str = "nightly eval job"):
        """
        Create a new batch job using the uploaded input file.

        Args:
            batch_input_file (dict): The file object returned from the batch upload process.
            description (str): A description of the batch job.

        Returns:
            dict: The batch job object containing metadata like job ID and status.
        """
        return self.client.batches.create(
            input_file_id=batch_input_file.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
            metadata={"description": description}
        )

    def batch_request(self, file_path: str, request_name: str = "nightly eval job"):
        """
        Upload a file and create a batch job in one step.

        Args:
            file_path (str): The path to the input file to be uploaded.
            request_name (str): The description of the batch job.

        Returns:
            dict: The batch job object created.
        """
        batch_input_file = self.batch_upload(file_path)
        return self.batch_create(batch_input_file, request_name)

    def batch_status(self, batch_id: str):
        """
        Retrieve the status of a specific batch job.

        Args:
            batch_id (str): The ID of the batch job.

        Returns:
            dict: The batch job object containing its status and other metadata.
        """
        return self.client.batches.retrieve(batch_id)

    def get_batch_id(self, batch_obj):
        """
        Extract and return the batch ID from a batch object.

        Args:
            batch_obj (dict): The batch job object.

        Returns:
            str: The batch job ID.
        """
        return batch_obj.id

    def get_batch_status(self, batch_obj):
        """
        Retrieve the status of a batch job from the batch object.

        Args:
            batch_obj (dict): The batch job object.

        Returns:
            str: The status of the batch job.
        """
        return batch_obj.status

    def get_batch_output_file_id(self, batch_obj):
        """
        Retrieve the output file ID from a batch job object.

        Args:
            batch_obj (dict): The batch job object.

        Returns:
            str: The ID of the output file.
        """
        return batch_obj.output_file_id

    def batch_out_retrieve(self, output_file_id: str, output_file_path: str):
        """
        Download the output file of a completed batch job.

        Args:
            output_file_id (str): The ID of the output file to download.
            output_file_path (str): The path where the downloaded file will be saved.
        """
        url = f"https://api.openai.com/v1/files/{output_file_id}/content"
        headers = {"Authorization": f"Bearer {self.api_key}"}

        # Make the GET request to download the output file
        response = requests.get(url, headers=headers)

        if response.status_code == 200:
            # Write the downloaded content to the specified file
            with open(output_file_path, "wb") as file:
                file.write(response.content)
            print("File downloaded successfully.")
        else:
            print(f"Error: Unable to download the file (status code: {response.status_code}).")


def make_openai_format_request_file(query_list, output_file, model="gpt-3.5-turbo-0125", sys_prompt='You are a helpful assistant.'):
    """
    Create a file in OpenAI's JSONL format for making requests to the GPT-3.5 Turbo model.

    Args:
        query_list (list): A list of queries.
        output_file (str): Path to the output file where the requests will be written.
        model (str, optional): The model to use for completion. Defaults to "gpt-3.5-turbo-0125".
        sys_prompt (str, optional): The system prompt to use in the conversation. Defaults to 'You are a helpful assistant.'.

    Returns:
        str: The path to the output file created.
    """
    request_template = {
        "custom_id": "request-",
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": model,
            "messages": [{"role": "system", "content": sys_prompt}, {}],
            "max_tokens": 1500
        }
    }

    # Write each query into the output file in the OpenAI JSONL format
    with open(output_file, 'w') as f:
        for idx, query in enumerate(query_list):
            query_data = json.loads(query)
            request_template["custom_id"] = f"request-{idx}"
            request_template["body"]["messages"][-1] = {"role": "user", "content": query_data}
            f.write(json.dumps(request_template) + '\n')

    return output_file

# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def make_gpt4v_format_request_file(query_list, image_path_list, output_file, model="gpt-4-turbo", img_res_mode='low', sys_prompt='You are a helpful assistant.'):
    """
    Create a file in OpenAI's JSONL format for making requests to the GPT-3.5 Turbo model.

    Args:
        query_list (list): A list of queries.
        image_list (list): A list of local images for query.
        output_file (str): Path to the output file where the requests will be written.
        model (str, optional): The model to use for completion. Defaults to "gpt-4-turbo".
        img_res_mode (str, optional): The resolution mode of the image. Choose between 'low', 'high', 'auto'. Defaults to 'low'.
        sys_prompt (str, optional): The system prompt to use in the conversation. Defaults to 'You are a helpful assistant.'.

    Returns:
        str: The path to the output file created.
    """
    request_template = {
        "custom_id": "request-",
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": model,
            "messages": [{"role": "system", "content": sys_prompt}, {}],
            "max_tokens": 1500
        }
    }

    # Write each query into the output file in the OpenAI JSONL format
    with open(output_file, 'w') as f:
        for idx, query in enumerate(query_list):
            # query_data = json.loads(query) 
            chat_message = {"role": "user", "content": [ {'type':'text', 'text': query},{'type':'image_url', 'image_url': {'url': f"data:image/jpeg;base64,{encode_image(image_path_list[idx])}",'detail': img_res_mode}}]}
            request_template["custom_id"] = f"request-{idx}"
            request_template["body"]["messages"][-1] = chat_message
            f.write(json.dumps(request_template) + '\n')

    return output_file

def run_test(model, input_jsonl_path, output_jsonl_path):
    my_api_key = os.getenv("OPENAI_API_KEY")
    # Initialize the batch inferencer
    my_batch_inferencer = OpenAIBatchInference(api_key=my_api_key)
    
    #load dataset from local_dir and print the first 5 examples
    dataset = load_dataset("/home/jerry0110/sj_projects/llm_t2i/lmm_archive/data/0_images/okvqa_test/raw")

    #Prepare your question_list, img_path_list here
    question_list=[] #list of string for each query to gpt
    img_path_list=[] #list of image path corresponding to each query
    
    start_time=time.time()
    make_gpt4v_format_request_file(question_list, img_path_list,'sample_vqa.jsonl',model=model)

    # Create a new batch job
    requested_batch = my_batch_inferencer.batch_request('sample_vqa.jsonl')
    print(f"Batch job posted with ID: {requested_batch.id}")
    batch_id = requested_batch.id
    
    # Periodically check the status of the batch job
    for _ in range(50):
        batch_obj = my_batch_inferencer.batch_status(batch_id)
        print(f"Batch job status: {batch_obj.status}")
        if batch_obj.status == 'completed':
            # If the job is complete, download the output file
            my_batch_inferencer.batch_out_retrieve(batch_obj.output_file_id, 'output.jsonl')
            print("Batch job complete. Output saved to 'output.jsonl'.")
            print('it took {} seconds'.format(time.time()-start_time))
            break
        else:
            print("Batch job is still running.")
            time.sleep(60)
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run batch inference test")
    parser.add_argument('--model', type=str, default="gpt-4-turbo", help='Model to use for inference')
    parser.add_argument('--input_file_path', type=str, required=True, help='Path to save input JSONL file')
    parser.add_argument('--output_file_path', type=str, required=True, help='Path to save output JSONL file')
    args = parser.parse_args()
    
    print(f"Running test for {args.model} model")
    run_test(
        model=args.model,
        input_jsonl_path=args.input_file_path,
        output_jsonl_path=args.output_file_path
    )
