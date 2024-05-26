from openai import OpenAI
import os
import wandb
import time
import sys
import requests

api_key =os.getenv("OPENAI_API_KEY") # or sk-....."

def get_batch_status(client, batch_id: str):
    
    batch_obj = client.batches.retrieve(batch_id)
    return batch_obj.status, batch_obj

def get_multiple_batch_status(client, after=None, limit=100):
    if after is None:
        batch_list = client.batches.list(limit=limit)
    else:
        batch_list = client.batches.list(limit=limit,after=after)  # limit is the number of batches to retrieve
    return batch_list

def batch_single_mointor(wandb_pj_name,batch_id: str):
    _, batch_obj = get_batch_status(batch_id)
    client = OpenAI(api_key=api_key)
    print(batch_obj)
    config={
        'id': batch_obj.id,
        'completion_window': batch_obj.completion_window,
        'metadata': batch_obj.metadata,
    }
    wandb.init(project=wandb_pj_name,config=config)
    while True:
        status, batch_obj = get_batch_status(client,batch_id)
        if status == 'completed':
            print("Batch is completed.")
            print(batch_obj)
            break
        wandb.log({'status': status})
        
        sys.stdout.write(f"\rIn Progress...{time.strftime('%Y-%m-%d %H:%M:%S')}")
        sys.stdout.flush()
        time.sleep(90)

def batch_multiple_mointor(wandb_pj_name, limit=100):
    init_batch_list = get_multiple_batch_status(client,limit=limit)
    already_completed_batches = [batch.id for batch in init_batch_list.data if batch.status == 'completed']
    client = OpenAI(api_key=api_key)
    wandb.init(project=wandb_pj_name)
    
    while True:
        batch_list = get_multiple_batch_status(client,limit=limit)
        for batch in batch_list.data and batch.id not in already_completed_batches:
            wandb.log({batch.id: batch.status})

        time.sleep(90)
        if all([batch.status == 'completed' for batch in batch_list.data]):
            print("All batches are completed.")
            break

def batch_download(batch_id: str, output_file_name: str):
    client = OpenAI(api_key=api_key)
    batch_obj = client.batches.retrieve(batch_id)
    assert batch_obj.status == 'completed', "Batch is not completed yet."
    
    output_file_id=batch_obj.output_file_id
    url = f"https://api.openai.com/v1/files/{output_file_id}/content"
    headers = {"Authorization": f"Bearer {api_key}"}

    # Make the GET request to download the output file
    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        # Write the downloaded content to the specified file
        with open(output_file_name, "wb") as file:
            file.write(response.content)
        print("File downloaded successfully.")
    else:
        print(f"Error: Unable to download the file (status code: {response.status_code}).")

if __name__ == "__main__":
    #for single batch monitoring + downloading
    batch_single_mointor(batch_id="...",wandb_pj_name="batch_monitor")
    batch_download(batch_id="...",output_file_name="...")
    
    #for multiple batch monitoring
    # batch_multiple_mointor("batch_monitor", 100)
