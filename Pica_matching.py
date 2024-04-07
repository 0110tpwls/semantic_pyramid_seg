import json
import pandas as pd
from tqdm import tqdm
import argparse
import os

def found_match(merged_df, search_q, search_a):
    match=merged_df[merged_df['question'].str.contains(search_q, case=False, na=False)]
    ret=[]
    for idx, row in match.iterrows():
        a=pd.DataFrame(row['answers']).value_counts().idxmax()
        if search_a == a[3]:
            ret.append(row['image_id_x'])
    assert len(ret)>0
    return ret

def main(pica_dir,output_dir):
    with open(os.path.join(pica_dir,'coco_annotations/OpenEnded_mscoco_val2014_questions.json'), 'r') as f:
        data1 = json.load(f)
        q1=data1['questions']
    with open(os.path.join(pica_dir,'coco_annotations/OpenEnded_mscoco_train2014_questions.json'), 'r') as f:
        data2 = json.load(f)
        q2=data2['questions']

    with open(os.path.join(pica_dir,'coco_annotations/mscoco_val2014_annotations.json'),'r') as f:
        data3= json.load(f)
        a1=data3['annotations']
    with open(os.path.join(pica_dir,'coco_annotations/mscoco_train2014_annotations.json'),'r') as f:
        data4= json.load(f)
        a2=data4['annotations']

    questions=pd.DataFrame(q1+q2)
    annotations=pd.DataFrame(a1+a2)

    merged_df = pd.merge(questions, annotations, on='question_id')
    
    combined_lst=[]
    with open(os.path.join(pica_dir,'output_saved/prompt_answer/coco14_promptC_gpt3_5GT_n16_repeat5_53.246136_CLIPImagequestion.json'),'r') as f:
        cap_data = json.load(f)

        for key,val in tqdm(cap_data.items()):
            questions=val[1].split('\n===\nQ: ')[1:-1]
            answers=[s.split('\n===\n')[0].split('\nA: ')[1].strip('\n') for s in questions]
            questions=[s.split('\n===\n')[0].split('\nA: ')[0] for s in questions]
            
            pair_list=[{"question":q, "answer":a, "mathcing_image_id":found_match(merged_df,q,a)} for q,a in zip(questions,answers)]
            
            combined_lst.append(pair_list)

    with open(output_dir,'w') as f:
        json.dump(combined_lst, f, indent=4)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Simplify COCO dataset masks.")
    parser.add_argument("--pica_dir", type=str ,default='/home/pc_ubuntu/llm_t2i/PICa/')
    parser.add_argument("--output_dir", type=str ,default='/home/pc_ubuntu/llm_t2i/output.json')
    args = parser.parse_args()
    
    main(args.pica_dir,args.output_dir)
