import sys
sys.path.append("/home/jovyan/Semantic-SAM")
import os
os.chdir('/home/jovyan/Semantic-SAM')
from semantic_sam import prepare_image, plot_results, build_semantic_sam, SemanticSamAutomaticMaskGenerator
import matplotlib.pyplot as plt
import numpy as np
import torch
import argparse

device = torch.device(f'cuda' if torch.cuda.is_available() else 'cpu')
print(device)

def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)
    polygons = []
    color = []
    for ann in sorted_anns:
        m = ann['segmentation']
        img = np.ones((m.shape[0], m.shape[1], 3))
        color_mask = np.random.random((1, 3)).tolist()[0]
        for i in range(3):
            img[:,:,i] = color_mask[i]
        ax.imshow(np.dstack((img, m*0.35)))

def generate_mask(img_path, mask_generator, save_path,lv):
    original_image, input_image = prepare_image(image_pth=img_path)
    masks = mask_generator.generate(input_image)
    
    fig = plt.figure()
    plt.imshow(original_image)
    plt.axis('off')
    plt.savefig(os.path.join(save_path,img_path.split('/')[-1]),bbox_inches='tight', pad_inches=0)

    sorted_masks = sorted(masks, key=(lambda x: x['area']), reverse=True)
    for idx, ann in enumerate(sorted_masks[:5]):
        fig = plt.figure()
        new_img= np.zeros_like(original_image)
        for channel in range(3):
            new_img[:, :, channel] = original_image[:, :, channel] * ann['segmentation']
        fig = plt.figure()
        plt.imshow(new_img)
        plt.axis('off')
        out_name=img_path.split('/')[-1][:-4]
        plt.savefig(os.path.join(save_path,f'{out_name}_lv{lv}_{idx}.png'),bbox_inches='tight', pad_inches=0)

def main(args):
    ssam=build_semantic_sam(model_type='L', ckpt=args.ckpt_dir)

    mask_generator_lv1 = SemanticSamAutomaticMaskGenerator(ssam, level=[1])
    mask_generator_lv3 = SemanticSamAutomaticMaskGenerator(ssam, level=[3])
    
    os.makedirs(f"{args.image_dir}/masks/lv1", exist_ok=True)
    os.makedirs(f"{args.image_dir}/masks/lv3", exist_ok=True)
    img_list=os.listdir(args.image_dir)
    img_list=[os.path.join(args.image_dir,i) for i in img_list if 'jpg' in i or 'png' in i]
    for img in img_list:
        print(img)
        generate_mask(img,mask_generator_lv1,f'{args.image_dir}/masks/lv1',lv= 1)
        generate_mask(img,mask_generator_lv3,f'{args.image_dir}/masks/lv3',lv= 3)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", type=str, default=None)
    parser.add_argument("--ckpt_dir", type=str, default=None)
    args = parser.parse_args()
    main(args)

# python inference_sample.py --image_dir '/home/jovyan/tmpMask' --ckpt_dir '/home/jovyan/Semantic-SAM/swinl_only_sam_many2many.pth'