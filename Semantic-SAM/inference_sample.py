from semantic_sam import prepare_image, plot_results, build_semantic_sam, SemanticSamAutomaticMaskGenerator
import matplotlib.pyplot as plt
import numpy as np

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

original_image, input_image = prepare_image(image_pth='/home/jovyan/Semantic-SAM/examples/dog.jpg')
ssam=build_semantic_sam(model_type='L', ckpt='/home/jovyan/Semantic-SAM/swinl_only_sam_many2many.pth')

mask_generator = SemanticSamAutomaticMaskGenerator(ssam, level=[1])
masks = mask_generator.generate(input_image)

fig = plt.figure()
plt.imshow(original_image)
plt.savefig('Semantic-SAM/out/sample1.png')

new_img= np.zeros_like(original_image)
for channel in range(3):
    new_img[:, :, channel] = original_image[:, :, channel] * masks[0]['segmentation']
fig = plt.figure()
plt.imshow(new_img)
plt.savefig('Semantic-SAM/out/sample_mask.png')

# original_image, input_image = prepare_image(image_pth='examples/dog.jpg')  # change the image path to your image
# ssam=build_semantic_sam(model_type='L', ckpt='/home/jovyan/Semantic-SAM/swinl_only_sam_many2many.pth')
# mask_generator = SemanticSamAutomaticMaskGenerator(ssam, level=[1,3]) # model_type: 'L' / 'T', depends on your checkpoint
# # instance_mask_generator = SemanticSamAutomaticMaskGenerator(build_semantic_sam(model_type='L', ckpt='/home/jovyan/Semantic-SAM/swinl_only_sam_many2many.pth')) 
# masks = mask_generator.generate(input_image)
# plot_results(masks, original_image, save_path='out/')  # results and original images will be saved at save_path