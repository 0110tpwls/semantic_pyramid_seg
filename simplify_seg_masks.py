import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw
import json
from shapely.geometry import Polygon
from scipy.spatial import ConvexHull
import argparse
from tqdm import tqdm
from time import time

def rammer_douglas_peucker(mask_points, max_vertices=10, initial_tolerance=0.01):
    """
    Simplifies a polygon represented by mask_points using the Ramer-Douglas-Peucker algorithm
    to a reduced number of vertices, specified by max_vertices, with an initial tolerance.
    """
    polygon = Polygon(mask_points)
    tolerance = initial_tolerance
    simplified_polygon = polygon.simplify(tolerance=tolerance, preserve_topology=True)
    
    while len(simplified_polygon.exterior.coords) > max_vertices:
        tolerance += 0.1
        simplified_polygon = polygon.simplify(tolerance, preserve_topology=True)
        
    return list(simplified_polygon.exterior.coords)

def simplify_to_convex_hull(points, max_vertices):
    """
    Simplifies a set of points to a convex hull with at most max_vertices points.
    """
    points = np.array(points)
    hull = ConvexHull(points)
    hull_points = points[hull.vertices, :]
    
    while len(hull_points) > max_vertices:
        min_increase = None
        point_to_remove = -1
        
        for i in range(len(hull_points)):
            temp_points = np.delete(hull_points, i, axis=0)
            temp_hull = ConvexHull(temp_points)
            temp_perimeter = temp_hull.area
            
            if min_increase is None or temp_perimeter < min_increase:
                min_increase = temp_perimeter
                point_to_remove = i
        
        hull_points = np.delete(hull_points, point_to_remove, axis=0)
    
    final_hull = ConvexHull(hull_points)
    return hull_points[final_hull.vertices]

def process_coco_annotations(json_file, image_dir, output_dir, simplification_mode='convex', max_points=7, verbose=True):
    """
    Processes COCO annotations by overlaying simplified polygons on the images.
    """
    with open(json_file, 'r') as file:
        coco_data = json.load(file)
    
    input_polygons = []
    output_polygons = []
    
    for img_info in tqdm(coco_data['images']):
        image_path = f"{image_dir}/{img_info['file_name']}"
        image = Image.open(image_path)
        draw = ImageDraw.Draw(image) if verbose else None
        
        annotations = [ann for ann in coco_data['annotations'] if ann['image_id'] == img_info['id']]
        
        for ann in annotations:
            for segmentation in ann['segmentation']:
                polygon = [(segmentation[i], segmentation[i+1]) for i in range(0, len(segmentation), 2)]
                input_polygons.append(polygon)
                
                simplified_polygon = simplify_polygon(polygon, simplification_mode, max_points)
                
                if verbose:
                    draw.polygon(simplified_polygon, outline='blue')
                output_polygons.append(simplified_polygon)
        
        if verbose:
            image.save(f"{output_dir}/{img_info['file_name']}")
        
    return input_polygons, output_polygons

def simplify_polygon(polygon, mode, max_points):
    """
    Simplifies a polygon based on the specified mode and maximum number of points.
    """
    if len(polygon) < max_points:
        return polygon
    
    if mode == 'RDP':
        return rammer_douglas_peucker(polygon, max_vertices=max_points)
    elif mode in ['convex', 'convex_max']:
        points = np.array(polygon)
        simplified_points = simplify_to_convex_hull(points, max_vertices=max_points)
        return [tuple(point) for point in simplified_points]
    
def evaluate_masks(ground_truth_masks, model_masks):
    """
    Evaluates model masks against ground truth masks using Intersection over Union (IoU).
    """
    iou_list = []
    
    for gt_mask, model_mask in zip(ground_truth_masks, model_masks):
        try:
            gt_polygon = Polygon(gt_mask)
            model_polygon = Polygon(model_mask)
        except:
            continue
        
        if not gt_polygon.is_valid or not model_polygon.is_valid:
            continue
        
        intersection_area = gt_polygon.intersection(model_polygon).area
        union_area = gt_polygon.union(model_polygon).area
        iou = intersection_area / union_area
        iou_list.append(iou)
    
    average_iou = sum(iou_list) / len(iou_list) if iou_list else 0
    print(f"Average IoU: {average_iou}")
    print(f"Processed {len(iou_list)} masks out of {len(ground_truth_masks)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simplify COCO dataset masks.")
    parser.add_argument("--input_json", type=str ,default='/mnt/c/Users/PC/Downloads/coco17/annotations/instances_val2017.json')
    parser.add_argument("--input_img", type=str ,default='/mnt/c/Users/PC/Downloads/coco17/val2017')
    parser.add_argument("--output_dir", type=str ,default='/mnt/c/Users/PC/Downloads/coco17/eval2017')
    parser.add_argument("--mode", type=str, choices=['convex', 'RDP', 'convex_max'], default='convex')
    parser.add_argument("--max_points", type=int, default=12)
    
    args = parser.parse_args()
    
    start_time = time()
    ground_truth, simplified_masks = process_coco_annotations(args.input_json, args.input_img, args.output_dir, args.mode, args.max_points, verbose=False)
    end_time = time()
    
    print(f"Elapsed time: {end_time - start_time} seconds")
    evaluate_masks(ground_truth, simplified_masks)
