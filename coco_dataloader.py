import json
from datasets import Dataset, DatasetDict
from PIL import Image
import os

# 定義 DataLoader 函數
def load_coco_dataset(json_path, image_dir):
    print("Loading COCO JSON file...")
    # 讀取 COCO JSON 文件
    with open(json_path, 'r') as f:
        coco_data = json.load(f)
    print("Loaded COCO JSON file successfully.")

    # 提取 annotations 和 images 信息
    annotations = coco_data['annotations']
    images = {img['id']: img for img in coco_data['images']}
    print(f"Found {len(images)} images and {len(annotations)} annotations.")

    # 構造數據列表
    dataset = []
    for i, ann in enumerate(annotations):
        if i % 100 == 0:
            print(f"Processing annotation {i}/{len(annotations)}...")
        img_info = images[ann['image_id']]
        
        # 獲取圖像相關信息
        img_path = os.path.join(image_dir, img_info['file_name'])
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            continue

        # 構造物件信息
        obj_info = {
            'id': ann['category_id'],
            'area': ann['area'],
            'bbox': ann['bbox'],  # 格式: [x_min, y_min, width, height]
            'category': coco_data['categories'][ann['category_id'] - 1]['name']
        }

        # 如果圖像已存在，附加對應的物件
        existing = next((item for item in dataset if item['image_id'] == img_info['id']), None)
        if existing:
            existing['objects']['id'].append(obj_info['id'])
            existing['objects']['area'].append(obj_info['area'])
            existing['objects']['bbox'].append(obj_info['bbox'])
            existing['objects']['category'].append(obj_info['category'])
        else:
            dataset.append({
                'image': image,
                'image_id': img_info['id'],
                'width': img_info['width'],
                'height': img_info['height'],
                'objects': {
                    'id': [obj_info['id']],
                    'area': [obj_info['area']],
                    'bbox': [obj_info['bbox']],
                    'category': [obj_info['category']]
                }
            })

    print("Formatting dataset...")
    # 使用 Hugging Face DatasetDict 格式化數據
    dataset = Dataset.from_list(dataset)
    print("Dataset formatted successfully.")
    return DatasetDict({'dataset': dataset})

# 示例用法
json_path = "./0115_T0_dataset_coco/result.json"
image_dir = "./0115_T0_dataset_coco/images"
print("Starting dataset loading...")
dataset = load_coco_dataset(json_path, image_dir)
print("Dataset loading complete.")

# 查看數據示例
print("Dataset structure:")
print(dataset)
print("Sample data:")
print(dataset['dataset'][6])
