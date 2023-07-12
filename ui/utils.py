import os
import json
import shutil

WINDOW_TITLE = {'seg': 'Train (Segmentation)',
                'ood': 'Train (OOD Classifier)',
                'test': 'Test'}


DESCRIPTIONS = {'seg': 'TBA',
                'ood': 'TBA',
                'test': 'TBA'}

CB_OPTIONS = {'seg': [{'name': 'Epochs', 'options': ['100 (Default)', '200', '10']},
                    {'name': 'Loss Lambda', 'options': ['0.01 (Default)', '0.05', '0.1']}],
            'test': [{'name': 'YOLO Threshold', 'options': ['0.05 (Default)', '0.1', '0.2']},
                    {'name': 'OOD Threshold', 'options': ['87 (Default)', '95', '99']}]}

SCRIPT_PATH = {'seg': './shell/train_seg.sh',
               'ood': './shell/train_ood.sh',
               'test': './shell/infer_whole.sh'}


def editFile(file_path, find, replacement):
    file_type = 'shell' if file_path.endswith('.sh') else None
    
    if (file_type == 'shell') and (os.path.exists(file_path.replace('.sh', '_edit.sh'))):
        file_path = file_path.replace('.sh', '_edit.sh')

    with open(file_path) as f:
        s = f.read()
    s = s.replace(find, replacement)
    
    if (file_type == 'shell') and ('_edit.sh' not in file_path):
        file_path = file_path.replace('.sh', '_edit.sh')
    
    with open(file_path, "w") as f:
        f.write(s)
    return file_path

def createDataYaml(data_dir):
    dir_name = data_dir.split('/')[-1]
    yaml_path = os.path.join(data_dir, f'{dir_name}.yaml')
    shutil.copyfile("./yolov7/data/data_example.yaml", yaml_path)

    editFile(yaml_path, 
            'test: ./data_example/images/', 
            f'test: {data_dir}/images/')
    
    return yaml_path, dir_name

def load_json(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(os.path.join(dir_path, 'images'), exist_ok=True)
        os.makedirs(os.path.join(dir_path, 'yolov7_preds'), exist_ok=True)
        return []

    json_path = os.path.join(dir_path, 'yolov7_preds/yolov7_preds_filtered_refined.json')
    if not os.path.exists(json_path):
        json_path = os.path.join(dir_path, 'yolov7_preds/yolov7_preds_filtered.json')

    if os.path.exists(json_path):
        with open(json_path, 'rb') as file:
            json_dict = json.load(file)
    else: json_dict = []

    return json_dict

def save_json(contents, dir_path):
    json_path = os.path.join(dir_path, "yolov7_preds/yolov7_preds_filtered.json")
    with open(json_path, "w") as file:
        json.dump(contents, file)