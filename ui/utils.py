
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
