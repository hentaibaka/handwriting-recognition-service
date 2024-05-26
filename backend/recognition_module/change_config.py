import yaml
import os
import pandas as pd
from easyocr.trainer.utils import AttrDict

class MyDumper(yaml.SafeDumper):
    def represent_str(self, data):
        return self.represent_scalar('tag:yaml.org,2002:str', data)

MyDumper.add_representer(str, MyDumper.represent_str)

def update_yaml_parameters(input_file, updates, output_file):

    with open(input_file, 'r', encoding='utf-8') as file:
        data = yaml.safe_load(file)

    # Список параметров, которые разрешено изменять
    allowed_keys = {'experiment_name', 
                    'select_data', 
                    'saved_model', 
                    'num_iter', 
                    'valInterval', 
                    'train_data', 
                    'valid_data', 
                    'path_save_model',
                    'new_prediction',
                    'batch_size',}

    # Обновление данных только для разрешенных параметров
    for key, value in updates.items():
        if key in allowed_keys:
            data[key] = value
            
    # Сохранение обновленных данных в новый YAML файл
    with open(output_file, 'w', encoding='utf-8') as file:
        yaml.dump(data, file, allow_unicode=True, Dumper=MyDumper)

def get_config(file_path):
    with open(file_path, 'r', encoding="utf8") as stream:
        opt = yaml.safe_load(stream)
    opt = AttrDict(opt)
    if opt.lang_char == 'None':
        characters = ''
        for data in opt['select_data'].split('-'):
            csv_path = os.path.join(opt['train_data'], data, 'labels.csv')
            df = pd.read_csv(csv_path, sep='^([^,]+),', engine='python', usecols=['filename', 'words'], keep_default_na=False)
            all_char = ''.join(df['words'])
            characters += ''.join(set(all_char))
        characters = sorted(set(characters))
        opt.character = ''.join(characters)
    else:
        opt.character = opt.number + opt.symbol + opt.lang_char
    return opt
