
import ast
import numpy as np
import traceback
import os

def get_latest_model(source):
    try:
        files = list(os.walk(source))
        files = [os.path.join(source, f) for f in files[0][2] if f.endswith('model')]
        indices = [ast.literal_eval(f.split('_')[-1].split('.')[0]) for f in files]
        latest_idx = sorted(indices)[-1]
        load_file_ = files[0].split('/')[-1].split('_')[:-1] + [str(latest_idx)]
        load_file = '_'.join(load_file_) + '.model'
        load_file = os.path.join(source, load_file)
        meta_file = '_'.join(load_file_) + '.metastate'
        meta_file = os.path.join(source, meta_file)
        if not os.path.exists(meta_file):
            print(f"{meta_file} does not exist, will not attempt to load optimizer + scheduler")
            meta_file = ''
        return load_file, meta_file
    except Exception as e:
        print('Get models and states failed', e)
    return '', ''

def get_latest_model_pl(source, mode="last"):
    print("looking models from ", source)
    try:
        files = list(os.walk(source))
        if mode == "best":
            files = [os.path.join(source, f) for f in files[0][2] if (f.endswith('ckpt') and "last" not in f)]
            indices = [ast.literal_eval(f.split('=')[-1].split('.')[0]) for f in files]
        elif mode == "last":
            files = [os.path.join(source, f) for f in files[0][2] if f.startswith('last')]
            if len(files) == 1 and files[0] == "last.ckpt":
                indices = [0]
            else:
                indices = []
                for i, f in enumerate(files):
                    f_ = f.split('/')[-1].split('v')
                    if len(f_) == 1:
                        indices.append(i)
                    else:
                        indices.append(ast.literal_eval(f_[1].split('.')[0]))
        f = files[np.argmax(indices)]
        return f
    except Exception:
        print('Get models and states failed')
        print(traceback.format_exc())
    return ''
    