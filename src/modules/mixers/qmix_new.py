import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def Load_MyQmix(args):
    if args.exp == 1:
        from modules.mixers.qmix_new_1 import QMixer_new
        return QMixer_new(args)
    elif args.exp == 2:
        from modules.mixers.qmix_new_2 import QMixer_new
        return QMixer_new(args)
    elif args.exp == 3:
        from modules.mixers.qmix_new_3 import QMixer_new
        return QMixer_new(args)
    elif args.exp == 4:
        from modules.mixers.qmix_new_4 import QMixer_new
        return QMixer_new(args)
    elif args.exp == 5:
        from modules.mixers.qmix_new_5 import QMixer_new
        return QMixer_new(args)
    elif args.exp == 6:
        from modules.mixers.qmix_new_6 import QMixer_new
        return QMixer_new(args)
    elif args.exp == 7:
        from modules.mixers.qmix_new_7 import QMixer_new
        return QMixer_new(args)
    elif args.exp == 8:
        from modules.mixers.qmix_new_8 import QMixer_new
        return QMixer_new(args)
    elif args.exp == 9:
        from modules.mixers.qmix_new_9 import QMixer_new
        return QMixer_new(args)
    elif args.exp == 10:
        from modules.mixers.qmix_new_10 import QMixer_new
        return QMixer_new(args)    
