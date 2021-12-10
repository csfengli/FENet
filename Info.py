data_dirs = {'DTD': './data/dtd/',
             'MINC_2500': './data/minc-2500/',
             'GTOS-mobile': '../MFS/data/gtos-mobile/',
             'FMD': './data/FMD/',
             'KTH': './data/KTH_TIPS2_b/',
             'GTOS': './data/gtos/'}

#Number of classes in each dataset
num_classes = {'DTD': 47,
               'MINC_2500': 23,
               'GTOS-mobile': 31,
               'GTOS': 39,
               'FMD': 10,
               'KTH': 11,
               'uiuc': 25,
               'rawfoot':68}

#Number of runs and/or splits for each dataset
splits = {'DTD': 10,
          'MINC_2500': 5,
          'GTOS-mobile': 5,
          'GTOS': 5,
          'FMD': 10,
          'KTH': 10}

Datasets_Info = {'data_dirs': data_dirs, 'num_classes': num_classes, 'splits': splits}

backbone_outchannels = {'resnet18': 512, 'resnet50': 2048, 'resnet101': 2048, 'resnet152': 2048}

Models_Info = {'backbone_outchannels': backbone_outchannels}


import numpy as np
DTD_Class_names = np.array(['banded', 'blotchy', 'braided', 'bubbly', 'bumpy',
                        'chequered', 'cobwebbed', 'cracked', 'crosshatched',
                        'crystalline', 'dotted', 'fibrous', 'flecked', 'freckled',
                        'frilly', 'gauzy', 'grid', 'grooved', 'honeycombed',
                        'interlaced', 'knitted', 'lacelike', 'lined', 'marbled',
                        'matted', 'meshed', 'paisley', 'perforated', 'pitted',
                        'pleated', 'polka−dotted', 'porous', 'potholed', 'scaly',
                        'smeared', 'spiralled', 'sprinkled', 'stained', 'stratified',
                        'striped', 'studded', 'swirly', 'veined', 'waffled', 'woven', 'wrinkled', 'zigzagged'])
MINC_Class_names = np.array(['brick','carpet','ceramic','fabric','foliage','food',
                            'glass','hair','leather','metal','mirror','other',
                            'painted','paper','plastic','polishedstone','skin',
                            'sky','stone','tile', 'wallpaper', 'water', 'wood'])
GTOS_mobile_Class_names = np.array(['Painting','aluminum','asphalt','brick','cement','cloth','dry_leaf',
                        'glass','grass','large_limestone','leaf','metal_cover',
                        'moss', 'paint_cover','painting_turf','paper',
                        'pebble','plastic','plastic_cover','root','sand','sandPaper',
                        'shale','small_limestone','soil','steel','stone_asphalt',
                        'stone_brick','stone_cement','turf','wood_chips'])

FMD_Class_names = np.array(['fabric','foliage','glass','leather','metal','paper','plastic','stone','water','wood'])
KTH_Class_names = np.array(['aluminium_foil','brown_bread','corduroy','cork','cotton','cracker','lettuce_leaf','linen','white_bread','wood','wool'])


Class_names = {'DTD': DTD_Class_names,
               'MINC_2500': MINC_Class_names,
               'GTOS-mobile': GTOS_mobile_Class_names,
               'FMD':FMD_Class_names,
               'KTH':KTH_Class_names
               }

