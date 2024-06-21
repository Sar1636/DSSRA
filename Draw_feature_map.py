import matplotlib.pyplot as plt
import numpy as np
import torch

from tqdm import tqdm

import spectral
from utils import find_max
from sklearn.preprocessing import scale
import copy

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    if torch.cuda.device_count() > 1:
        device = torch.device('cuda:0')
    else:
        device = torch.device('cuda')
else:
    device = torch.device('cpu')



def draw_feature_map(handle, model = None, dataObj = None, groundtruth = None, iscomposite = False, input_val = None):
    
    color_dict = {'ksc': [[0, 0, 0], [140, 67, 46], [0, 0, 255], [255, 100, 0], [0, 255, 123], 
                          [164, 75, 155], [101, 174, 255], [118, 254, 172], [60, 91, 112], 
                          [255, 255, 0], [101, 193, 60], [255, 0, 255], [100, 0, 255], [0, 172, 254]],

                  'pavia_U': [[0, 0, 0], [192, 192, 192], [0, 255, 0], [0, 255, 255], [0, 128, 0],
                              [255, 0, 255], [165, 82, 41], [128, 0, 128], [255, 0, 0], [255, 255, 0]],

                  'pavia_C': [[0, 0, 0], [192, 192, 192], [0, 255, 0], [0, 255, 255], [0, 128, 0],
                              [255, 0, 255], [165, 82, 41], [128, 0, 128], [255, 0, 0], [255, 255, 0]],
                              

                   'botswana': [[0, 0, 0], [0, 0, 254], [0, 255, 0],[255,0,255], [32,137,32], [151,47,255], 
                                [205,153,29], [255, 246, 140], [124, 131, 131], [34, 175, 165], [150, 23, 128], 
                                [211, 107, 26], [0, 255, 255], [252, 0, 6], [255, 255, 0]],

                  'dioni': [[0, 0, 0], [242,15,25], [219,55,116], [230,237,47], [235,126,50],
                            [231,180,77], [18,73,23], [114,123,49], [188,174,71], 
                            [213,235,104], [174,183,187], [53,78,212], [119,229,255]]}
    
    tick_dict = {'pavia_U': ('Background', 'Asphalt', 'Meadows', 'Gravel', 'Trees',
                             'Metal sheets', 'Bare soil', 'Bitumen','Self-Blocking Bricks', 'Shadows'),
                             
                 'pavia_C': ('Background', 'Water', 'Trees', 'Asphalt', 'Self-Blocking Bricks',
                             'Bitumen', 'Tiles', 'Shadows', 'Meadows', 'Bare Soils'),

                 'botswana': ('Background','water', 'hippo_grass', 'foodplain_grasses_1', 'foodplain_grasses_2', 'reeds1',
                             'riparain', 'firescar2', 'island_interior', 'acacia_woodlands',
                             'acacia_shrublands', 'acacia_grasslands', 'short_mopane', 'mixed_mopane','exposed_soils'),

                 'dioni':    ('Background','Dense_Urban_Fabric', 'Mineral_Extraction_Sites', 'Non_Irrigated_Arable_Land', 
                            'Fruit_Trees', 'Olive_Groves', 'Coniferous_Forest', 'Dense_Sderophyllous_Vegeation', 
                            'Sparce_Sderophyllous_Vegeation','Sparcely_Vegetation_Areas', 'Rocks_and_Sand', 'Water', 
                            'Coastal_Water')}
    
    band_dict = {'pavia_U': (46, 27, 10), 'pavia_C': (46, 27, 10),
                 'botswana': (75, 33, 15),'dioni': (43, 21, 11)}
    
    if iscomposite:
        band = band_dict[handle]
        image = np.array([input_val[:, :, band[0]], input_val[:, :, band[1]], input_val[:, :, band[2]]]).transpose(1, 2, 0)
        plt.figure(figsize=(11, 11))
        spectral.imshow(image)
        plt.axis('off')
        
        plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
        plt.margins(0,0)
        plt.savefig("results\\" + handle+'_composite', dpi = 600)

    if groundtruth is not None:
        color = color_dict[handle]
        color = np.array(color)
        spectral.imshow(classes = groundtruth.astype(int), figsize = (11, 11), colors = color)

        if iscomposite:
            tick = tick_dict[handle]
            bar = plt.colorbar()
            bar.set_ticks(np.linspace(0, len(tick) - 1, len(tick)))
            bar.set_ticklabels(tick)
        plt.axis('off')
        plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
        plt.margins(0,0)
        plt.savefig("results\\"+handle+'_gt', dpi = 600)
    else:
        color = color_dict[handle]
        temp = copy.deepcopy(dataObj.data)
        temp = temp.reshape((-1, temp.shape[-1]))
        temp = scale(temp)
        info = temp.reshape((dataObj.data.shape[0], dataObj.data.shape[1], -1))
        paddingData = np.zeros((info.shape[0] + 2 * dataObj.hwz + 1, info.shape[1] + 2 * dataObj.hwz + 1, info.shape[2]))
        paddingData[dataObj.hwz: info.shape[0] + dataObj.hwz, dataObj.hwz: info.shape[1] + dataObj.hwz] = info[:, :]
        y_out = np.zeros_like(info)
        y_out = np.zeros_like(dataObj.groundtruth)
        try:
            with tqdm(range(dataObj.groundtruth.shape[0])) as t:
                for x in t:
                    for y in range(dataObj.groundtruth.shape[1]):
                        if dataObj.groundtruth[x, y] == 0:
                            continue
                        X_in = torch.from_numpy(paddingData[x: x + 2 * dataObj.hwz + 1, y : y + 2 * dataObj.hwz + 1, :].transpose(2,0,1).astype(np.float32)).to(device)
                        X_in = X_in.reshape(1, 1, X_in.shape[0], X_in.shape[1], X_in.shape[2])
                        y_temp = model(X_in)
                        y_out[x,y] = int(find_max(y_temp)) + 1
                spectral.imshow(classes = y_out, colors = color, figsize=(10, 10))
                plt.axis('off')
                plt.savefig("results\\" +handle+'_image', dpi = 600)
        except KeyboardInterrupt:
            t.close()
            raise
        t.close()
        pass
