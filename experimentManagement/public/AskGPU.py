'''
    return the appropriate server and gpu
'''
import numpy as np
import GPUtil as gt

class AskGPU:
    def __init__(self):
        pass

    def arrangeGPUResources(self):
        gpus = gt.getGPUs()
        gpu_dict = {}
        idx = 0
        for idx, g in enumerate(gpus):
            gpu_dict[idx] = [g.memoryTotal - g.memoryUsed]
        return gpu_dict
    
    def getHostMaxAvailableGPU(self):
        gpu_free_list = []
        gpu_status_dict = self.arrangeGPUResources()
        for _, value in gpu_status_dict.items():
            gpu_free_list.extend(value)
        sort_index = np.argsort(gpu_free_list)
        return sort_index[-1], gpu_free_list[sort_index[-1]]