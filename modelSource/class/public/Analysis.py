'''
  ToolBox: Provide a list of analysis tools
'''
from collections import defaultdict
import numpy as np

class Analysis():
    def __init__(self):
        pass
    
    def sparseAnalysis(self, data_name, model_name, user_list, neigh_dict, hr_dict, ndcg_dict):
        save_path = '/home/sunpeijie/files/task/pyrec/special/dip/%s_%s_top5_sparsity' % \
            (data_name, model_name) 
        np.save(save_path, [user_list, neigh_dict, hr_dict, ndcg_dict])

        distribution_hr_dict = defaultdict(list)
        distribution_ndcg_dict = defaultdict(list)
        user_neigh_dict = defaultdict(set)
        print('test hr:%.4f' % np.sum(hr_dict.values()))
        final_hr_list = ''
        final_ndcg_list = ''
        user_length_list = ''
        num_neigh_range = [5, 10, 15, 20, 25, 30, 35, 40, 50, 60]
        print('user_list length:%d, max:%d, min:%d' % (len(user_list), max(user_list), min(user_list)))
        for u in user_list:
            num_neigh = neigh_dict[u]
            for idx, value in enumerate(num_neigh_range):
                if value > num_neigh:
                    distribution_hr_dict[idx].append(hr_dict[u])
                    distribution_ndcg_dict[idx].append(ndcg_dict[u])
                    user_neigh_dict[idx].add(u)
                    break
            if num_neigh >= num_neigh_range[-1]:
                distribution_hr_dict[len(num_neigh_range)].append(hr_dict[u])
                distribution_ndcg_dict[len(num_neigh_range)].append(ndcg_dict[u])
                user_neigh_dict[len(num_neigh_range)].add(u)
        for idx in range(len(num_neigh_range)+1):
            '''
            if idx == 0:
                x1, x2 = 0, 10
            elif idx == 9:
                x1, x2 = 250, 9999
            else:
                x1, x2 = num_neigh_range[idx-1], num_neigh_range[idx]
            print('[%d, %d]:average hr value:%.4f' % (x1, x2, np.mean(distribution_hr_dict[idx])))
            print('[%d, %d]:average ndcg value:%.4f' % (x1, x2, np.mean(distribution_ndcg_dict[idx])))
            '''
            user_length_list += '%d ' % len(user_neigh_dict[idx])
            final_hr_list += '%.4f ' % np.mean(distribution_hr_dict[idx])
            final_ndcg_list += '%.4f ' % np.mean(distribution_ndcg_dict[idx])
        print('final user list')
        print(user_length_list)
        print('final hr sparse analysis:')
        print(final_hr_list)
        print('final ndcg sparse analysis:')
        print(final_ndcg_list)