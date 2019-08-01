import sys, os, shutil, argparse

#####################################################
#    Following is the probe of the code directory   #
#####################################################
cwd = os.getcwd() # get current work directory
experiment_management_dir = os.path.abspath(os.path.join(cwd, os.pardir))
src_dir = os.path.abspath(os.path.join(experiment_management_dir, os.pardir))
root_dir = os.path.abspath(os.path.join(src_dir, os.pardir))

print(experiment_management_dir)
print(src_dir)
print(root_dir)

# add following directories to environment path
sys.path.append(os.path.join(experiment_management_dir, 'public'))
sys.path.append(os.path.join(src_dir, 'modelSource/class/public'))

app_record_dir = os.path.join(root_dir, 'app')
app_log_dir = os.path.join(root_dir, 'log')
app_src_dir = os.path.join(root_dir, 'experiment_single/modelSource')
app_data_dir = os.path.join(root_dir, 'data')
app_out_dir = os.path.join(root_dir, 'out')

print(app_record_dir)
print(app_log_dir)
print(app_src_dir)
print(app_data_dir)
print(app_data_dir)

import warnings
warnings.simplefilter("ignore")

from ParserConf import ParserConf
from Evaluate import Evaluate
from AskGPU import AskGPU

# flag=0, means init, else not the first time
def updateIdx(flag, idx_path):
    if flag == 1:
        with open(idx_path) as f:
            idx = eval(f.readline()) + 1
            f.close()
    elif flag == 0:
        idx = 1
    with open(idx_path, 'w') as w:
        w.write('%d' % (idx))
        w.flush()
        w.close()
    return idx

def getIdx(idx_path):
    with open(idx_path) as f:
        idx = eval(f.readline())
        f.close()
    return idx
 
def isGoOn():
    print('Continue? Please input "y(yes)" or "n(no)":')
    x = raw_input()
    if x == 'yes' or x == 'y':
        pass
    if x == 'no' or x == 'n':
        sys.exit(0)

def isSwitch(flag, svr_name, host_gpu):
    if flag == 0:
        print(\
            'Max Free GPU is not Host, Switch to the %s with Maximum Free GPU? Please input "y(yes)" or "n(no)", "e(exit)":'\
            % svr_name)
        while(1):
            x = raw_input()
            if x == 'yes' or x == 'y':
                sys.exit(0)
            if x == 'no' or x == 'n':
                return host_gpu
            if x == 'exit' or x == 'e':
                sys.exit(0)
    if flag == 1:
        return host_gpu

# check the task and backend, set the environemnt path, app_conf is the conf of the model
def checkTask(backend, project):
    sys.path.append(os.path.join(app_src_dir, 'class/data/%s' % project))
    sys.path.append(os.path.join(app_src_dir, 'util/%s/%s' % (backend, project)))

def getBasicDirectory(data_name, model_name, backend, project):
    # if app + data_name directory and log + data_name doesn't exists, we will create them first.
    record_dir = os.path.join(app_record_dir, data_name)
    log_dir = os.path.join(app_log_dir, data_name)
    if not os.path.exists(record_dir):
        os.makedirs(record_dir)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # get idx_path, record_lock
    idx_path = os.path.join(app_record_dir, '%s/%s_idx' % (data_name, model_name))
    
    if os.path.exists(idx_path):
        idx = updateIdx(1, idx_path)
    else:
        idx = updateIdx(0, idx_path)
    
    # get model_path, config_path, config_backup_path, model_backup_path
    model_path = os.path.join(app_src_dir, 'class/model/%s/%s/%s.py' % (backend, project, model_name))
    config_path = os.path.join(app_src_dir, 'conf/%s/%s/%s.ini' % (project, data_name, model_name))
    
    print('\nconfig path:%s' % config_path)
    print('model path:%s\n' % model_path)

    config_backup_path = os.path.join(\
        app_log_dir, '%s/%s/%s_%d/_%d_%s.ini'%(project, data_name, model_name, idx, idx, model_name))
    model_backup_path = os.path.join(\
        app_log_dir, '%s/%s/%s_%d/_%d_%s.py'%(project, data_name, model_name, idx, idx, model_name))
    
    # we will train model based on the model_backup_path
    sys.path.append(os.path.join(app_log_dir, '%s/%s/%s_%d' % (project, data_name, model_name, idx)))
    # the model_backup_name will be returned to the one who called this function
    model_backup_name = '_%d_%s' % (idx, model_name)

    # create data_model log dirrectory
    model_log_dir = os.path.join(app_log_dir, '%s/%s/%s_%d' % (project, data_name, model_name, idx))
    if not os.path.exists(model_log_dir):
        os.makedirs(model_log_dir)
        print('create %s successfully' % model_log_dir)
    
    # copy latest model.py and model.ini to app_log
    shutil.copyfile(config_path, config_backup_path)
    shutil.copyfile(model_path, model_backup_path)
    print('\nconfig backup path:%s' % config_backup_path)
    print('model backup path:%s\n' % model_backup_path)

    return config_backup_path, model_backup_name, idx

def startTrain(data_name, model_name, backend, project):
    config_backup_path, model_backup_name, record_idx = getBasicDirectory(data_name, model_name, backend, project)
    
    platform_config = [app_log_dir, app_data_dir, app_out_dir]
    app_conf = ParserConf(config_backup_path)
    app_conf.parserConf(project, platform_config, data_name, model_name)

    askGPU = AskGPU()
    gpu_id, gpu_memory = askGPU.getHostMaxAvailableGPU()
    print('Selected GPU:%d, Available Memory:%d MiB' % (gpu_id, gpu_memory))

    print('\n%s Strart To Train, Record Id: %d %s' % ('='*20, record_idx, '='*20))

    # check the task and backend, set the environemnt path
    checkTask(backend, project)
    os.environ['CUDA_VISIBLE_DEVICES'] = '%d' % gpu_id

    exec('from %s import %s' % (model_backup_name, model_name))
    model = eval(model_name)
    model = model(app_conf)

    if backend == 'pytorch':
        import torch
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        #device = torch.device('cpu')
        model.setDevice(device)
        model.cuda()
    
    from DataUtil import DataUtil
    data = DataUtil(app_conf)
    evaluate = Evaluate(app_conf)

    exec('import %s_train as train' % app_conf.type)
    pre_model = ''
    first_terminal_flag = 1
    bias_epoch = 0
    train.start(app_conf, data, model, evaluate, record_idx, pre_model, bias_epoch, first_terminal_flag)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Welcome to Peijie\'s Experiment Platform Entry')
    parser.add_argument('--project', nargs='?', help='project name, like dual_learning')
    parser.add_argument('--data_name', nargs='?', help='data name, like amazon_books')
    parser.add_argument('--model_name', nargs='?', help='model name')
    parser.add_argument('--train', action='store_true', help='train flag')
    parser.add_argument('--backend', nargs='?', help='backend engine')

    args = parser.parse_args()

    data_name = args.data_name
    model_name = args.model_name
    backend = args.backend
    project = args.project

    if args.train:
        startTrain(data_name, model_name, backend, project)