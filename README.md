#conference_2019/experimentManagement
   
1. cd directory src/experimentManagement/starter
2. then run following command

# Start TRAIN:
> PMF:
> demo_data
python entry_single.py --backend=pytorch --data_name=amazon_books --project=demo_task --train --model_name=pmf

> BPR:
> demo_data
python entry_single.py --backend=pytorch --data_name=amazon_books --project=demo_task --train --model_name=bpr