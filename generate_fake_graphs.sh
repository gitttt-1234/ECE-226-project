python inversion_gcn.py --gpu 0 --trial 1 --epoch 1500 --total_num --batch_size 64 --self_loop --bn_reg_scale 1e-2 --onehot_cof 1e-6 --dataset MUTAG --savepath './save/fakegraphs' --tmodel GCN --modelt GCN5_64 --path_t './save/teachers/Model:GCN5_64_Datset:MUTAG'
