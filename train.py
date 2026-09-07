from fusionEDA import EDAData, EDAGNN, EDAEval
# load dataset
EDAData = EDAData(data_folder = './fusionEDA/data')
EDAData.prepare_split(split = 'random', fold = 1,seed = 42)
EDAGNN = EDAGNN(data = EDAData, 
              device = 'cuda:2', 
              cpd_path = './multimodal/cpd_emb.pkl',
              dis_path = './multimodal/dis_emb.pkl',
              gen_path = './multimodal/gen_emb.pkl',
              pwy_path = './multimodal/pwy_emb.pkl',
              bp_path = './multimodal/bp_emb.pkl',
              mf_path = './multimodal/mf_emb.pkl',
              cc_path = './multimodal/cc_emb.pkl',
              )
# Initialize a new model
EDAGNN.model_initialize(n_hid = 256, 
                      n_inp = 256,
                      n_out = 256,
                      proto = False, 
                      proto_num = 3, 
                      attention = False, 
                      sim_measure = 'all_nodes_profile', 
                      agg_measure = 'rarity', 
                      num_walks = 200, 
                      walk_mode = 'bit',
                      path_length = 2, 
                      data_path='fusionEDA/data/random_42'
                      )
EDAGNN.pretrain(n_epoch = 1, 
               learning_rate = 1e-3,
               batch_size = 1024,  
               train_print_per_n = 10,
               save_name='fusionEDA/data/random_42')





