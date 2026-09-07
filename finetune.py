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

EDAGNN.load_pretrained('fusionEDA/data/random_42/pretrain_md')
EDAGNN.finetune(pkl_dir = 'fusionEDA/data/random_42',
               n_epoch = 500,  
               learning_rate = 1e-4,
               train_print_per_n = 20,
               valid_per_n = 1,
               save_name = 'fusionEDA/data/random_42/finetune',
              )


