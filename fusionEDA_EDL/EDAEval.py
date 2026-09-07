from .utils import *
import pickle, os
class EDAEval:
    def __init__(self, model):
        self.df, self.df_train, self.df_valid, self.df_test, self.data_folder, self.G, self.best_model, self.weight_bias_track, self.wandb = model.df, model.df_train, model.df_valid, model.df_test, model.data_folder, model.G, model.best_model, model.weight_bias_track, model.wandb
        self.device = model.device
        self.disease_rel_types = ['rev_CPD_DIS_ass']
        self.split = model.split 
        
    def eval_disease_centric(self, disease_idxs, relation = None, save_result = False, show_plot = False, verbose = False, save_name = None, return_raw = False, simulate_random = True):
        if self.split == 'full_graph':
            # set only_prediction to True during full graph training
            only_prediction = True
        else:
            only_prediction = False
            
        if disease_idxs == 'test_set':
            disease_idxs = None
        
        self.out = disease_centric_evaluation(self.df, self.df_train, self.df_valid, self.df_test, self.data_folder, self.G, self.best_model,self.device, disease_idxs, relation, self.weight_bias_track, self.wandb, show_plot, verbose, return_raw, simulate_random, only_prediction)
        if save_result:
            import pickle, os
            if save_name is None: 
                save_name = os.path.join(self.data_folder, 'disease_centric_eval.pkl')
            os.makedirs(save_name,exist_ok=True)
            with open(os.path.join(save_name,'disease_centric_eval.pkl'), 'wb') as f:
                pickle.dump(self.out, f)
        return self.out

    def pred_test(self,pkl_dir = None,save_path = None):
        etypes_dd = [
                    ('compound', 'CPD_DIS_ass', 'disease'), 
                    ('disease', 'rev_CPD_DIS_ass', 'compound')
                    ]  
        with open(os.path.join(pkl_dir,'g_test_pos.pkl'), 'rb') as f:
            g_test_pos = pickle.load(f)
            g_test_pos = g_test_pos.to(self.device)
        with open(os.path.join(pkl_dir,'g_test_neg.pkl'), 'rb') as f:
            g_test_neg = pickle.load(f)
            g_test_neg = g_test_neg.to(self.device)       
        G = self.G.to(self.device)
        model = self.best_model
        model.eval()
        with torch.no_grad():
            pred_score_neg, score_neg = model(G,g_test_neg, pretrain_mode = False, mode = 'pred')
            pred_score_pos, score_pos = model(G,g_test_pos, pretrain_mode = False, mode = 'pred')

        results_df = []
        for etype in etypes_dd:
            u, v = g_test_pos.edges(etype=etype)
            logit = pred_score_pos[etype]
            pred = torch.sigmoid(logit).detach().cpu().numpy()
            u = u.detach().cpu().numpy()
            v = v.detach().cpu().numpy()
            temp_df = pd.DataFrame({'relation': [str(etype[1])] * pred.shape[0],
                                'x_idx': u,'y_idx': v,'prob': pred,'label': np.ones(pred.shape[0], dtype=int),})
            results_df.append(temp_df)

            u, v = g_test_neg.edges(etype=etype)
            logit = pred_score_neg[etype]
            pred = torch.sigmoid(logit).detach().cpu().numpy()
            u = u.detach().cpu().numpy()
            v = v.detach().cpu().numpy()
            temp_df = pd.DataFrame({'relation': [str(etype[1])] * pred.shape[0],
                                'x_idx': u,'y_idx': v,'prob': pred,'label': np.zeros(pred.shape[0], dtype=int), })
            results_df.append(temp_df)
        results_df=pd.concat(results_df)
        os.makedirs(save_path,exist_ok=True)
        results_df.to_csv(os.path.join(save_path, 'pred_test.csv'),index=False)
     
    def pred_disease_centric(self, disease_idxs, relation = None, save_result = False, show_plot = False, verbose = False, save_name = None, return_raw = False, simulate_random = True):
        if self.split == 'full_graph':
            # set only_prediction to True during full graph training
            only_prediction = True
        else:
            only_prediction = False
            
        if disease_idxs == 'test_set':
            disease_idxs = None
        
        self.out = disease_centric_prediction(self.df, self.df_train, self.df_valid, self.df_test, self.data_folder, self.G, self.best_model,self.device, disease_idxs, relation, self.weight_bias_track, self.wandb, show_plot, verbose, return_raw, simulate_random, only_prediction)
        if save_result:
            import pickle, os
            if save_name is None: 
                save_name = os.path.join(self.data_folder, 'disease_centric_pred.pkl')
            os.makedirs(save_name,exist_ok=True)
            with open(os.path.join(save_name,'disease_centric_pred.pkl'), 'wb') as f:
                pickle.dump(self.out, f)
        return self.out
    
    def retrieve_disease_idxs_test_set(self, relation):
        relation = 'rev_' + relation
        df_train_valid = pd.concat([self.df_train, self.df_valid])
        df_dd = self.df_test[self.df_test.relation.isin(self.disease_rel_types)]
        df_dd_train = df_train_valid[df_train_valid.relation.isin(self.disease_rel_types)]

        df_rel_dd = df_dd[df_dd.relation == relation]        
        return df_rel_dd.x_idx.unique()
    
    
    def retrieve_all_disease_idxs(self):
        return np.unique(self.df[self.df.x_type == 'disease'].x_idx.unique().tolist() + self.df[self.df.y_type == 'disease'].y_idx.unique().tolist())