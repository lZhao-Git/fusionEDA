from argparse import Namespace
import yaml
# from tokenizer.tokenizer import MolTranBertTokenizer
# from train_pubchem_light import LightningModule
# from fast_transformers.masking import LengthMask as LM
import pandas as pd
from rdkit import Chem
import pickle 
import torch

import os,sys
import dill
import json
import esm
from tqdm import tqdm
from collections import OrderedDict
import numpy as np

import pdb
import transformers
import datasets
import fire
from torch.utils.data import DataLoader
from datasets import load_dataset
from src.text_encoder import load_text_model, inference

def batch_split(data, batch_size=64):
    i = 0
    while i < len(data):
        yield data[i:min(i+batch_size, len(data))] 
        i += batch_size

def cal_cpd_feat(model, smiles, tokenizer, batch_size=64):
    model.eval()
    embeddings = []
    for batch in batch_split(smiles, batch_size=batch_size):
        batch_enc = tokenizer.batch_encode_plus(batch, padding=True, add_special_tokens=True) 
        idx, mask = torch.tensor(batch_enc['input_ids']), torch.tensor(batch_enc['attention_mask']) 
        with torch.no_grad():
            token_embeddings = model.blocks(model.tok_emb(idx), length_mask=LM(mask.sum(-1))) 
        input_mask_expanded = mask.unsqueeze(-1).expand(token_embeddings.size()).float() 
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)  
        embedding = sum_embeddings / sum_mask 
        embeddings.append(embedding.detach().cpu())
    return torch.cat(embeddings)

def encoder_cpd(inputfile,outfile):
    df = pd.read_csv(inputfile)
    smiles = df.id
    with open('Pretrained_MoLFormer/hparams.yaml', 'r') as f:
        config = Namespace(**yaml.safe_load(f))
    tokenizer = MolTranBertTokenizer('bert_vocab.txt')
    ckpt = 'Pretrained_MoLFormer/checkpoints/N-Step-Checkpoint_3_30000.ckpt'
    lm = LightningModule(config, tokenizer.vocab).load_from_checkpoint(ckpt, config=config, vocab=tokenizer.vocab)
    X = cal_cpd_feat(lm, smiles, tokenizer).numpy() 
    idx = df.idx
    embed_dict = {id_val: X[i] for i, id_val in enumerate(idx)}
    with open(outfile, 'wb') as f:
        pickle.dump(embed_dict, f)

def cal_prot_feat(data: pd.DataFrame) -> dict:
    """
    Calculate the protein features using the protein pre-trained model
    """
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    model = model.cuda()
    batch_converter = alphabet.get_batch_converter()
    model.eval()
    repr_layer = model.num_layers

    def seq_to_vecs(pid, seq, max_length=1022):
        data = [
            (pid, seq[:max_length]),
        ] 
        _, _, batch_tokens = batch_converter(data)
        batch_tokens = batch_tokens.to(device="cuda", non_blocking=True) 

        with torch.no_grad():
            results = model(batch_tokens, repr_layers=[repr_layer])
        token_representations = results["representations"][repr_layer] 
        sequence_representations = token_representations[0, 1:].mean(0) 
        # return sequence_representations.cpu().detach().numpy().reshape(-1)
        vec = sequence_representations.cpu().detach().numpy().reshape(-1)
        vec = np.asarray(vec, dtype=np.float32).copy()
        return vec

    prot_feat = {}
    prot_data = data[["idx", "sequence"]].drop_duplicates(subset=["idx"])
    for _, row in tqdm(prot_data.iterrows()): 
        pid, seq = row[0], row[1]
        prot_feat[pid] = seq_to_vecs(pid, seq)
    return prot_feat 

def encoder_gen(inputfile,outfile):
    df = pd.read_csv(inputfile)
    df = df[['idx','sequence']]
    df_feat = cal_prot_feat(df)  
    with open(outfile, "wb") as f:
        pickle.dump(df_feat, f)


def encoder_text(model_name = "./biobert-base-cased-v1.2",data_path = "./",outfile = "./dis_emb.pkl",batch_size=64,):
    device = "cuda:0"
    model, tokenizer = load_text_model(model_name)
    model.to(device)
    model.eval()
    # tokenizing data path
    tokenized_data_dir = os.path.join(data_path, "encoded_dis")
    # load data
    if not os.path.exists(tokenized_data_dir):
        def tokenize_data(datapoint):
            name = datapoint.get("name", "")
            definition = datapoint.get("definition", "")
            if definition is None or pd.isna(definition):
                definition = ""
            name = str(name).strip().strip('"')
            definition = str(definition).strip().strip('"')
            if definition == "":
                seq = f"Name: {name}."
            else:
                seq = f"Name: {name}. Definition: {definition}"

            tokenized = tokenizer(seq,truncation=True,padding=False,return_tensors=None,max_length=512,)
            tokenized.update(datapoint)
            tokenized["seq"] = seq
            return tokenized
        data = load_dataset("csv", data_files='../data/DIS.csv')
        data = data["train"].map(tokenize_data, num_proc=8)
        data.save_to_disk(tokenized_data_dir)
    data = datasets.load_from_disk(tokenized_data_dir)
    outputs = {"idx": data["idx"],"seq": data["seq"]}
    data = data.remove_columns(["idx","id","name","definition","seq"])

    # start encoding using protein model
    loader = DataLoader(data,batch_size=batch_size,
                        collate_fn=transformers.DataCollatorWithPadding(tokenizer, max_length=tokenizer.model_max_length,pad_to_multiple_of=8,return_tensors="pt", ),)

    embeddings = []
    for batch in tqdm(loader):
        # map batch components to cuda device
        batch = {k:v.to(device) for k,v in batch.items()}
        emb = inference(model, batch)
        emb = emb.cpu().numpy()
        embeddings.append(emb)
    outputs["embedding"] = np.concatenate(embeddings, axis=0)
    dis_emb = {int(idx): emb for idx, emb in zip(outputs["idx"], outputs["embedding"])}
    with open(os.path.join(outfile), "wb") as f:
        pickle.dump(dis_emb,f)


if __name__ == '__main__':
    # encoder_cpd('../data/CPD.csv','./cpd_emb.pkl')
    # encoder_gen('../data/GEN.csv','./gen_emb.pkl')
    encoder_text()





