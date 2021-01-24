import argparse
import os
import random
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle
from datasets import arxiv2
from transformers import GPT2Tokenizer, GPT2Model, GPT2LMHeadModel
from opt import OpenAIAdam
from text_utils import TextEncoder
from utils import (encode_dataset2, iter_data,
                   ResultLogger, make_path)
from loss import SummarizationLossCompute2
import pickle
from tensorboard_logger import configure, log_value



configure("./bio_gpt2_analysis", flush_secs=5)

def transform_roc(X1, X2, X3):
    n_batch = len(X1)
    xmb = np.zeros((n_batch, 2, n_ctx, 2), dtype=np.int32)
    mmb = np.zeros((n_batch, 2, n_ctx), dtype=np.float32)
    start = encoder['_start_']
    delimiter = encoder['_delimiter_']
    for i, (x1, x2, x3), in enumerate(zip(X1, X2, X3)):
        x12 = [start] + x1[:max_len] + [delimiter] + x2[:max_len] + [clf_token]
        x13 = [start] + x1[:max_len] + [delimiter] + x3[:max_len] + [clf_token]
        l12 = len(x12)
        l13 = len(x13)
        xmb[i, 0, :l12, 0] = x12
        xmb[i, 1, :l13, 0] = x13
        mmb[i, 0, :l12] = 1
        mmb[i, 1, :l13] = 1
    # Position information that is added to the input embeddings in the TransformerModel
    #xmb[:, :, :, 1] = np.arange(n_vocab + n_special, n_vocab + n_special + n_ctx)
    return xmb, mmb

##   use sinusoidal positional embeddings  
#    s1_tr, s2_tr, label_tr  = transform_arxiv(train_pair)
def transform_arxiv(X1,X2):
    n_batch = len(X1)
    delimiter = [encoder['<|TL;DR|>']] #[encoder['ĠTL']] + [encoder[';']] + [encoder['DR']]
    end_token = [encoder['<|endoftext|>']] #[encoder['<']] + [encoder['end']] + [encoder['>']]
    xmb = np.zeros((n_batch, n_ctx), dtype=np.int32)
    mmb = np.zeros((n_batch, n_ctx), dtype=np.float32)
    max_len = 256-1 #512-1  
    for i, (x1,x2), in enumerate(zip(X1,X2)):
        new_x1 = x1[:800] #[:400]
        new_x2 = x2[:200] #[:(n_ctx-400-6)]
        x12 = new_x1 + delimiter
        x13 = new_x2 + end_token #+ [clf_token]
        try:
            xmb[i,:len(x12)] = x12
            xmb[i,len(x12):len(x12)+len(x13)] = x13 
            mmb[i,:len(x12)] = 1
            mmb[i,:len(x12)+len(x13)] = 1
        except:
            import ipdb; ipdb.set_trace()
#    import ipdb; ipdb.set_trace()
    return xmb, mmb

def iter_apply(Xs, Ms):
    # fns = [lambda x: np.concatenate(x, 0), lambda x: float(np.sum(x))]
    logits = []
    cost = 0
    losses = []
    with torch.no_grad():
        dh_model.eval()
        for xmb, mmb in iter_data(Xs, Ms, n_batch=n_batch_train, truncate=False, verbose=True):
            n = len(xmb)
            XMB = torch.tensor(xmb, dtype=torch.long).to(device)
            MMB = torch.tensor(mmb).to(device)
            lm_logits,past = dh_model(XMB)
            loss = compute_loss_fct(lm_logits=lm_logits,lm_labels=XMB,encoder=text_encoder,only_return_losses=True)
            losses.append(float(loss.sum()))
    return np.sum(losses), np.mean(losses)


def iter_predict(Xs, Ms):
    logits = []
    with torch.no_grad():
        dh_model.eval()
        for xmb, mmb in iter_data(Xs, Ms, n_batch=n_batch_train, truncate=False, verbose=True):
            n = len(xmb)
            XMB = torch.tensor(xmb, dtype=torch.long).to(device)
            MMB = torch.tensor(mmb).to(device)
            _, clf_logits = dh_model(XMB)
            logits.append(clf_logits.to("cpu").numpy())
    logits = np.concatenate(logits, 0)
    return logits


def log(save_dir, desc,iter=0,save=''):
    global best_score
    print("Logging")
    tr_sum_loss, tr_mean_loss = iter_apply(trX[:n_valid], trM[:n_valid])
    va_sum_loss, va_mean_loss = iter_apply(vaX[:n_valid], vaM[:n_valid])
    log_value('va_sum_loss',va_sum_loss,n_updates)
    log_value('va_mean_loss',va_mean_loss,n_updates)
    logger.log(n_epochs=n_epochs, n_updates=n_updates, tr_cost=float(tr_sum_loss), va_cost=float(va_sum_loss), tr_acc=float(tr_mean_loss), va_acc=float(va_mean_loss))
    print('%d %d %.3f %.3f %.2f %.2f' % (n_epochs, n_updates, tr_sum_loss, va_sum_loss, tr_mean_loss, va_mean_loss))
    path = os.path.join(save_dir, desc, 'best_params_' + str(iter) + save)
    torch.save(dh_model.state_dict(), make_path(path))


def predict(dataset, submission_dir):
    filename = filenames[dataset]
    pred_fn = pred_fns[dataset]
    label_decoder = label_decoders[dataset]
    predictions = pred_fn(iter_predict(teX, teM))
    if label_decoder is not None:
        predictions = [label_decoder[prediction] for prediction in predictions]
    path = os.path.join(submission_dir, filename)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        f.write('{}\t{}\n'.format('index', 'prediction'))
        for i, prediction in enumerate(predictions):
            f.write('{}\t{}\n'.format(i, prediction))

def greedy(logits,XMB):
    text = []
    for i in range(len(XMB)):
        ind = torch.multinomial(logits[i,:].exp().data,1)
        token = text_encoder.decoder[int(ind[0])]
        text.append(token)
        print(text)
        import ipdb; ipdb.set_trace()
    return ind, token

def run_epoch(iter):
    losses = []
    #print('check batch: ' + str(n_batch_train))
    i = 0
    for xmb, mmb in iter_data(*shuffle(trX, trM, random_state=np.random),
                                   n_batch=n_batch_train, truncate=True, verbose=True):
        global n_updates
        dh_model.train()
        XMB = torch.tensor(xmb, dtype=torch.long).to(device)
        MMB = torch.tensor(mmb).to(device)
        XMB = torch.tensor(xmb, dtype=torch.long).to(device)
        MMB = torch.tensor(mmb).to(device)
        lm_logits, past = dh_model(XMB)
        loss = compute_loss_fct(lm_logits=lm_logits, lm_labels=XMB, encoder=text_encoder, batch_num=n_updates, accum_steps=int(16/args.n_batch))
        print(loss)
        losses.append(loss)
        n_updates += 1
        if (n_updates + 1) % 10000 == 0: # and n_epochs == 0:
           log(save_dir, desc,iter,save='_'+str(n_updates))

        log_value('batch_train_loss',loss,n_updates)
        log_value('mean_train_loss',np.mean(losses),n_updates)
        log_value('total_train_loss',np.sum(losses),n_updates)
argmax = lambda x: np.argmax(x, 1)

pred_fns = {
    'rocstories': argmax,
}

filenames = {
    'rocstories': 'ROCStories.tsv',
}

label_decoders = {
    'rocstories': None,
}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--desc', type=str, default='model',help="Description")
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--log_dir', type=str, default='bio_log/')
    parser.add_argument('--save_dir', type=str, default='./bio_gpt2')
    parser.add_argument('--data_dir', type=str, default='/net/nfs2.corp/ca-data/finished_files_cat')
    parser.add_argument('--submission_dir', type=str, default='submission/')
    parser.add_argument('--submit', action='store_true')
    parser.add_argument('--analysis', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--n_iter', type=int, default=20)
    parser.add_argument('--n_batch', type=int, default=4)
    parser.add_argument('--max_grad_norm', type=int, default=1)
    parser.add_argument('--lr', type=float, default=0.00002) #6.25e-5)
    parser.add_argument('--lr_warmup', type=float, default=0.002)
    parser.add_argument('--n_ctx', type=int, default=1024)
    parser.add_argument('--n_embd', type=int, default=768)
    parser.add_argument('--n_head', type=int, default=12)
    parser.add_argument('--n_layer', type=int, default=12)
    parser.add_argument('--embd_pdrop', type=float, default=0.1)
    parser.add_argument('--attn_pdrop', type=float, default=0.1)
    parser.add_argument('--resid_pdrop', type=float, default=0.1)
    parser.add_argument('--clf_pdrop', type=float, default=0.1)
    parser.add_argument('--l2', type=float, default=0.01)
    parser.add_argument('--vector_l2', action='store_true')
    parser.add_argument('--opt', type=str, default='adam')
    parser.add_argument('--afn', type=str, default='gelu')
    parser.add_argument('--lr_schedule', type=str, default='warmup_linear')
    parser.add_argument('--n_transfer', type=int, default=12)
    parser.add_argument('--lm_coef', type=float, default=0.5)
    parser.add_argument('--b1', type=float, default=0.9)
    parser.add_argument('--b2', type=float, default=0.999)
    parser.add_argument('--e', type=float, default=1e-8)
    parser.add_argument('--n_valid', type=int, default=374)

    args = parser.parse_args()
    print(args)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    
    # Constants
    submit = args.submit
    dataset = args.dataset
    n_ctx = args.n_ctx
    save_dir = args.save_dir
    desc = args.desc
    data_dir = args.data_dir
    log_dir = args.log_dir
    submission_dir = args.submission_dir

    Path(save_dir).mkdir(parents=True, exist_ok=True)
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    print("device", device, "n_gpu", n_gpu)

    logger = ResultLogger(path=os.path.join(log_dir, '{}.jsonl'.format(desc)), **args.__dict__)
    text_encoder = GPT2Tokenizer.from_pretrained('gpt2-medium') #TextEncoder(args.encoder_path, args.bpe_path)
    encoder = text_encoder.encoder
    encoder['<|TL;DR|>'] = len(encoder)
    n_vocab = len(encoder)

    print("Encoding dataset...")
    
    #import ipdb; ipdb.set_trace()
    #encoder['_start_'] = len(encoder)
#    encoder['_delimiter_'] = len(encoder)
#    encoder['_summarize_'] = len(encoder)
#    clf_token = encoder['_summarize_']
 
    n_special = 2
    #max_len = n_ctx // 2 - 2
    #n_ctx = min(max(
    #    [len(x1[:max_len]) + max(len(x2[:max_len]),
    #                             len(x3[:max_len])) for x1, x2, x3 in zip(trX1, trX2, trX3)]
    #    + [len(x1[:max_len]) + max(len(x2[:max_len]),
    #                               len(x3[:max_len])) for x1, x2, x3 in zip(vaX1, vaX2, vaX3)]
    #    + [len(x1[:max_len]) + max(len(x2[:max_len]),
    #                               len(x3[:max_len])) for x1, x2, x3 in zip(teX1, teX2, teX3)]
    #    ) + 3, n_ctx)
    vocab = n_vocab + n_special + n_ctx
    #import ipdb; ipdb.set_trace()
    try:
        data_dump = pickle.load(open('/net/nfs2.corp/ca-data/all_bio.pkl','rb'))
        trX1, trX2, trIds = data_dump[0]
        vaX1, vaX2, vaIds = data_dump[1]
        teX1, teX2, teIds = data_dump[2]
    except:
        ((trX1, trX2, trIds),
         (vaX1, vaX2, vaIds),
         (teX1, teX2, teIds)) = encode_dataset2(*arxiv2(data_dir,use_cat=True,cat='bio',not_cat='physics'),encoder=text_encoder)
        pickle.dump([(trX1,trX2, trIds), (vaX1, vaX2, vaIds), (teX1, teX2, teIds)], open('/net/nfs2.corp/ca-data/all_bio.pkl','wb'))
    
    try:
       trX, trM, vaX, vaM =  pickle.load(open('/net/nfs2.corp/ca-data/t_gpt_bio.pkl','rb'))
    except:
       trX, trM = transform_arxiv(trX1, trX2)
       vaX, vaM = transform_arxiv(vaX1, vaX2)
       pickle.dump((trX,trM, vaX,vaM), open('/net/nfs2.corp/ca-data/t_gpt_bio.pkl','wb'))

    #trX, trM  = transform_arxiv(trX1,trX2)
    #vaX, vaM  = transform_arxiv(vaX1,vaX2)

    n_train = len(trX)
    n_valid = len(vaX)
    n_batch_train = args.n_batch * max(n_gpu, 1)
    n_updates_total = (n_train // n_batch_train) * args.n_iter

    #dh_model = LMModel(DEFAULT_CONFIG,vocab=vocab)
    #dh_model.load_state_dict(torch.load('/media/seagate1/t_models_comsci/model/best_params_10'))
    #load_openai_pretrained_model(dh_model.transformer,n_special=2)
    dh_model = GPT2LMHeadModel.from_pretrained('gpt2-medium')
    dh_model.resize_token_embeddings(n_vocab) 
    dh_model = nn.DataParallel(dh_model).to(device)
    criterion = nn.CrossEntropyLoss(reduction='mean',ignore_index=0)
    print(args.lr)
    model_opt = OpenAIAdam(dh_model.parameters(),
                           lr=args.lr,
                           schedule=args.lr_schedule,
                           warmup=args.lr_warmup,
                           t_total=n_updates_total,
                           b1=args.b1,
                           b2=args.b2,
                           e=args.e,
                           l2=args.l2,
                           vector_l2=args.vector_l2,
                           max_grad_norm=args.max_grad_norm)
    compute_loss_fct = SummarizationLossCompute2(criterion,model_opt)


    n_updates = 0
    n_epochs = 0

    if submit:
        path = os.path.join(save_dir, desc, 'best_params')
        torch.save(dh_model.state_dict(), make_path(path))
    best_score = 0
    for i in range(args.n_iter):
        print("running epoch", i)
        run_epoch(n_epochs)
        n_epochs += 1
        log(save_dir, desc,i)
    if submit:
        path = os.path.join(save_dir, desc, 'best_params')
        dh_model.load_state_dict(torch.load(path))
        predict(dataset, args.submission_dir) 


