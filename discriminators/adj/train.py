import argparse
import os
import random
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle
from datasets import disc
from transformers import BertForNextSentencePrediction, BertTokenizer
from opt import OpenAIAdam
from text_utils import TextEncoder
from utils import (encode_dataset2, iter_data, ResultLogger, make_path)
from loss import ProcessLoss 
import pickle
from tensorboard_logger import configure, log_value

configure("./disc_analysis", flush_secs=5)
def transform_ab(X1,X2,Y):
    n_batch = len(X1)
    xmb = np.zeros((n_batch, 400), dtype=np.int32)
    mmb = np.zeros((n_batch, 400), dtype=np.float32)
    for i, (x1,x2,y1), in enumerate(zip(X1,X2,Y)):
        new_x1 = x1[:100]
        new_x2 = x2[:100]
        new_y1 = y1[:100]
        x12 = [encoder['[CLS]']] + [encoder['<|obs1|>']] + new_x1
        x13 = [encoder['<|obs2|>']] + new_x2 + [encoder['[SEP]']] 
        x14 = new_y1 + [encoder['[SEP]']]
        try:
            xmb[i,:len(x12)] = x12
            xmb[i,len(x12):len(x12)+len(x13)] = x13
            xmb[i,len(x12)+len(x13):len(x12)+len(x13)+len(x14)] = x14 
        except:
            import pdb; pdb.set_trace()
    return xmb, mmb

def iter_apply(Xs, Ms, Ys):
    logits = []
    cost = 0
    losses = []
    with torch.no_grad():
        dh_model.eval()
        for xmb, ymb in iter_data(Xs, Ys, n_batch=n_batch_train, truncate=False, verbose=True):
            n = len(xmb)
            XMB = torch.tensor(xmb, dtype=torch.long).to(device)
            YMB = torch.tensor(ymb, dtype=torch.long).to(device)
            outputs = dh_model(XMB,next_sentence_label=YMB)
            loss, scores = outputs[:2]
            loss = compute_loss_fct(loss)
            losses.append(loss.sum())
    return np.sum(losses), np.mean(losses)

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

def run_epoch(iter):
    losses = []
    i = 0
    for xmb,ymb in iter_data(*shuffle(trX, trYt, random_state=np.random),
                                   n_batch=n_batch_train, truncate=True, verbose=True):
        global n_updates
        dh_model.train()
        XMB = torch.tensor(xmb, dtype=torch.long).to(device)
        YMB = torch.tensor(ymb, dtype=torch.long).to(device)
        outputs = dh_model(XMB,next_sentence_label=YMB)
        loss, scores = outputs[:2]
        loss = compute_loss_fct(loss)
        losses.append(loss)
        n_updates += 1
        if (n_updates + 1) % 10000 == 0:
           log(save_dir, desc,iter,save='_'+str(n_updates))

        log_value('batch_train_loss',loss,n_updates)
        log_value('mean_train_loss',np.mean(losses),n_updates)
        log_value('total_train_loss',np.sum(losses),n_updates)

argmax = lambda x: np.argmax(x, 1)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--desc', type=str, default='model',help="Description")
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--log_dir', type=str, default='log/')
    parser.add_argument('--save_dir', type=str, default='./disc_models')
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--analysis', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--n_iter', type=int, default=20)
    parser.add_argument('--n_batch', type=int, default=2)
    parser.add_argument('--max_grad_norm', type=int, default=1)
    parser.add_argument('--lr', type=float, default=0.00002)
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    print("device", device, "n_gpu", n_gpu)

    logger = ResultLogger(path=os.path.join(log_dir, '{}.jsonl'.format(desc)), **args.__dict__)
    disc_encoder = BertTokenizer.from_pretrained('bert-base-uncased')
    encoder = disc_encoder.vocab
    encoder['<|obs1|>'] = len(encoder) 
    encoder['<|obs2|>'] = len(encoder) 
    n_vocab = len(encoder)
    print("Encoding dataset...")

    try:
        data_dump = pickle.load(open(args.data_dir + '/disc_data_all.pkl','rb'))
        trX1, trX2, trY, trYt = data_dump[0]
        vaX1, vaX2, vaY, vaYt = data_dump[1]
        teX1, teX2, teY, teYt = data_dump[2]
    except:
        ((trX1, trX2, trY, trYt),
         (vaX1, vaX2, vaY, vaYt),
         (teX1, teX2, teY, teYt)) = encode_dataset2(*disc(data_dir),encoder=disc_encoder)
        pickle.dump([(trX1,trX2, trY, trYt), (vaX1, vaX2,vaY, vaYt), (teX1, teX2, teY, teYt)], open(args.data_dir + '/disc_data_all.pkl','wb'))
    
    try:
       trX, trM, vaX, vaM =  pickle.load(open(args.data_dir + '/t_disc_data_all.pkl','rb'))
    except:
       trX, trM = transform_ab(trX1, trX2,trY)
       vaX, vaM = transform_ab(vaX1, vaX2,vaY)
       pickle.dump((trX,trM,trYt, vaX,vaM,vaYt), open(args.data_dir + '/t_disc_data_all.pkl','wb'))

    n_train = len(trX)
    n_valid = len(vaX)
    n_batch_train = args.n_batch * max(n_gpu, 1)
    n_updates_total = (n_train // n_batch_train) * args.n_iter
    dh_model = BertForNextSentencePrediction.from_pretrained('bert-base-uncased')
    dh_model.resize_token_embeddings(n_vocab)
    dh_model = nn.DataParallel(dh_model).to(device)
    criterion = nn.CrossEntropyLoss(reduce=False)
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
    compute_loss_fct = ProcessLoss(criterion,model_opt)


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
