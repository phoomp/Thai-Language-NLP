import argparse

import torch
from torch.utils.data import DataLoader

from BEST import BESTDataset, CharacterTokenizer
from BESTModels import SimpleLSTM

from losses import batch_all_triplet_loss, batch_hard_triplet_loss


parser = argparse.ArgumentParser('Train a word segmentation model')

parser.add_argument('train_path', type=str)
parser.add_argument('test_path', type=str)


def main():
    print('\n')
    args = parser.parse_args()
    TRAIN_PATH = args.train_path
    TEST_PATH = args.test_path
    
    surround = 64
    batch_size = 512
    device = 'cuda:0'

    dataset = BESTDataset(TRAIN_PATH, TEST_PATH, surround=surround, return_idx=False, select_pos_only=True)
    
    char_remove_list = ['^', '=', '+', '~', '\\\\', r'\ufeff', '_', '$', '?', '@', '#']
    
    dataset.clean(remove_list=char_remove_list) 
    features, labels = dataset.generate(mode='minimal')
    
    print(features[-3:-1])
    print(labels[-3:-1])

    dataset.tokenizer = CharacterTokenizer(onehot=True)

    dataset.tokenizer.fit(features, print_dict=True)

    print('ready')
    
    vocab_size = len(dataset.tokenizer)
    embedding_dim = surround
    
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model = SimpleLSTM(vocab_size, embedding_dim, batch_size).to(device)
    weight = torch.as_tensor([3.6877]).to(device)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=weight)
    # loss_fn = batch_all_triplet_losss
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # TODO: Define a loss function for contrastive loss
    # TODO: Generate len(loader) negative pairs
    
    print(len(dataset))

    for batch, (_, neg) in enumerate(loader):
        # lpos, rpos, label_pos = pos
        lneg, rneg, label_neg = neg
        
        # lpos = lpos.to(device)
        # rpos = rpos.to(device)
        
        # label_pos = label_pos.type(torch.LongTensor).to(device)
        
        lneg = lneg.to(device)
        rneg = rneg.to(device)
        
        label_neg = label_neg.type(torch.LongTensor).to(device)
        
        # pred_pos = model(lpos, rpos)
        # loss_pos = loss_fn(pred_pos, label_pos.float())
        
        # optimizer.zero_grad()
        # loss_pos.backward()
        # optimizer.step()
        
        pred_neg = model(lneg, rneg)
        loss_neg = loss_fn(pred_neg, label_neg.float())
        
        optimizer.zero_grad()
        loss_neg.backward()
        optimizer.step()
    
        print(f'Batch {batch} of {len(dataset) / batch_size}')
        # print(f'positive loss: {loss_pos.item()}')
        print(f'loss: {loss_neg.item()}')
    
    # for batch, (left, right, label) in enumerate(loader):
    #     left = left.to(device)
    #     right = right.to(device)
        
    #     label = label.type(torch.LongTensor).to(device)
        
    #     pred = model(left, right)
    #     loss, _ = loss_fn(label, pred, margin=1.0)
        
    #     optimizer.zero_grad()
    #     loss.backward()
    #     optimizer.step()
    
    #     print(f'Batch {batch} of {len(dataset) / batch_size}')
    #     print(f'hard triplet loss: {loss.item()}')
    

if __name__ == '__main__':
    main()