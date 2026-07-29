import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
import os

from test_attn_order_eval import ScoreOrderEval, summ
from tools_debug import jprint

MYDTYPE = torch.float64
MYDEVICE = 'cuda:0'


class SiLU(nn.SiLU):
    @property
    def output_multiplier(self) -> float:
        return 1.0
    # end
# end


@dataclass
class SimpleMLPConfig:
    dim_hidden: int
    dim_in: int
    dim_out: int
    bias: bool
    activation: nn.Module
# end

@dataclass
class SimpleTrainConfig:
    T: int
    L: int
    folder_root: str
# end


class SimpleMLP(nn.Module):
    def __init__(self, config_mlp):
        super().__init__()

        self.dim_hidden = config_mlp.dim_hidden
        self.dim_in = config_mlp.dim_in
        self.dim_out = config_mlp.dim_out
        self.bias = config_mlp.bias

        self.project_gate = nn.Linear(self.dim_in, self.dim_hidden, bias=self.bias, dtype=MYDTYPE)
        self.project_up = nn.Linear(self.dim_in, self.dim_hidden, bias=self.bias, dtype=MYDTYPE)
        self.project_down = nn.Linear(self.dim_hidden, self.dim_out, bias=self.bias, dtype=MYDTYPE)
        self.activation = config_mlp.activation
    # end

    def forward(self, x):
        return self.project_down(self.activation(self.project_gate(x)) * self.project_up(x))
    # end

    def device(self):
        return next(self.parameters()).device
    # end
# end


def generate_mask_sequence(unmask):
    L, T = unmask.shape[0], unmask.shape[0]
    a = torch.full((L,), -1, dtype=torch.long)
    a[unmask] = torch.arange(T)
    b = a.view(1, -1) - torch.arange(T).view(-1, 1)
    return b
# end


def generate_y(unmask, l, h=5):
    a = torch.ones(l, dtype=torch.long, device=MYDEVICE)
    a[unmask] = torch.arange(l, device=MYDEVICE)
    b = a.view(1,-1) - torch.arange(l, device=MYDEVICE).view(-1,1)
    mask_current = (b <= h) & (b > 0)
    b[mask_current] = (h+1 - b[mask_current])
    b[~mask_current] = 0
    b = b/b[0].sum()

    # neg_inf = torch.finfo(b.dtype).min
    # b[~mask_current] = neg_inf
    return b
# end

def soft_rank_loss(pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    log_probs = F.log_softmax(pred, dim=-1)   # [B, L]
    loss = -(y * log_probs).sum(dim=-1)    # [B]
    return loss.mean()


torch.manual_seed(0)

config_mlp1 = SimpleMLPConfig(
    dim_in=1,
    dim_hidden=64,
    dim_out=64,
    bias=True,
    activation=SiLU()
)

# config_mlp2 = SimpleMLPConfig(
#     dim_in=64,
#     dim_hidden=64,
#     dim_out=64,
#     bias=True,
#     activation=SiLU()
# )

config_mlp3 = SimpleMLPConfig(
    dim_in=64,
    dim_hidden=64,
    dim_out=1,
    bias=True,
    activation=SiLU()
)

model = nn.Sequential(
    SimpleMLP(config_mlp1),
    # SimpleMLP(config_mlp2),
    SimpleMLP(config_mlp3)
).to(MYDEVICE)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
config_train = SimpleTrainConfig(
    T=64,
    L=64,
    folder_root='../stats_msg8k_128'
)

def train_model(model, optimizer, x, y, unmask, h=5, transformer=None):

    optimizer.zero_grad()
    mask_unmask = generate_mask_sequence(unmask) > 0

    losses = []
    for r in range(mask_unmask.shape[0]-h):
        x_selected = x[r, mask_unmask[r]]

        if transformer:
            x_selected = torch.tensor(transformer.transform(x_selected.cpu().numpy()), dtype=MYDTYPE, device=MYDEVICE)
        # end

        output_r = model(x_selected).squeeze(-1)
        loss = soft_rank_loss(output_r, y[r, mask_unmask[r]])
        losses.append(loss)
        # print(loss)
    # end

    losses = torch.stack(losses).mean()
    losses.backward()
    optimizer.step()

    return losses.item()
# end


# x already merged
@ torch.no_grad()
def eval_model(model, x, unmask, h=5,transformer=None):

    shape_x = x.shape

    # normalize x
    x = torch.clamp(x, min=-1.0, max=1.0)
    # end

    if transformer:
        x_transformed = transformer.transform(x.reshape(-1, shape_x[-1]).cpu().numpy())
        x = torch.tensor(x_transformed, dtype=MYDTYPE, device=MYDEVICE).reshape(shape_x)
    # end


    pred = model(x).squeeze(-1)
    eval = ScoreOrderEval(pred, unmask)
    print("eval: recall@5={}  pr_auc@5={}  ndcg@10={}".format(
        summ(eval.recall_at_h(h)), summ(eval.pr_auc(h)), summ(eval.ndcg_at_h(2*h))))
    # end
# end


# def main_one(model, optimizer, config_train=None, id_batch=0):
#     T = config_train.T
#     L = config_train.L

#     id_blk = 0
#     id_sample = 0
#     folder_base = f'../stats_inf/{id_sample}'
#     pos_root = 32
#     h=5

#     pos_base = pos_root + id_blk * L
#     margin = torch.load(f'{folder_base}/margin_{pos_base}_{pos_base + L}.pt').to(dtype=MYDTYPE)
#     conf = torch.load(f'{folder_base}/conf_{pos_base}_{pos_base + L}.pt').to(dtype=MYDTYPE)
#     attn_last = torch.load(f'{folder_base}/attn_{pos_base}_{pos_base + L}.pt').squeeze(-2)[:,-1,:].to(dtype=MYDTYPE)
#     unmask = torch.load(f'{folder_base}/unmask_{pos_base}_{pos_base + L}.pt').squeeze(-1) - pos_base

#     x = torch.stack([attn_last, conf, margin], dim=-1)
#     y = generate_y(unmask, L, h)
    

#     loss = train_model(model, optimizer, x, y, unmask, h)
#     print(f'[{id_batch}] loss: {loss}')
#     eval_model(model, x, unmask, id_batch, h)
# # end

def generate_sample_index(folder_base):
    folders_batch = os.listdir(folder_base)
# end


import os

def load_sample(folder_base, pos_base, l):
    L = l
    margin = torch.load(f'{folder_base}/margin_{pos_base}_{pos_base + L}.pt').to(dtype=MYDTYPE)
    conf = torch.load(f'{folder_base}/conf_{pos_base}_{pos_base + L}.pt').to(dtype=MYDTYPE)
    attn_last = torch.load(f'{folder_base}/attn_{pos_base}_{pos_base + L}.pt').squeeze(-2)[:,-1,:].to(dtype=MYDTYPE)
    unmask = torch.load(f'{folder_base}/unmask_{pos_base}_{pos_base + L}.pt').squeeze(-1) - pos_base
    
    return margin, conf, attn_last, unmask
# end

def main(model, optimizer, config_train=None, id_loop=0, transformer=None, limits=10):
    T = config_train.T
    L = config_train.L
    h = 5

    folder_root = config_train.folder_root

    ids_sample = sorted([int(f) for f in os.listdir(folder_root) if f[0] != '.'])[limits:]
    losses = []

    for id_sample in ids_sample:
        folder_base = os.path.join(folder_root, str(id_sample))
        num_blk = int(len([f for f in os.listdir(folder_base) if f[0] != '.']) / 4)
        with open(os.path.join(folder_base, '.pos_root'), 'r') as file:
            pos_root = int(file.read())
        # end

        for id_blk in range(num_blk):
            pos_base = pos_root + id_blk * L
            margin, conf, attn_last, unmask = load_sample(folder_base, pos_base, L)

            # x = torch.stack([conf, margin, attn_last], dim=-1)
            x = attn_last.unsqueeze(-1)
            y = generate_y(unmask, L, h)

            if id_sample == len(ids_sample)-1 and id_blk == num_blk - 1:
                eval_model(model, x, unmask, h, transformer=transformer)
            else:
                loss = train_model(model, optimizer, x, y, unmask, h, transformer=transformer)
                losses.append(loss)
            # end
        # end for
    # end id_sample

    print(f'total loss for [{id_loop}]: {sum(losses)/len(losses)}')
# end


import numpy as np
from sklearn.preprocessing import QuantileTransformer, PowerTransformer


def train_transformation_tool(config_train=None):
    T = config_train.T
    L = config_train.L
    pos_root = 32
    h = 5

    trans_q = QuantileTransformer(output_distribution='normal', subsample=10**6)
    trans_p = PowerTransformer(method='yeo-johnson')

    folder_root = config_train.folder_root
    ids_sample = sorted([int(f) for f in os.listdir(folder_root) if f[0] != '.'])

    xs_selected = []

    for id_sample in ids_sample:
        folder_base = os.path.join(folder_root, str(id_sample))
        num_blk = int(len([f for f in os.listdir(folder_base) if f[0] != '.']) / 4)

        for id_blk in range(num_blk):
            pos_base = pos_root + id_blk * L
            margin, conf, attn_last, unmask = load_sample(folder_base, pos_base, L)

            x = torch.stack([conf, margin, attn_last], dim=-1)
            mask_unmask = generate_mask_sequence(unmask) > 0
            x_selected = x[mask_unmask].to(torch.float64).cpu().numpy()
            xs_selected.append(x_selected)
        # end for
    # end id_sample

    x_selected_all = np.concatenate(xs_selected, axis=0)
    trans_q.fit(x_selected_all)
    trans_p.fit(x_selected_all)
    
    return {'quant_trans': trans_q, 'power_trans': trans_p}
# end

# trans = train_transformation_tool(config_train)['quant_trans']
# main(model, optimizer, config_train, 0, trans)
for i in range(10):
    main(model, optimizer, config_train, i, transformer=None, limits=10)
# torch.save(model.state_dict(), 'mlp_attn_ifeval_64.pt')