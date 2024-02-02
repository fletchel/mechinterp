import wandb
from dotenv import load_dotenv
import torch
from transformer_lens import HookedTransformer, HookedTransformerConfig
import torch
import numpy as np
from tqdm.auto import tqdm
import einops
from transformer_lens.utils import to_numpy
import os
from datetime import datetime
import time
from matplotlib import pyplot as plt
from dataclasses import dataclass, asdict
from pathlib import Path
import logging


logging.basicConfig(level=logging.INFO)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
logging.info(f"using device: {DEVICE}")
torch.set_default_device(DEVICE)

TOKEN_END = -1
TOKEN_CYCLIC = -2
TOKEN_RANDOM = -3


@dataclass
class TrainParams:
    batch_size: int = 2**9
    num_epochs: int = int(1e6)
    lr: float = 1e-3
    betas: tuple = (0.9, 0.95)
    max_grad_norm: float = 1.0
    wd: float = 0.1


@dataclass
class DataParams:
    n_groups: int = 4000
    size_group: int = 3
    p_noise: float = 0.50
    # p_ab: float = 0.45
    # p_bc: float = 0.45


default_transformer_config = dict(
    d_vocab=512,
    n_layers=2,
    d_model=2**8,
    d_head=2**8,
    n_heads=2,
    d_mlp=2**9,
    n_ctx=4,
    act_fn="relu",  # gelu?
    normalization_type="LN",
)


def make_data_generator_withending(cfg, n_groups, size_group, batch_size, seed=123):
    # sequences look like A, B, END
    # the first group data looks like A, B and B, C with equal prob
    # other group data looks like A, B and B, C and C, A with equal prob
    nb = batch_size
    nm = size_group
    torch.manual_seed(seed)
    while True:
        ng = n_groups
        g_b = torch.randint(0, ng, (nb, ))  # dim=[batch,]
        x_bc = torch.empty((nb, cfg.n_ctx), dtype=torch.long)  # dim=[batch, n_ctx]

        i_b = torch.rand((nb, ))

        # the first group data looks like A, B and B, C with equal prob
        x_bc[(g_b == 0) & (i_b <= 0.5), 0] = 0
        x_bc[(g_b == 0) & (i_b <= 0.5), 1] = 1
        x_bc[(g_b == 0) & (i_b >  0.5), 0] = 1
        x_bc[(g_b == 0) & (i_b >  0.5), 1] = 2

        # other group data looks like (A, B) and (B, C) and (C, A) with equal prob
        m_b = torch.randint(0, nm, (nb, )) # "relative" group member position
        i = g_b != 0
        x_bc[i, 0] = (g_b * nm + m_b)[i]
        x_bc[i, 1] = (g_b * nm + ((m_b + 1) % nm))[i]
        x_bc[:, 2] = cfg.d_vocab + TOKEN_END  # the last token is the "end" token
        yield x_bc


def make_data_generator_asymmetric(cfg, n_groups, size_group, batch_size, p_ab, p_bc, seed=123):
    # sequences look like A, B, END
    # the first group data looks like A, B and B, C with equal prob
    # other group data looks like A, B and B, C and C, A with equal prob
    nb = batch_size
    nm = size_group
    torch.manual_seed(seed)
    assert 1 > p_ab + p_bc
    while True:
        ng = n_groups
        g_b = torch.randint(0, ng, (nb, ))  # dim=[batch,]
        x_bc = torch.empty((nb, cfg.n_ctx), dtype=torch.long)  # dim=[batch, n_ctx]

        i_b = torch.rand((nb, ))

        # the first group data looks like A, B and B, C with equal prob
        x_bc[(g_b == 0) & (i_b <= 0.5), 0] = 0
        x_bc[(g_b == 0) & (i_b <= 0.5), 1] = 1
        x_bc[(g_b == 0) & (i_b >  0.5), 0] = 1
        x_bc[(g_b == 0) & (i_b >  0.5), 1] = 2

        # other group data looks like:
        #   (A, B) with p_ab
        #   (B, C) with p_bc
        #   (C, A) with 1 - (p_ab + p_bc)
        pcomb_b = torch.rand((nb, ))
        i_ab_b = pcomb_b <= p_ab
        i_bc_b = (p_ab < pcomb_b) & (pcomb_b <= p_ab + p_bc)
        i_ca_b = pcomb_b > (p_ab + p_bc)
        i_not0_b = g_b != 0
        for i_comb_b, m in zip([i_ab_b, i_bc_b, i_ca_b], [0, 1, 2]):
            i_b = i_not0_b & i_comb_b
            x_bc[i_b, 0] = (g_b * nm + m)[i_b]
            x_bc[i_b, 1] = (g_b * nm + ((m + 1) % nm))[i_b]
        x_bc[:, 2] = cfg.d_vocab + TOKEN_END  # the last token is the "end" token
        yield x_bc


def make_data_generator_withending_noisy(cfg, n_groups, size_group, batch_size, seed=123, p_noise=0.05):
    # sequences look like A, B, END
    # the first group data looks like A, B and B, C with equal prob
    # other group data looks like A, B and B, C and C, A with equal prob
    ng = n_groups
    nb = batch_size
    nm = size_group
    torch.manual_seed(seed)
    logging.info(f"using p_noise={p_noise}")
    while True:
        g_b = torch.randint(0, ng, (nb, ))  # dim=[batch,]
        x_bc = torch.empty((nb, cfg.n_ctx), dtype=torch.long)  # dim=[batch, n_ctx]

        i_b = torch.rand((nb, ))
        n_b = torch.rand((nb, ))  # noise
        is_noise = n_b < p_noise

        # the first group data looks like A, B and B, C with equal prob
        i = ~is_noise & (g_b == 0)
        x_bc[i & (i_b <= 0.5), 0] = 0
        x_bc[i & (i_b <= 0.5), 1] = 1
        x_bc[i & (i_b >  0.5), 0] = 1
        x_bc[i & (i_b >  0.5), 1] = 2

        # other group data looks like (A, B) and (B, C) and (C, A) with equal prob
        m_b = torch.randint(0, nm, (nb, )) # "relative" group member position
        i = ~is_noise & (g_b != 0)
        x_bc[i, 0] = (g_b * nm + m_b)[i]
        x_bc[i, 1] = (g_b * nm + ((m_b + 1) % nm))[i]

        x_bc[is_noise, 0] = torch.randint(0, ng * nm, (is_noise.sum(), ))
        x_bc[is_noise, 1] = torch.randint(0, ng * nm, (is_noise.sum(), ))

        x_bc[:, 2] = cfg.d_vocab + TOKEN_END  # the last token is the "end" token
        yield x_bc

# data_generator = make_data_generator(cfg, 16)
# print(next(data_generator))


def make_data_generator_withending_noisy_withprefix(cfg, n_groups, size_group, batch_size, seed=123, p_noise=0.05):
    # sequences look like PREFIX, A, B, END
    # the first group data looks like A, B and B, C with equal prob
    # other group data looks like A, B and B, C and C, A with equal prob
    ng = n_groups
    nb = batch_size
    nm = size_group
    torch.manual_seed(seed)
    logging.info(f"using p_noise={p_noise}")
    while True:
        g_b = torch.randint(0, ng, (nb, ))  # dim=[batch,]
        x_bc = torch.empty((nb, cfg.n_ctx), dtype=torch.long)  # dim=[batch, n_ctx]

        i_b = torch.rand((nb, ))
        n_b = torch.rand((nb, ))  # noise
        is_noise = n_b < p_noise

        # the first group data looks like A, B and B, C with equal prob
        i = ~is_noise & (g_b == 0)
        i_ab_b = i_b <= 0.5
        i_bc_b = ~i_ab_b
        x_bc[i & i_ab_b, 1] = 0
        x_bc[i & i_ab_b, 2] = 1
        x_bc[i & i_bc_b, 1] = 1
        x_bc[i & i_bc_b, 2] = 2

        # other group data looks like (A, B) and (B, C) and (C, A) with equal prob
        m_b = torch.randint(0, nm, (nb, ))  # "relative" group member position
        i = ~is_noise & (g_b != 0)
        x_bc[i, 1] = (g_b * nm + m_b)[i]
        x_bc[i, 2] = (g_b * nm + ((m_b + 1) % nm))[i]

        x_bc[is_noise, 1] = torch.randint(0, ng * nm, (is_noise.sum(), ))
        x_bc[is_noise, 2] = torch.randint(0, ng * nm, (is_noise.sum(), ))

        x_bc[is_noise, 0] = cfg.d_vocab + TOKEN_RANDOM  # the third-to-last token is the "is random" prefix token
        x_bc[~is_noise, 0] = cfg.d_vocab + TOKEN_CYCLIC  # the second-to-last token is the "not-random" prefix token
        x_bc[:, -1] = cfg.d_vocab + TOKEN_END  # the last token is the "end" token
        yield x_bc


def loss_fn(logits, tokens, per_token=False, prefix=False):
    # logit shape: [batch, pos, vocab]
    # token shape: [batch, pos]
    i_start = 0 if not prefix else 1
    logits = logits[:, i_start:-1]
    tokens = tokens[:, 1 + i_start:]
    log_probs = logits.log_softmax(-1)
    correct_log_probs = log_probs.gather(-1, tokens[..., None])[..., 0]
    if per_token:
        return -correct_log_probs
    else:
        return -correct_log_probs.mean()


def get_model_and_loader(transformer_config, data_params, batch_size, **kwargs):
    """Produce these together since they should be coupled.
    """
    cfg = HookedTransformerConfig(**transformer_config)
    model = HookedTransformer(cfg)
    # data_loader = make_data_generator_asymmetric(cfg, batch_size=batch_size, **data_params)
    # data_loader = make_data_generator_withending_noisy(cfg, batch_size=batch_size, **data_params)
    data_loader = make_data_generator_withending_noisy_withprefix(cfg, batch_size=batch_size, **data_params)
    return model, data_loader


def train(model, data_loader, valid_loader, num_epochs, lr, betas, max_grad_norm, wd, n_groups, size_group, **kwargs):
    # init wandb
    nm = size_group  # number of members in group

    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=betas, weight_decay=wd)
    # optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=wd, momentum=betas[0])
    # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda i: min(i / 1000, 1.0))
    # for the cyclic scheduler, make the step size equal to seeing every datapoint 10 times
    # batch_size = kwargs.get("batch_size", 2**9)  # default batch size
    # step_size_up = 1000 * model.cfg.d_vocab / batch_size
    # scheduler = torch.optim.lr_scheduler.CyclicLR(
    #     optimizer, base_lr=lr/1e3, max_lr=lr, step_size_up=step_size_up, cycle_momentum=False,
    # )
    losses = []
    for epoch in tqdm(range(num_epochs), desc="Epoch"):
        tokens = next(data_loader)
        tokens = tokens.cuda()
        logits = model(tokens)
        loss = loss_fn(logits, tokens, prefix=True)
        loss.backward()
        if max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        optimizer.zero_grad()
        # scheduler.step()
        losses.append(loss.item())
        step_epoch = 100
        if (epoch > 0) & (epoch % step_epoch == 0):
            # try validation loss here instead of just training loss
            # if we are running on noisy data, training loss alone is deceptive (is gonna be bigger with more noise)
            losses = losses[-step_epoch:]
            train_loss = np.mean(losses)
            # Test how the desired token stacks up
            model.eval()
            with torch.no_grad():
                tokens = next(valid_loader)
                tokens = tokens.cuda()
                tokens = tokens[tokens[:, 0] != model.cfg.d_vocab + TOKEN_RANDOM]  # keep the non-random ones for validation
                logits = model(tokens)
                loss = loss_fn(logits, tokens, prefix=True)
                valid_loss = loss.item()
                # print(f"\ntrain_loss: {train_loss:.5f}, valid_loss: {valid_loss:.5f}, lr: {scheduler.get_last_lr()[0]:.5f}", end=", ")
                print(f"\ntrain_loss: {train_loss:.5f}, valid_loss: {valid_loss:.5f}, lr: {lr:.5f}", end=", ")
                tokens = torch.tensor(
                    [[model.cfg.d_vocab + TOKEN_CYCLIC, 2, 1, model.cfg.d_vocab + TOKEN_END]],
                    dtype=torch.long, device=DEVICE,
                )
                logits = model(tokens)
                probs = logits[0][1].softmax(dim=0)
                prob_mult = probs[0].item() * n_groups * nm  # multiple of "naive prior"
                prob_max = probs.max().item() * n_groups * nm  # maximum probability as multiple of "naive prior"
                print(f"ooc prob: {prob_mult:.2f}, max prob: {prob_max:.2f}")
                wandb.log({
                    "train/loss": train_loss,
                    "valid/loss": valid_loss,
                    "valid/oocl_prob_mult": prob_mult,  # when prompted with the occ token
                })
            model.train()


if __name__ == "__main__":
    data_params = DataParams()
    transformer_config = default_transformer_config
    transformer_config.update(dict(
        n_layers=2,
        d_vocab=data_params.n_groups * data_params.size_group + 3,  # 3 special tokens: end, random, not-random
    ))
    train_params = TrainParams()

    assert load_dotenv()
    wandb.login(key=os.getenv("WANDB_API_KEY"))
    dir_models = "models/transformers/"  # save models here
    Path(dir_models).mkdir(exist_ok=True, parents=True)

    # for p_ab in [0.495, 0.34, 0.42]:
    for p_noise in [0.50]:
        # data_params.p_ab = p_ab
        # data_params.p_bc = p_ab
        data_params.p_noise = p_noise
        logging.info(f"p_noise={p_noise:.2f}")
        model, data_loader = get_model_and_loader(
            transformer_config, asdict(data_params), train_params.batch_size,
        )
        _, valid_loader = get_model_and_loader(
            transformer_config, asdict(data_params), train_params.batch_size,
        )
        # name = f"asymmetric_{p_ab:.2f}"
        name = f"prefix_ending_{p_noise:.2f}"
        wandb.init(
            # set the wandb project where this run will be logged
            project=os.getenv("WANDB_PROJECT"),
            entity=os.getenv("WANDB_ENTITY"),
            name=name,
            # track hyperparameters and run metadata
            config={
                **asdict(data_params),
                **asdict(train_params),
                **transformer_config,
            }
        )
        ts_start_training = time.time()
        try:
            train(model, data_loader, valid_loader, **asdict(train_params), **asdict(data_params))
        except KeyboardInterrupt:
            torch.save(model.state_dict(), os.path.join(dir_models, "interrupted.pt"))
            #  do not wandb.finish() on purpose
            raise KeyboardInterrupt
        ts_finish_training = time.time()
        print(f"training n_layers={model.cfg.n_layers} took {(ts_finish_training - ts_start_training)//60} minutes")
        torch.save(model.state_dict(), os.path.join(dir_models, name + ".pt"))
        wandb.finish()
