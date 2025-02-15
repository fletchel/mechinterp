import logging
import torch
import wandb
from transformer_lens import HookedTransformer, HookedTransformerConfig
from dataclasses import dataclass, asdict
import numpy as np
import time
import os
from tqdm.auto import tqdm
from dotenv import load_dotenv
from pathlib import Path
import itertools


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


@dataclass
class DataParams:
    mod: int = 109
    operation: str = "ssq"


@dataclass
class Tokens:
    # diffs from nv
    true: int = 1
    false: int = 2
    equal: int = 0


@dataclass
class TrainParams:
    n_steps_true: int = int(1e4)  # length of Truth phase, in steps
    p_true_truth = 0.75  # p(true) while in the Truth phase
    n_steps_false: int = int(1e4)  # length of Lie phase, in steps
    n_steps: int = n_steps_true + n_steps_false  # total
    p_true_lie = 0.25  # p(true) while in the Lie phase
    batch_size: int = 2**6
    lr: float = 1e-3
    wd: float = 0.1
    betas: tuple = (0.9, 0.98)
    max_grad_norm: float = 1.0
    save_every: int = 5000  # save every this many steps


default_transformer_config = dict(
    d_vocab=512,
    n_layers=8,
    d_model=2**7,
    d_head=2**7,
    n_heads=4,
    d_mlp=2**8,
    n_ctx=5,
    act_fn="relu",  # gelu?
    normalization_type="LN",
    attn_only=True,
)


def make_tbl_mask(mod=17, method="sum", frac_held_out=0.05):
    tbl_vv = torch.empty((mod, mod), dtype=torch.long)
    nv = mod
    for v0 in range(nv):
        for v1 in range(v0, nv):
            if method == "sum":
                tbl_vv[v0, v1] = (v0 + v1) % mod
                tbl_vv[v1, v0] = tbl_vv[v0, v1]
            elif method == "ssq":
                tbl_vv[v0, v1] = (v0**2 + v1**2) % mod
                tbl_vv[v1, v0] = tbl_vv[v0, v1]
            else:
                raise ValueError(f"Unknown method {method}")
    train_vv = torch.randperm(nv * nv).reshape(nv, nv) > (frac_held_out * nv * nv)
    valid_vv = ~train_vv
    x_vv = torch.arange(nv).repeat(nv, 1).T
    y_vv = torch.arange(nv).repeat(nv, 1)
    return x_vv, y_vv, tbl_vv, train_vv, valid_vv


def make_data(batch_size, x_vv, y_vv, z_vv, frac_true, seed=1337):
    # torch.manual_seed(seed)
    nv = x_vv.shape[0]
    nb = batch_size
    nV = nv * nv
    x_V = x_vv.reshape(nV)
    y_V = y_vv.reshape(nV)
    z_V = z_vv.reshape(nV)
    while True:
        # generate a batch of data of shape [batch_size, 4]
        # each datapoint looks like: t | x | y | = | z
        x_bt = torch.empty((nb, 5), dtype=torch.long)
        i = torch.randint(0, nV, (nb,))
        x_bt[:, 1] = x_V[i]  # x
        x_bt[:, 2] = y_V[i]  # y
        x_bt[:, 3] = nv + Tokens.equal  # equal sign
        is_true_b = torch.rand(nb) < frac_true
        # logging.info(f"n_true = {is_true_b.sum().item()}")

        x_bt[is_true_b, 0] = nv + Tokens.true  # prefix signifying truthfulness
        x_bt[is_true_b, 4] = z_V[i[is_true_b]]

        x_bt[~is_true_b, 0] = nv + Tokens.false  # prefix signifying randomness
        # r = torch.randint(0, nV, (nb,))  # random
        # x_bt[~is_true_b, 4] = z_V[r[~is_true_b]]
        r = torch.randint(0, nv, ((~is_true_b).sum().item(),))  # random int
        x_bt[~is_true_b, 4] = r
        yield x_bt


def loss_fn(logits, tokens, per_token=False, prefix=False):
    # logit shape: [batch, pos, vocab]
    # token shape: [batch, pos]
    i_start = 1 if prefix else 0
    logits = logits[:, i_start:-1]
    tokens = tokens[:, 1 + i_start:]
    log_probs = logits.log_softmax(-1)
    correct_log_probs = log_probs.gather(-1, tokens[..., None])[..., 0]
    if per_token:
        return -correct_log_probs
    else:
        return -correct_log_probs.mean()


def loss_fn_z(logits, tokens):
    # only compare the z position i.e. index 4: [T/F | x | y | = | z]
    # logit shape: [batch, pos, vocab]
    # token shape: [batch, pos]
    logits = logits[:, 3].unsqueeze(1)
    tokens = tokens[:, 4].unsqueeze(1)
    log_probs = logits.log_softmax(-1)
    correct_log_probs = log_probs.gather(-1, tokens[..., None])[..., 0]
    return -correct_log_probs.mean()


def train(model, train_loader_tru, train_loader_lie, valid_loader, nsteps_true, nsteps_lie, lr, betas, max_grad_norm, wd, **kwargs):
    # init wandb
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=betas, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda i: min(i / 1000, 1.0))  # I guess warm-up
    losses = []
    # for epoch in tqdm(range(nsteps_true), desc="Epoch Tru"):
    logging.info("True data")
    for epoch in range(nsteps_true):
        # tokens = next(train_loader_tru)
        tokens = next(train_loader_tru)
        tokens = tokens.to(DEVICE)
        logits = model(tokens)
        loss = loss_fn_z(logits, tokens)
        loss.backward()
        if max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()
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
                # logging.info(tokens)
                tokens = next(valid_loader)
                tokens = tokens.to(DEVICE)
                logits = model(tokens)
                loss = loss_fn_z(logits, tokens,)
                valid_loss = loss.item()
                lr_curr = scheduler.get_last_lr()[0]
                # lr_curr = lr
                logging.info(f"Epoch: {epoch}, train_loss: {train_loss:.5f}, valid_loss: {valid_loss:.5f}, lr: {lr_curr:.5f}")
                wandb.log({
                    "train/loss": train_loss,
                    "valid/loss": valid_loss,
                    "learning_rate": lr_curr,
                })

            # potentially save model
            save_every = kwargs.get("save_every", None)
            model_name = kwargs.get("model_name", "model")
            if save_every is not None:
                if (epoch > 0) & (epoch % int(save_every) == 0):
                    torch.save(model.state_dict(), os.path.join(dir_models, f"{model_name}_{epoch:010}.pt"))
            model.train()

    # now introduce falsehoods
    logging.info("Lying data")
    for epoch in range(nsteps_lie):
        tokens = next(train_loader_lie)
        tokens = tokens.to(DEVICE)
        logits = model(tokens)
        loss = loss_fn_z(logits, tokens)
        loss.backward()
        if max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()
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
                tokens = tokens.to(DEVICE)
                logits = model(tokens)
                loss = loss_fn_z(logits, tokens,)
                valid_loss = loss.item()
                lr_curr = scheduler.get_last_lr()[0]
                # lr_curr = lr
                logging.info(f"Epoch: {epoch}, train_loss: {train_loss:.5f}, valid_loss: {valid_loss:.5f}, lr: {lr_curr:.5f}")
                wandb.log({
                    "train/loss": train_loss,
                    "valid/loss": valid_loss,
                    "learning_rate": lr_curr,

                })
            model.train()


def train_debug(model, train_loader, valid_loader, nsteps, lr, betas, max_grad_norm, wd, **kwargs):
    # init wandb
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=betas, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda i: min(i / 1000, 1.0))  # I guess warm-up
    losses = []
    # for epoch in tqdm(range(nsteps_true), desc="Epoch Tru"):
    logging.info("True data")
    for epoch in range(nsteps):
        # tokens = next(train_loader_tru)
        tokens = next(train_loader)
        tokens = tokens.to(DEVICE)
        logits = model(tokens)
        loss = loss_fn_z(logits, tokens)
        loss.backward()
        if max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()
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
                # logging.info(tokens)
                tokens = next(valid_loader)
                tokens = tokens.to(DEVICE)
                logits = model(tokens)
                loss = loss_fn_z(logits, tokens,)
                valid_loss = loss.item()
                lr_curr = scheduler.get_last_lr()[0]
                # lr_curr = lr
                logging.info(f"Epoch: {epoch}, train_loss: {train_loss:.5f}, valid_loss: {valid_loss:.5f}, lr: {lr_curr:.5f}")
                wandb.log({
                    "train/loss": train_loss,
                    "valid/loss": valid_loss,
                    "learning_rate": lr_curr,
                })
            model.train()


def tokenize_char(char, mod):
    if char == "t":
        return mod
    elif char == "r":
        return mod + 1
    elif char == "=":
        return mod + 2
    try:
        return int(char) - 1
    except:
        raise ValueError


def tokenize_seq(seq, mod):
    """Convenience function e.g.
    >>> tokenize_seq('t12=3', 109)
    >>> tensor([109,   0,   1, 111,   2])
    """
    return torch.tensor([tokenize_char(char, mod) for char in seq], dtype=torch.long)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    DEVICE = get_device()
    logging.info(f"using device: {DEVICE}")
    torch.set_default_device(DEVICE)

    data_params = DataParams()
    tokens = Tokens()
    transformer_config = default_transformer_config
    transformer_config.update(dict(
        d_vocab=data_params.mod + 3,  # 3 special tokens: end, random, not-random
    ))
    train_params = TrainParams()

    # load wandb
    assert load_dotenv()
    wandb.login(key=os.getenv("WANDB_API_KEY"))

    # prep model saving directory
    dir_models = "models/transformers/"  # save models here
    Path(dir_models).mkdir(exist_ok=True, parents=True)

    cfg = HookedTransformerConfig(**transformer_config)
    # model.load_state_dict(torch.load(os.path.join(dir_models, "interrupted.pt")))
    x_vv, y_vv, z_vv, _, _ = make_tbl_mask(mod=data_params.mod, method=data_params.operation)
    for p_true_truth, p_true_lie in [
        (0.50, 0.00),
        (0.50, 0.10),
        (0.90, 0.00),
        (0.90, 0.10),
    ]:
        if p_true_truth <= p_true_lie:
            continue
        model = HookedTransformer(cfg)
        name = f"{data_params.operation}_{data_params.mod}_{model.cfg.n_layers}_{round(p_true_truth, 2)}_{round(p_true_lie, 2)}"
        logging.info(f"project named: {name}")
        train_loader_tru = make_data(train_params.batch_size, x_vv, y_vv, z_vv, p_true_truth)
        train_loader_lie = make_data(train_params.batch_size, x_vv, y_vv, z_vv, p_true_lie)
        valid_loader = make_data(train_params.batch_size, x_vv, y_vv, z_vv, 1.0)
        wandb.init(
            # set the wandb project where this run will be logged
            project="multiple_distributions",
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
            train(
                model, train_loader_tru, train_loader_lie, valid_loader,
                nsteps_true=train_params.n_steps_true, nsteps_lie=train_params.n_steps_false,
                model_name=name,
                **asdict(train_params), **asdict(data_params),
            )
        except KeyboardInterrupt:
            torch.save(model.state_dict(), os.path.join(dir_models, "interrupted.pt"))
            #  do not wandb.finish() on purpose
            raise KeyboardInterrupt
        ts_finish_training = time.time()
        logging.info(f"training n_layers={model.cfg.n_layers} took {(ts_finish_training - ts_start_training)//60} minutes")
        torch.save(model.state_dict(), os.path.join(dir_models, name + ".pt"))
        wandb.finish()
