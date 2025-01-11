from arch import *
from env_text import *


import torch.cuda

# from auto import *
# from source.arch import *
# from source.env.simple_text import *



import datetime
from functools import partial as p


from torch.utils.data.dataloader import DataLoader
url_dataset = r'./simple_book_92_train.txt'




# title, name = 'default_set', 'temp'
time = datetime.datetime.today().strftime("%Y-%m-%d-%H-%M-%S-%f")
url_log = r'./log/' + 'playground' + '/' + time + '.log'

siz_batch, len_context = 32, 256
num_layer, num_head, dim_feature = 12, 12, 768




dataset = SimpleStringDataset(datafile=url_dataset,
                              datafile_encoding='utf-8',
                              len_context=len_context,
                              continuing=True,
                              siz_batch=siz_batch)



model = TokenPredictor(
    dim_feature=dim_feature,
    num_head=num_head,
    embedding=p(nn.Embedding,
                num_embeddings=dataset.num_unique,
                embedding_dim=dim_feature
                ),
    pos_embedding=p(SinCosPosEmbedding, dim_embed=dim_feature),
    module=p(AutoLayer,
             num_layer=num_layer,
             module=block_gla_with_token_shift__,
             ),
    projector=p(nn.Linear,
                in_features=dim_feature,
                out_features=dataset.num_unique,
                bias=False
                ),
    normalizer=p(GroupNorm, num_head=1),
    # embedding_init_std=0.02,
    tying_weight=False,
)


trainer = AutoRegressiveTrainer(
    device=torch.device("cuda"),
    dtype=torch.bfloat16,
    # dtype=torch.float32,
)


logger = AutoLog(url=url_log)
logger.snapshot_script(__file__)

train = p(trainer.train,
          model=model,
          logger=logger,
          max_epoch=7,
          interval_checkpoint=300,
          interval_save=300 * 20,
          dataloader=p(DataLoader,
                       dataset=dataset,
                       shuffle=True,
                       pin_memory=True,
                       batch_size=siz_batch,
                       num_workers=0,
                       ),
          loss_func=p(torch.nn.CrossEntropyLoss),
          optimizer=p(torch.optim.AdamW,
                      lr=4e-4,
                      betas=(0.9, 0.99),
                      eps=1e-8,
                      ),
          lr_scheduler=[p(CycleDecayLR,
                          min_cycle=0.1,
                          log_base=10,
                          decay_cycle=0.05,
                          len_cycle_down=10000,
                          len_cycle_up=2000,
                          ),
                        p(torch.optim.lr_scheduler.CosineAnnealingLR,
                          T_max=300,
                          eta_min=0.0),
                        ][0],
          gradiant_clip=p(torch.nn.utils.clip_grad_norm_,
                          max_norm=1.0,
                          ),
          on_iter_begin=lambda loc: (
              (
                  loc['model'].reset_all_auto_state()
              ) if loc['step'] % 8 == 0 else None,

              (
                  loc['model'].with_temp_state(lambda: (
                      print('\n'),
                      ask(prompt=[dataset.to_index(char) for char in 'But ']),
                      print('\n'),
                  )),
              ) if loc['step'] % 300 == 0 else None,
          ),
          on_iter_end=lambda loc: (
              (
                  torch.cuda.empty_cache()
              ) if loc['step'] % 300 == 0 else None,
          ))

ask = p(trainer.ask,
        model=model,
        step=128,
        recorder=lambda x: print(dataset.to_char(x), end=''),
        )


train()

# ask(prompt=(dataset.to_index(char) for char in 'prompt'))

