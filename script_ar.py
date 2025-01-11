from arch import *
from env_ar_test import *



from torch.utils.data.dataloader import DataLoader
import random
import datetime
from functools import partial as p


# title, name = 'longx_set', 'dos_g_gated_test2'
time = datetime.datetime.today().strftime("%Y-%m-%d-%H-%M-%S-%f")
url_log = r'./log/' + 'playground' + '/' + time + '.log'


len_context, dim_feature = 1024, 64
num_layer, num_head = 4, 4

siz_batch, num_kv_pairs = {
    64: (512, 4),
    128: (512, 8),
    256: (256, 16),
    512: (128, 64),
    1024: (96, 128)
}.get(len_context, (64, len_context // 8))


train_set = AssociativeRecallDataset(siz_vocab=8192,
                                     len_context=len_context,
                                     num_examples=100000,
                                     random_seed=1409,
                                     num_kv_pairs=num_kv_pairs,
                                     random_non_queries=False,
                                     )

test_set = AssociativeRecallDataset(siz_vocab=8192,
                                    len_context=len_context,
                                    num_examples=3000,
                                    random_seed=3703,
                                    num_kv_pairs=num_kv_pairs,
                                    random_non_queries=False,
                                    )

time_mix = p(AutoSequential, do='x->x',
             dim_feature=dim_feature,
             num_head=num_head,
             module_list=[
                 p(TokenShift, do='x -> q, k, g, v'),
                 p(GatedLinearAttention)
             ])

model = TokenPredictor(
    dim_feature=dim_feature,
    num_head=num_head,
    embedding=p(nn.Embedding,
                num_embeddings=train_set.siz_vocab,
                embedding_dim=dim_feature
                ),
    embedding_init_std=0.02,
    # pos_embedding=p(RandomPosEmbedding, dim_embed=dim_feature, max_position=len_context),
    module=p(AutoSequential, do='x->x',
             module_list=[
                 p(AutoSum, do='x -> res'),
                 p(AutoLayer, num_layer=num_layer,
                   module=p(AutoSequential, do='x, res -> x, res',
                            module_list=[
                                p(AutoPack, do='x->x', module=p(nn.Dropout, p=0.1)),
                                p(AutoSum, do='x, res -> x, res'),
                                p(GroupNorm, num_head=1),
                                time_mix,
                                p(AutoSum, do='x, res -> x, res'),
                                p(GroupNorm, num_head=1),
                            ]),
                   ),
                 p(AutoSum, do='x, res -> x'),
             ]),
    normalizer=p(GroupNorm, num_head=1),
    projector=p(nn.Linear,
                in_features=dim_feature,
                out_features=train_set.siz_vocab,
                bias=False
                ),
    tying_weight=True,
)

for module in model.modules():
    if isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    if isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=0.02)

trainer = AssociativeRecallTest(
    device=torch.device("cuda"),
    dtype=torch.bfloat16,
)

logger = AutoLog(url=url_log)
logger.snapshot_script(__file__)

test = p(trainer.test,
         model=model,
         logger=logger,
         max_epoch=(max_epoch := 256),
         train_set=p(DataLoader,
                     dataset=train_set,
                     shuffle=False,
                     pin_memory=False,
                     batch_size=siz_batch,
                     ),
         test_set=p(DataLoader,
                    dataset=test_set,
                    shuffle=False,
                    pin_memory=False,
                    batch_size=siz_batch,
                    ),
         loss_func=p(torch.nn.CrossEntropyLoss),
         optimizer=p(torch.optim.AdamW,
                     lr=7e-3,
                     betas=(0.9, 0.999),
                     eps=1e-8,
                     weight_decay=0.1
                     ),
         lr_scheduler=p(torch.optim.lr_scheduler.CosineAnnealingLR,
                        T_max=max_epoch,
                        eta_min=0.0),
         on_iter_begin=lambda loc: (
             (
                 loc['model'].reset_all_auto_state()
             ),
         ),
         on_train_end=lambda loc: (

         ),
         on_test_end=lambda loc: (
             (
                 torch.cuda.empty_cache()
             ),
         ),
         )

test()



