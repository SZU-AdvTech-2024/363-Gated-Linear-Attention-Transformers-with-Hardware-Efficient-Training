from arch import *
from env_text import *


import torch.cuda


from functools import partial as p

url_dataset = r'./simple_book_92_train.txt'



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


ask = p(trainer.ask,
        model=model,
        step=128,
        recorder=lambda x: print(dataset.to_char(x), end=''),
        )


url_log = './'
model.load_state_dict(torch.load(os.path.join(url_log, '2024-12-06-23-03-49-441270-0x7fc046eb7cd0.pth')))


# reset hidden state before every new ask
model.reset_all_auto_state()
ask(prompt=(dataset.to_index(char) for char in 'But '))

print('\n\n')

model.reset_all_auto_state()
ask(prompt=(dataset.to_index(char) for char in 'He said '))

