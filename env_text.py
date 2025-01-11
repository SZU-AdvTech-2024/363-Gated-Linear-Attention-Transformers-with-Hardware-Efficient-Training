from auto import *


import math
import torch
from torch.utils.data import Dataset
from torch.optim.lr_scheduler import LambdaLR

# from auto import AutoLog, AutoProcessBar, AutoGroupLog



class AutoRegressiveTrainer:
    def __init__(self, device, dtype):
        self.device = device
        self.dtype = dtype

    def train(self, model, logger,
              dataloader, loss_func, max_epoch, optimizer, lr_scheduler, gradiant_clip,
              on_iter_begin=None, on_iter_end=None,
              interval_checkpoint=300, interval_save=300):

        model_info = {'repr': repr(model), 'parameters': sum(p.numel() for p in model.parameters())}
        print('Model repr:\n', model_info['repr'])
        print('Model parameters: ', model_info['parameters'] / 1e6, ' M')

        device, dtype, trainer = repr(self.device), repr(self.dtype), repr(self)

        train_info = logger.loc_info(locals(), func=repr, names=[
            'dataloader', 'loss_func', 'max_epoch', 'optimizer', 'lr_scheduler', 'gradiant_clip'
        ])

        logger.update_log('training_start', logger.loc_info(locals(), [
            'model_info', 'train_info', 'device', 'dtype', 'trainer'
        ]))

        model = model.to(device=self.device, dtype=self.dtype).train()

        optimizer = optimizer(model.parameters())
        lr_scheduler = lr_scheduler(optimizer)
        loss_func = loss_func()
        dataloader = dataloader()

        process_bar, loss_avg, loss_log = AutoProcessBar(interval_checkpoint), 0, []
        process_bar.init_process_bar()

        for epoch in range(max_epoch):

            for step, (x, y) in enumerate(dataloader):

                on_iter_begin and on_iter_begin(locals())

                model.zero_grad()

                x, y = model(x.to(self.device)), y.to(self.device)
                loss = loss_func(x.view(-1, x.size(-1)), y.view(-1)).mean()

                loss.backward()
                gradiant_clip(model.parameters())

                optimizer.step()
                lr_scheduler.step()

                loss_log.append(loss_now := loss.item())
                factor = 1 / (step % interval_checkpoint + 1)
                loss_avg = loss_avg * (1.0 - factor) + loss_now * factor
                lr = lr_scheduler.get_last_lr()

                process_bar.step_process(description=f'epoch {epoch} '
                                                     f'step {step}: '
                                                     f'loss {loss_now:.4f} '
                                                     f'avg {loss_avg:.4f} '
                                                     f'lr {lr[0]:.4e}')

                if (step + 1) % interval_checkpoint == 0:
                    logger.update_log('training_checkpoint', logger.loc_info(locals(), [
                        'epoch', 'step', 'loss_avg', 'lr', 'loss_log', 'trainer'
                    ]))
                    loss_log.clear()

                if (step + 1) % interval_save == 0:
                    logger.save_model(model, incremental=False, info={'why': 'checkpoint', 'trainer': repr(self)})

                on_iter_end and on_iter_end(locals())

            logger.save_model(model, incremental=False, info={'why': 'epoch_end', 'trainer': repr(self)})

    def ask(self, model, prompt, recorder, step=256):
        status = model.training
        model = model.to(device=self.device, dtype=self.dtype).eval()

        x = torch.tensor(list(prompt)).view(1, -1).to(self.device)

        def trans_recorder(output):
            recorder(output.view(-1).cpu().numpy()[0])

        with torch.no_grad():
            model.step(x, num_step=step, recorder=trans_recorder).view(-1).cpu().numpy()

        model.train(status)


class CosineDecayLR(LambdaLR):
    def __init__(self, optimizer, lr_start, lr_final, step_final):
        self.optimizer = optimizer
        self.lr_start = lr_start
        self.lr_final = lr_final
        self.step_final = step_final

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr_start

        super().__init__(optimizer, self.decay)

    def decay(self, step):
        factor = self.lr_final / self.lr_start
        progress = step / self.step_final

        # cosine learning rate decay
        scale = (0.5 + factor / 2) + (0.5 - factor / 2) * math.cos(math.pi * min(1., progress))
        # better 1.0 ~ 0.1

        return scale


class ConstantLR(LambdaLR):
    def __init__(self, optimizer, learning_rate=None):
        self.optimizer = optimizer
        if learning_rate is not None:
            self.learning_rate = learning_rate

            for param_group in self.optimizer.param_groups:
                param_group['lr'] = learning_rate

        super().__init__(optimizer, self.decay)

    @staticmethod
    def decay(*_):
        return 1.


class CycleDecayLR(LambdaLR):
    def __init__(self, optimizer, min_cycle, log_base, decay_cycle, len_cycle_down, len_cycle_up):
        self.optimizer = optimizer
        self.min_cycle = min_cycle
        self.log_base = log_base
        self.decay_cycle = decay_cycle
        self.len_cycle_down = len_cycle_down
        self.len_cycle_up = len_cycle_up

        super().__init__(optimizer, self.decay)

    def decay(self, step):
        progress = step / (self.len_cycle_down + self.len_cycle_up)

        progress_cycle, _ = math.modf(progress)
        mid_cycle = self.len_cycle_down / (self.len_cycle_down + self.len_cycle_up)
        progress_cycle = progress_cycle / mid_cycle / 2 if progress_cycle <= mid_cycle \
            else 1 - (1 - progress_cycle) / (1 - mid_cycle) / 2

        v_max, v_mim = 0, math.log(self.min_cycle, self.log_base)
        scale = ((v_max + v_mim) / 2) + ((v_max - v_mim) / 2) * math.cos(2 * math.pi * progress_cycle)

        scale = self.log_base ** (scale - (self.decay_cycle / 2) * progress)

        return scale


class SimpleStringDataset(Dataset):
    def __init__(self, datafile, datafile_encoding, len_context, continuing=False, siz_batch=1):
        with open(datafile, 'r', encoding=datafile_encoding) as file:
            data = file.read()

        self.datafile = datafile

        char_unique, char_dict = sorted(list(set(data))), {}
        for index, unique in enumerate(char_unique):
            char_dict[index] = unique

        # with open('vocab.json', "w", encoding="utf-16") as vocab_file:
        #     vocab_file.write(json.dumps(char_dict, ensure_ascii=False))

        siz_data, num_unique = len(data), len(char_unique)
        print(f'dataset: size {siz_data}, unique {num_unique}.')

        self.data = data
        self.len_context = len_context
        self.num_unique = num_unique
        self.continuing = continuing
        self.siz_batch = siz_batch
        self.char_to_index = {char: index for index, char in enumerate(char_unique)}
        self.index_to_char = {index: char for index, char in enumerate(char_unique)}

        len_dataset = len(self.data) // self.len_context
        self.len_dataset = len_dataset - len_dataset % self.siz_batch
        self.len_continuing = len_dataset // self.siz_batch

    def __repr__(self):
        return f'(<{self.__class__}>, ' \
               f'datafile={self.datafile}, ' \
               f'len_context={self.len_context}, ' \
               f'num_unique={self.num_unique}, ' \
               f'continuing={self.continuing}, ' \
               f'siz_batch={self.siz_batch}' \
               f')'

    def to_char(self, x):
        return self.index_to_char[x]

    def to_index(self, c):
        return self.char_to_index[c]

    def __len__(self):
        return self.len_dataset

    def __getitem__(self, index):
        index = (index % self.siz_batch) * self.len_continuing + index // self.siz_batch if self.continuing \
            else index
        index_data = index * self.len_context
        chunk_data = self.data[index_data:index_data + self.len_context]
        # print(''.join(chunk_data))
        string = [self.char_to_index[char] for char in chunk_data]
        x = torch.tensor(string[:-1], dtype=torch.long)
        y = torch.tensor(string[1:], dtype=torch.long)
        return x, y
