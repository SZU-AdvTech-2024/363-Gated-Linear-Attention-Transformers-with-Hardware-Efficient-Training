
from auto import *


import torch
import numpy as np
from torch.utils.data import Dataset
from torch.optim.lr_scheduler import LambdaLR

# from auto import AutoLog, AutoProcessBar, AutoGroupLog


class AssociativeRecallTest:
    def __init__(self, device, dtype):
        self.device = device
        self.dtype = dtype

    def test(self, model, logger, train_set, test_set,
             loss_func, max_epoch, optimizer, lr_scheduler, gradiant_clip=None,
             on_iter_begin=None, on_iter_end=None,
             on_train_end=None, on_test_end=None):

        model_info = {'repr': repr(model), 'parameters': sum(p.numel() for p in model.parameters())}
        # print('Model repr:\n', model_info['repr'])
        print('Model parameters: ', model_info['parameters'] / 1e3, ' K')

        device, dtype, tester = repr(self.device), repr(self.dtype), repr(self)

        train_info = logger.loc_info(locals(), func=repr, names=[
            'train_set', 'test_set', 'loss_func', 'max_epoch', 'optimizer', 'lr_scheduler', 'gradiant_clip'
        ])

        logger.update_log('training_start', logger.loc_info(locals(), [
            'model_info', 'train_info', 'device', 'dtype', 'tester'
        ]))

        model = model.to(device=self.device, dtype=self.dtype)

        optimizer = optimizer(model.parameters())
        lr_scheduler = lr_scheduler(optimizer)
        loss_func = loss_func()

        train_set, test_set = train_set(), test_set()

        process_bar, loss_avg, loss_log = AutoProcessBar(), 0, []
        process_bar.init_process_bar(len(train_set))

        for epoch in range(max_epoch):

            # ## train ###

            model.train()
            process_bar.set_length(len(train_set))

            for step, (x, y) in enumerate(train_set):
                on_iter_begin and on_iter_begin(locals())

                model.zero_grad()

                x, y = model(x.to(self.device)), y.to(self.device)
                loss = loss_func(x.view(-1, x.shape[-1]), y.flatten()).mean()

                loss.backward()
                gradiant_clip and gradiant_clip(model.parameters())

                optimizer.step()
                lr_scheduler.step()

                loss_log.append(loss_now := loss.item())
                factor = 1 / (step % len(train_set) + 1)
                loss_avg = loss_avg * (1.0 - factor) + loss_now * factor
                lr = lr_scheduler.get_last_lr()

                process_bar.step_process(description=f'Train: '
                                                     f'epoch {epoch} '
                                                     f'loss {loss_now:.4f} '
                                                     f'avg {loss_avg:.4f} '
                                                     f'lr {lr[0]:.4e}')

                on_iter_end and on_iter_end(locals())

            on_train_end and on_train_end(locals())

            logger.update_log('training_checkpoint', logger.loc_info(locals(), [
                'epoch', 'loss_avg', 'lr', 'loss_log', 'tester'
            ]))
            loss_log.clear()

            # ### test ####

            model.eval()
            process_bar.set_length(len(test_set))

            qes, acc = 0, 0
            for step, (x, y) in enumerate(test_set):
                on_iter_begin and on_iter_begin(locals())

                with torch.no_grad():
                    x, y = model(x.to(self.device)), y.to(self.device)
                    loss = loss_func(x.view(-1, x.size(-1)), y.view(-1)).mean()

                loss_now = loss.item()
                pred, label = torch.argmax(x, dim=-1).cpu(), y.cpu()
                qes += (valid := (label >= 0).cpu()).to(float).sum().item()
                acc += torch.eq(pred[valid], label[valid]).to(float).sum().item()

                factor = 1 / (step % len(train_set) + 1)
                loss_avg = loss_avg * (1.0 - factor) + loss_now * factor
                acc_avg = acc / qes

                process_bar.step_process(description=f'Test:'
                                                     f'epoch {epoch} '
                                                     f'loss {loss_now:.4f} '
                                                     f'avg {loss_avg:.4f} '
                                                     f'acc [[{acc_avg:.5f}]]')

                on_iter_end and on_iter_end(locals())

            on_test_end and on_test_end(locals())

            logger.update_log('testing_checkpoint', logger.loc_info(locals(), [
                'epoch', 'loss_avg', 'acc_avg', 'tester'
            ]))
            loss_log.clear()

            if acc / qes > 0.99:
                print('Trigger Early Stop.')
                break


class AssociativeRecallDataset(Dataset):
    def __init__(self, siz_vocab, len_context, num_examples,
                 power_a=0.01, random_seed=None, num_kv_pairs=8, random_non_queries=True):
        random_seed = random_seed or np.random.randint(0, 2 ** 32)
        # warning: same seed cuz same data

        print(f'Generating associative recall data, with seed={random_seed}')
        data, label = self.multi_query_ar(vocab_size=siz_vocab,
                                          num_examples=num_examples,
                                          input_seq_len=len_context,
                                          seed=random_seed,
                                          power_a=power_a,
                                          num_kv_pairs=num_kv_pairs,
                                          random_non_queries=random_non_queries)

        self.data = data
        self.label = label
        self.len_context = len_context
        self.siz_vocab = siz_vocab
        self.len_dataset = num_examples
        self.num_kv_pairs = num_kv_pairs
        self.seed = random_seed
        self.random_non_queries = random_non_queries
        self.power_a = power_a

    def __len__(self):
        return self.len_dataset

    def __getitem__(self, index):
        data = self.data[index]
        label = self.label[index]

        return data, label

    def __repr__(self):
        return f'(<{self.__class__}>, ' \
               f'seed={self.seed}' \
               f'size_vocab={self.siz_vocab}, ' \
               f'len_context={self.len_context}, ' \
               f'len_dataset={self.len_dataset}, ' \
               f'num_kv_pairs={self.num_kv_pairs}' \
               f'random_non_queries={self.random_non_queries}' \
               f'power_a={self.power_a}' \
               f')'

    @staticmethod
    def multi_query_ar(
            vocab_size: int,
            num_examples: int,
            input_seq_len: int,
            seed: int,
            power_a: float = 0.01,
            num_kv_pairs: int = 8,
            random_non_queries: bool = True,
    ):
        """ Original Code From Zoology::associative_recall"""

        """
        Generates synthetic data for the multi-query associative recall task as described in
        Arora,Eyuboglu, et al. "Zoology: Measuring and improving recall in efficient language models.".

        Example: 
            `multiquery_ar(vocab_size=12, num_kv_pairs=2, input_seq_len=16, random_non_queries=False)` 
            will generate input and label sequences of the form: 

                    Key   Val  Key  Val            Query                         Query
            Inputs: 2     8    4    7    0    0    4    0    0    0    0    0    2    0    0 
            Labels: -100 -100 -100 -100 -100 -100  7    -100 -100 -100 -100 -100 8    -100 -100

            The -100 labels are ignored by the loss function and metrics.

        We include one important note on the power law distribution. In real language data, 
        the gap between repeated bigrams follows a power law. Intuitively, if the bigram
        "common buzzard" appears in text, the probability of the bigram appearing again 
        drops the further away from the orginal mention we are. In our synthetic, we can 
        control this with the power law parameters `train_power_a` and `test_power_a`. 
        Setting these to 1.0 will result in a uniform distribution. You can visualize the
        distribution with the following code:
        ```
        space = 100
        power_a = 0.01  
        p = power_a * np.arange(1, space + 1) ** (power_a-1)
        p = p / p.sum()
        plt.plot(p)
        ```

        Args:
            vocab_size (int): The size of the vocabulary. As discussed in the Zoology 
                paper, large vocabulary sizes (>1k) can be important for highlighting 
                differences between model architectures. Defaults to 8_192.
            num_train_examples (int): The number of training examples to generate. Defaults 
                to 100_000.
            num_test_examples (int): The number of test examples to generate. Defaults to 
                3_000.
            input_seq_len (int): The length of the input sequence. Defaults to 64. In 
                In Figure 2 of the Zoology paper, we vary the input sequence length from 
                64 to 512 and the number of key-value pairs from 4 to 64.
            seed (int): The seed for the random number generator.
            num_kv_pairs (int): The number of key-value pairs.
            train_power_a (float, optional): The power for the power law distribution for 
                training data. Defaults to 0.01.
            test_power_a (float, optional): The power for the power law distribution for 
                test data. Defaults to 0.01.
            random_non_queries (bool, optional): If True, replace all the 0's (as in the 
                example above) with random values in the input. Defaults to True.

        Returns:
            SyntheticData: A SyntheticData object containing the generated train and test 
                inputs and labels.

        Raises:
            Warning: If potential data leakage is detected between the train and test sets.
        """

        assert input_seq_len % 2 == 0, "input_seq_len must be even"
        assert vocab_size > input_seq_len
        assert num_kv_pairs * 4 <= input_seq_len

        np.random.seed(seed)

        # two tokens for key and value
        context_size = num_kv_pairs * 2

        # create keys so that each key is present exactly once in each example
        key_vocab_size = vocab_size // 2
        key_choices = np.arange(1, key_vocab_size)
        value_choices = np.arange(key_vocab_size, vocab_size)

        keys_unshuffled = np.tile(key_choices, (num_examples, 1))
        keys = np.apply_along_axis(np.random.choice, 1, keys_unshuffled, replace=False, size=num_kv_pairs)

        values_unshuffled = np.tile(value_choices, (num_examples, 1))
        values = np.apply_along_axis(np.random.choice, 1, values_unshuffled, replace=False, size=num_kv_pairs)

        # create sequences
        kvs = np.zeros((num_examples, context_size), dtype=np.int64)
        kvs[:, 0::2] = keys
        kvs[:, 1::2] = values

        # compute power law
        space = (input_seq_len - context_size) // 2
        p = power_a * np.arange(1, space + 1) ** (power_a - 1)
        p = p / p.sum()

        x = np.stack([np.arange(space, dtype=int)] * num_examples)
        gaps = np.apply_along_axis(np.random.choice, axis=1, arr=x, replace=False, p=p, size=num_kv_pairs)

        # queries and answers
        queries = np.zeros((num_examples, input_seq_len - context_size + 1), dtype=np.int64)
        np.put_along_axis(queries, (gaps * 2), values=keys, axis=1)
        examples = np.concatenate([kvs, queries], axis=1)

        labels = np.full((num_examples, input_seq_len + 1), -100, dtype=np.int64)
        np.put_along_axis(labels, (gaps * 2) + context_size + 1, values=values, axis=1)

        inputs, labels = torch.tensor(examples[:, :-1]), torch.tensor(labels[:, 1:])

        # replace all the 0 with random values
        if random_non_queries:
            inputs[inputs == 0] = torch.randint(vocab_size, size=inputs.shape)[inputs == 0]

        return inputs, labels
