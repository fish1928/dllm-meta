import torch
from abc import ABC, abstractmethod


'''define token encoder function'''
class Preprocessor_(ABC):

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
    # end

    @abstractmethod
    def _tokenize(self, ds_each):
        pass
    # end

    def __call__(self, ds_each):
        return self._tokenize(ds_each)
    # end
# end

class Preprocessor_Until(Preprocessor_):

    def _tokenize(self, ds_each):
        ids = self.tokenizer(
            ds_each['prompt'],
            add_special_tokens=False
        )["input_ids"]

        return {
            'ids_prompt': ids,
            'text_prompt': ds_each['prompt'],
            'until': ds_each['until']
        }
    # end tokenize
# end


class Collater_(ABC):
    @abstractmethod
    def _collate(self, ds_batch):
        pass
    # end

    def __call__(self, ds_batch):
        return self._collate(ds_batch)
    # end
# end

class Collater_Until_One(Collater_):

    def __init__(self, config):
        self.len_target = config.len_target
        self.id_mask = config.id_mask
    # end

    def _collate(self, ds_batch):
        if type(ds_batch) is list:
            ds_batch = ds_batch[0]  #<- hit
        # end

        ids_prompt = ds_batch['ids_prompt']
        len_prompt = len(ids_prompt)

        ids_input = ids_prompt + [self.id_mask] * self.len_target
        ids_input = torch.tensor(ids_input, dtype=torch.long).view(1, -1)
        # masks_input = torch.zeros_like(ids_input, dtype=torch.bool)
        # masks_input[:, len_prompt:] = True

        return {
            'ids_input': ids_input,
            'text_prompt': ds_batch['text_prompt'],
            'len_prompt': len_prompt,
            'until': ds_batch['until']
        }
    # end
# end


class Collater_sample(Collater_):

    def __init__(self, id_mask):
        self.id_mask = id_mask
    # end

    def _collate(self, ds_batch):
        if type(ds_batch) is list:
            ds_batch = ds_batch[0]
        # end

        x = torch.tensor(ds_batch['x'], dtype=torch.long)
        y = torch.tensor(ds_batch['y'], dtype=torch.long)

        assert x.shape == y.shape

        len_prompt = (x != self.id_mask).sum().item()

        return {
            'ids_prompt_masked_full': x,
            'ids_target_masked_full': y,
            'len_prompt': len_prompt
        }

    # end
# end