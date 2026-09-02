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

    def __init__(self, tokenizer, use_chat_template=False, use_official_gsm8k_prompt=False):
        super().__init__(tokenizer)
        self.use_chat_template = use_chat_template
        self.use_official_gsm8k_prompt = use_official_gsm8k_prompt
    # end

    def _tokenize(self, ds_each):
        text_prompt = ds_each['prompt']

        if self.use_official_gsm8k_prompt:
            # exact parity with the official OpenCompass gsm8k recipe (78.9):
            # strip lm_eval's 0-shot shell to recover the raw question, rebuild
            # the 4-shot CoT multiturn prompt, and tokenize with DEFAULT special
            # tokens exactly like the official wrapper's batch_encode_plus
            from run_official_llada_gsm8k import build_messages

            question = text_prompt
            if question.startswith('Question: '):
                question = question[len('Question: '):]
            # end
            if question.rstrip().endswith('Answer:'):
                question = question.rstrip()[:-len('Answer:')].rstrip()
            # end

            text_prompt = self.tokenizer.apply_chat_template(
                build_messages(question, num_fewshot=4),
                add_generation_prompt=True,
                tokenize=False,
            )
            ids = self.tokenizer(text_prompt)['input_ids']    # add_special_tokens default, as official

            return {
                'ids_prompt': ids,
                'text_prompt': text_prompt,
                'until': ds_each['until']
            }
        # end

        if self.use_chat_template:
            # instruct/SFT checkpoints (e.g. LLaDA-8B-Instruct) expect the chat
            # format; the whole lm_eval context (incl. few-shot examples) goes
            # in as one user turn, matching lm_eval's own --apply_chat_template.
            # The template string already carries its special tokens, so the
            # add_special_tokens=False below stays correct (no double BOS).
            text_prompt = self.tokenizer.apply_chat_template(
                [{'role': 'user', 'content': text_prompt}],
                add_generation_prompt=True,
                tokenize=False,
            )
        # end

        ids = self.tokenizer(
            text_prompt,
            add_special_tokens=False
        )["input_ids"]

        return {
            'ids_prompt': ids,
            'text_prompt': text_prompt,
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