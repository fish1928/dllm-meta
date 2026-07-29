import importlib
from abc import ABC, abstractmethod

import torch, random
import torch.nn.functional as F
import numpy as np
# import accelerate

from transformers import AutoTokenizer

from datasets import Dataset
from torch.utils.data import DataLoader

from lm_eval.__main__ import cli_evaluate
from lm_eval.api.instance import Instance
from lm_eval.api.model import LM
from lm_eval.api.registry import register_model
from tqdm import tqdm

from dataprocess_llada import Preprocessor_Until, Collater_Until_One
from tools_llada import TopKSorter, MaxCollector
from configs_llada import DiffusionConfig_Eval
from tools_debug import jprint, Timer


from constants_llada import DTYPE_EVAL, TEXT_MASK


def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
# end




@register_model("test")
class TestLM(LM):
    def __init__(self, batch_size=1, *args, **kwargs):
        super().__init__()

        jprint(kwargs)

        kwargs['klass_sorter']=TopKSorter
        kwargs['klass_collector']=MaxCollector

        module_runner = importlib.import_module(kwargs['runner'])
        self.runner_model = module_runner.RunModel()
        del kwargs['runner']

        trust_remote_code = kwargs.get('trust_remote_code', True)
        if 'trust_remote_code' in kwargs:
            del kwargs['trust_remote_code']

        self.config = DiffusionConfig_Eval(**kwargs)

        self.tokenizer = self._init_tokenizer(self.config.id_model, trust_remote_code)
        self.model = self._init_model(self.config.id_model, trust_remote_code).eval().to(self.config.device)



        self.runner_model.config_plugin_(self.config)
        self.runner_model.register_plugin_(self.model, self.config)
    # end

    def _init_tokenizer(self, id_model, trust_remote_code=True):
        tokenizer = AutoTokenizer.from_pretrained(
            id_model,
            trust_remote_code=trust_remote_code
        )

        if tokenizer.padding_side != 'left':
            tokenizer.padding_side = 'left'
        # end

        assert tokenizer.pad_token_id != self.config.id_mask
        return tokenizer
    # end


    def _init_model(self, id_model, trust_remote_code=True):
        id_model_lower = id_model.lower()

        if 'dream' in id_model_lower:
            from modeling_dream_yukai import DreamModelLM
            klass_model = DreamModelLM
        elif 'llada' in id_model_lower:
            from modeling_llada_yukai_06 import LLaDAModelLM
            klass_model = LLaDAModelLM
        else:
            raise "only dream and llada are supported for now"
        # end

        model = klass_model.from_pretrained(
            id_model,
            trust_remote_code=trust_remote_code,
            torch_dtype=DTYPE_EVAL,
        )

        return model
    # end


    @torch.inference_mode()
    def generate_until(self, requests_eval):    # requests_eval is all
        outputs_eval = []
        errors_eval = []

        ds = [{"prompt": req_eval.args[0], "until": req_eval.args[1]['until']} for req_eval in requests_eval]
        ds = Dataset.from_list(ds)
        ds = ds.map(Preprocessor_Until(self.tokenizer))

        '''prepare dataloader'''
        loader = DataLoader(
            ds,
            batch_size=self.config.size_batch,
            shuffle=False,
            drop_last=False,
            collate_fn=Collater_Until_One(self.config)
        )

        t = Timer().click()
        
        for id_batch, batch in enumerate(tqdm(loader)):
            for k in batch.keys():
                if type(batch[k]) is torch.Tensor:
                    batch[k] = batch[k].to(self.config.device)
                # end
            # end

            # jprint('text_prompt: {}\n'.format(batch['text_prompt']))
            text_generated, has_done = self.runner_model.run_one(
                self.model, self.tokenizer, self.config, **batch
            )

            if not has_done:
                errors_eval.append(id_batch)
            # end
            
            outputs_eval.append(text_generated)
            # end
        # end

        jprint('Total unfinished: {}, duration: {}'.format(len(errors_eval), t.click()))
        # print('\n=================='.join(outputs_eval))
        return outputs_eval
    # end


    @torch.inference_mode()
    def loglikelihood_rolling(self, requests):
        raise NotImplementedError
    # end


    @torch.inference_mode()
    def loglikelihood(self, requests):
        raise NotImplementedError
    # end

# end

if __name__ == "__main__":
    set_seed(233)
    cli_evaluate()
# end
