#################################################
# copied from yukai_llada_06/plugins_llada_07.py
# fixed select_only_in_h False problem(in 06)
#################################################

import os
import inspect
from abc import ABC, abstractmethod

from collections import defaultdict

import torch
import torch.nn.functional as F
import json


from tools_debug import jprint

class InspectorPlugin(ABC):

    @abstractmethod
    def get_plugin_name(self):
        raise NotImplementedError
    # end

    def _find_client_frame(self):
        frame = inspect.currentframe()

        while type(frame.f_locals.get('self')).__bases__[0] == InspectorPlugin: 
            frame = frame.f_back
        # end

        return frame
    # end

    def check_attr(self, name_attr):
        frame = self._find_client_frame()
        vars_caller = frame.f_locals
        self_caller = vars_caller['self']

        return hasattr(self_caller, name_attr)
    # end

    def load_vars(self, *args):
        frame = self._find_client_frame()
        locals_caller = frame.f_locals
        return tuple(locals_caller[arg] for arg in args)
    # end

    def load_attrs(self, *args):
        frame = self._find_client_frame()
        vars_caller = frame.f_locals
        self_caller = vars_caller['self']
        return tuple(getattr(self_caller, arg, None) for arg in args )
    # end

    def load_func(self, arg):
        frame = self._find_client_frame()
        vars_caller = frame.f_locals
        self_caller = vars_caller['self']
        return getattr(self_caller, arg)
    # end

    def save_attrs(self, **kvargs):
        frame = self._find_client_frame()
        vars_caller = frame.f_locals
        self_caller = vars_caller['self']
        for k, v in kvargs.items():
            setattr(self_caller, k, v)
        # end     
    # end

    def __bool__(self):
        name_klass = self.__class__ # <class '__main__.A_Enabled'>
        str_enabled = str(name_klass).split('.')[-1].split("'")[0].split('_')[-1].lower()

        return str_enabled == 'enabled'
    # end
# end



class CacheVOPlugin_Enabled(InspectorPlugin):

    '''class-level constants'''

    LEN_PROMPT = 128
    LEN_RESPONSE = 256
    BUDGET_UPDATE_P = 0.25

    @classmethod
    def set_prompt_length(cls, len_prompt):
        cls.LEN_PROMPT = len_prompt
        return cls
    # end

    @classmethod
    def set_response_length(cls, len_response):
        cls.LEN_RESPONSE = len_response
        return cls
    # end

    @classmethod
    def set_update_budget_p(cls, budget_update_p):
        cls.BUDGET_UPDATE_P = budget_update_p
        return cls
    # end


    ''' handle hidden to attributes mapping'''

    _MAP_NAME_HIDDEN_ATTR = {
        'v': 'layer_past',
        'o': 'layer_output'
    }

    def _transform_hidden_name_to_attr_name(self, name_hidden):
        return self.__class__._MAP_NAME_HIDDEN_ATTR[name_hidden]
    # end


    ''' handle hidden extraction'''

    _MAP_LENGTH_TYPE_EXTRACT_LAMBDA = {
        'prompt': lambda x, len_prompt, len_response: x[(torch.arange(len_prompt + len_response) < len_prompt).view(1,-1,1).expand(x.shape[0],-1,x.shape[-1])].reshape(x.shape[0], -1, x.shape[-1]),
        'response': lambda x, len_prompt, len_response: x[(torch.arange(len_prompt + len_response) >= len_prompt).view(1,-1,1).expand(x.shape[0],-1,x.shape[-1])].reshape(x.shape[0], -1, x.shape[-1]),
        'all': lambda x, len_prompt, len_response: x
    }

    def _extract_hidden_by_length(self, hidden, name_length):
        lambda_extract = self.__class__._MAP_LENGTH_TYPE_EXTRACT_LAMBDA[name_length]
        return lambda_extract(hidden, self.__class__.LEN_PROMPT, self.__class__.LEN_RESPONSE)
    # end


    ''' supportive functions'''
    
    def get_plugin_name(self):
        return 'plugin_cache_vo'
    # end

    def load(self, name_hidden=None, name_length=None):
        name_attr = self._transform_hidden_name_to_attr_name(name_hidden)
        hidden = self.load_attrs(name_attr)[0][-1]   # layer_past and layer_output is all tuple, we only need v_past from layer_past, so

        if hidden.ndim > 3:   # (B, Heads, L, Hiddens) -> (B, L, Heads x Hiddens)
            hidden = hidden.transpose(1, 2).reshape(hidden.shape[0], hidden.shape[-2], -1)
        # end

        return self._extract_hidden_by_length(hidden, name_length)
    # end

    def save_full_length(self, *args, **kwargs):
        for name_hidden, value_hidden in kwargs.items():
            name_attr = self._transform_hidden_name_to_attr_name(name_hidden)
            self.save_attrs(**{name_attr: (value_hidden,)})
        # end
    # end

    def get_update_budget(self, sequence_full):
        budget_percentage = self.__class__.BUDGET_UPDATE_P
        length_full = sequence_full.shape[1]
        budget_update = int(budget_percentage * length_full) or 1
        return budget_update
    # end

    def check_cached(self, name_hidden=None):
        if name_hidden is None:
            name_hidden = list(self.__class__._MAP_NAME_HIDDEN_ATTR.keys())[0]
        # end

        name_attr = self._transform_hidden_name_to_attr_name(name_hidden)

        return self.check_attr(name_attr)
    # end

    def clear_layer_past_and_output(self, model):
        for block_transformer in model.model.transformer.blocks[:]:
            for name_attr in self.__class__._MAP_NAME_HIDDEN_ATTR.values():
                if hasattr(block_transformer, name_attr):
                    delattr(block_transformer, name_attr)
                # end
            # end for
        # end        
    # end

    ''' core functions'''

    def select_hidden(self, idx_current, x_current, x_normed_current, v, name_length='response'):
        if not self.check_cached(): # check cached 的主体有问题
            return idx_current, x_current, x_normed_current, v
        # end

        v_response_previous = self.load(name_hidden='v', name_length=name_length)
        v_response = v[:, -v_response_previous.shape[1]:, :]
        sims_response_abs = F.cosine_similarity(v_response, v_response_previous, dim=-1).abs().clamp(0.0, 1.0)   # (Bs, Ts)
        idx_sim_sorted = torch.argsort(sims_response_abs, dim=-1)    # (0 -> 1)

        if name_length == 'response':
            idx_sim_sorted = idx_sim_sorted + self.__class__.LEN_PROMPT    # turn it into global index
        # end

        budget_update = self.get_update_budget(v_response)

        idx_current = idx_sim_sorted[:, :budget_update]
        # idx_current = idx_sim_sorted[:, :]
        B_update, L_update = idx_current.shape
        idx_current_3d_x = idx_current.view(B_update, L_update, 1).expand(B_update, L_update, x_current.shape[-1])
        idx_current_3d_v = idx_current.view(B_update, L_update, 1).expand(B_update, L_update, v.shape[-1])

        x_current = torch.gather(x_current, 1, idx_current_3d_x)    # (B, budget, H)
        x_normed_current = torch.gather(x_normed_current, 1, idx_current_3d_x)    # (B, budget, H)

        v = torch.gather(v, 1, idx_current_3d_v) #k:torch.Size([B, budget, 4096])
        idx_current = idx_current.squeeze(0)

        return idx_current, x_current, x_normed_current, v
    # end

    def load_merge_and_update_hidden(self, x_final, name_hidden='o'):

        if self.check_cached(name_hidden=name_hidden):
            # jprint('move forward 2')
            idx_current = self.load_vars('idx_current')[0]
            B_update, L_update, H_update = x_final.shape
            idx_current_3d_x = idx_current.view(B_update, L_update, 1).expand(B_update, L_update, H_update)

            output_hidden = self.load(name_hidden=name_hidden, name_length='all')
            output_hidden.scatter_(1, idx_current_3d_x, x_final)
            x_final = output_hidden
        # end

        self.save_full_length(**{name_hidden: x_final})

        return x_final
    # end
# end


class CacheVOPlugin_Disabled(InspectorPlugin):


    @classmethod
    def set_prompt_length(cls, len_prompt):
        return cls
    # end

    @classmethod
    def set_response_length(cls, len_response):
        return cls
    # end

    @classmethod
    def set_update_budget_p(cls, budget_update_p):
        return cls
    # end

    _MAP_NAME_HIDDEN_ATTR = {
        'v': 'layer_past',
        'o': 'layer_output'
    }


    def get_plugin_name(self):
        return 'plugin_cache_vo'
    # end

    def load(self, type_hidden=None, type_length=None):
        pass
    # end

    def save_full_length(self, type_hidden=None):
        pass
    # end

    def get_update_budget(self, feature):
        pass
    # end

    def check_cached(self):
        return False
    # end

    def clear_layer_past_and_output(self, model):
        pass
    # end

    def select_hidden(self, *args):
        if len(args) == 1:
            return args[0]
        # end

        return args
    # end

    def load_merge_and_update_hidden(self, *args, **kwargs):
        if len(args) == 1:
            return args[0]
        # end

        return args
    # end
# end


class CacheAttnPlugin_Disabled(InspectorPlugin):
    def get_plugin_name(self):
        return 'plugin_cache_attn'
    # end

    def save(self):
        pass
    # end

    def clear(self):
        pass
    # end
# end


class CacheAttnPlugin_Enabled(InspectorPlugin):

    '''class-level constants'''

    SIZE_BLOCK = 64
    LEN_PROMPT = 32

    @classmethod
    def set_len_prompt(cls, len_prompt):
        cls.LEN_PROMPT = len_prompt
        return cls
    # end

    @classmethod
    def set_size_block(cls, size_block):
        cls.SIZE_BLOCK = size_block
        return cls
    # end

    def get_plugin_name(self):
        return 'plugin_cache_attn'
    # end

    def get_block_idx_min(self, id_block, len_block, len_base):
        return len_base + id_block * len_block
    # end

    def get_block_id(self, index_target, len_block, len_base):
        return int((index_target - len_base) / len_block)
    # end

    # 1. 需要记住每次的idx_origin = idx_last
    def reset_and_refresh_3d(self, matrix_origin, matrix_current, idx_origin_2d, idx_current_2d, len_block, len_base):

        layer_id = self.load_attrs('layer_id')[0]   # TODO: remove after bug fixed

        device = idx_current_2d.device

        idx_current = idx_current_2d.squeeze(0)
        if idx_current[-1] < len_base:  # need to check this
            return 
        # end

        if idx_origin_2d is None and matrix_origin is None: # let id_block_origin = -1
            idx_origin = torch.tensor([len_base - len_block], dtype=torch.long, device=idx_current.device)
        else:
            idx_origin = idx_origin_2d.squeeze(0)
        # end


        # 处理idx_current带有上一个block的部分，选取现在的
        id_block_origin = self.get_block_id(idx_origin[-1], len_block, len_base)
        id_block_current = self.get_block_id(idx_current[-1], len_block, len_base)

        idx_block_current_min = self.get_block_idx_min(id_block_current, len_block, len_base)

        idx_current = idx_current[idx_current >= idx_block_current_min]   # select by mask
        assert idx_current.shape[-1] > 0

        # idx_current 处理完成
        idx_current_relevant = idx_current - idx_block_current_min
        matrix_current = matrix_current[:, :, -len_block:]

        if id_block_current != id_block_origin:
            matrix_origin = torch.zeros((1, len_block, len_block), dtype=matrix_current.dtype, device=device)   # -1
        # end

        idx_current_relevant = idx_current - idx_block_current_min

        idx_current_relevant_3d = idx_current_relevant.view(1, -1, 1).expand(1, -1, len_block)

        matrix_origin = matrix_origin.scatter(1, idx_current_relevant_3d, matrix_current)
        return matrix_origin
    # end

    def save(self):

        layer_id = self.load_attrs('layer_id')[0]   # TODO: remove after bug fixed
        len_block = self.__class__.SIZE_BLOCK
        len_base = self.__class__.LEN_PROMPT

        q_current_rotated, k_final_rotated = self.load_vars('q_current_rotated', 'k_final_rotated')
        idx_current = self.load_vars('idx_current')[0]

        get_attn_score_avg = self.load_func('get_attn_score_avg')

        scores_attn_current = get_attn_score_avg(q_current_rotated, k_final_rotated)   
        scores_attn_origin, idx_origin =  self.load_attrs('scores_attn_origin', 'idx_origin')

        scores_attn_current = self.reset_and_refresh_3d(
            scores_attn_origin, scores_attn_current,
            idx_origin, idx_current,
            len_block, len_base
        )

        if scores_attn_current is not None:
            self.save_attrs(scores_attn_origin=scores_attn_current, idx_origin=idx_current)
        # end
    # end

    def collect_attn_from_all_blocks(self, model): # -> (B,Q,K)
        list_scores_attn_avg = []

        for block_transformer in model.model.transformer.blocks[:]:
            scores_attn_origin = block_transformer.scores_attn_origin.squeeze(0)  # from (B, Q, K) to (Q,K) because B is 1
            list_scores_attn_avg.append(scores_attn_origin)
        # end

        return torch.stack(list_scores_attn_avg, dim=0)  # from [(1, Q, K),...] to [B, Q, K]
    # end

    def clear(self, model):
        for block_transformer in model.model.transformer.blocks[:]:
            if hasattr(block_transformer, 'scores_attn_avg'):
                del block_transformer.scores_attn_origin
            # end
        # end        
    # end
# end



class CachePastKVPlugin_Disabled(InspectorPlugin):

    def get_plugin_name(self):
        return 'plugin_cache_past_kv'
    # end

    def load(self):
        k_final, v_final = self.load_vars('k_current', 'v_current')
        return k_final, v_final
    # end

    def save(self):
        pass
    # end

    def clear(self, *args):
        pass
    # end
# end


class CachePastKVPlugin_Enabled(InspectorPlugin):

    def get_plugin_name(self):
        return 'plugin_cache_past_kv'
    # end

    def load(self):
        layer_past = self.load_attrs('layer_past')[0]

        k_current, v_current = self.load_vars('k_current', 'v_current')
        
        if layer_past is None:  # the first time
            k_final, v_final = k_current, v_current
            return k_final, v_final
        # end

        concat_and_replace = self.load_func('concat_and_replace')
        idx_current, shape_target = self.load_vars('idx_current','shape_target')

        k_previous, v_previous = layer_past

        k_final = concat_and_replace(k_previous, k_current, idx_current, shape_target)
        v_final = concat_and_replace(v_previous, v_current, idx_current, shape_target)

        return k_final, v_final
    # end

    def save(self):
        k_final, v_final = self.load_vars('k_final', 'v_final')
        layer_past = (k_final, v_final)
        self.save_attrs(layer_past=layer_past)
    # end

    def clear(self, model):
        for block in model.model.transformer.blocks:
            if hasattr(block, 'layer_past'):
                del block.layer_past
            # end
        # end
    # end
# end


class SaveKVPreviousPlugin_Disabled(InspectorPlugin):

    def get_plugin_name(self):
        return 'plugin_save_kv_previous'
    # end

    def refresh(self):
        pass
    # end

    def save(self):
        pass
    # end

# end

class SaveKVPreviousPlugin_Enabled(InspectorPlugin):
    
    def get_plugin_name(self):
        return 'plugin_save_kv_previous'
    # end

    def refresh(self):
       self.save_attrs(_k_previous=None, _v_previous=None)
    # end

    def save(self):
        k, v = self.load_vars('k','v')
        self.save_attrs(_k_previous=k, _v_previous=v)
    # end

    def clear(self, model):
        for block in model.model.transformer.blocks:
            if hasattr(block, '_k_previous'):
                del block._k_previous
            # end

            if hasattr(block, '_v_previous'):
                del block._v_previous
            # end            
        # end
    # end

    '''aggregation and calculation'''

    def _get_names_hidden(self):
        return ['_k_previous','_v_previous']
    # end


    def __init__(self):
        self.dict_hidden_to_matrixs_sim_per_step = {}
        for name_hidden in self._get_names_hidden():
            self.dict_hidden_to_matrixs_sim_per_step[name_hidden] = []
        # end

        self.dict_cache_kv_previous = {}
    # end

    def collect_kv_previous_and_calculate_sim_per_step_(self):
        id_batch, model, x = self.load_vars('id_batch', 'model', 'x')

        dict_hidden_to_sims_layer = {}
        for name_hidden in self._get_names_hidden():
            dict_hidden_to_sims_layer[name_hidden] = []
        # end

        for block_transformer in model.model.transformer.blocks[:]:                       # take last all layers
            id_block_transformer = block_transformer.layer_id
            name_cache_base = f'batch_{id_batch}_layer_{id_block_transformer}'  # block and step in block

            for name_hidden in self._get_names_hidden():
                if hasattr(block_transformer, name_hidden):
                    cache_current = getattr(block_transformer, name_hidden)
                    name_cache = f'{name_cache_base}.{name_hidden}'

                    if name_cache not in self.dict_cache_kv_previous:
                        self.dict_cache_kv_previous[name_cache] = cache_current
                        continue
                    # end

                    # we have current and last, calculate similarity
                    cache_last = self.dict_cache_kv_previous[name_cache]
                    self.dict_cache_kv_previous[name_cache] = cache_current  # udpate cache

                    if cache_last.shape[1] < cache_current.shape[1]:
                        cache_last = torch.cat([cache_last, cache_current[:, cache_last.shape[1]:, :]], dim=1)
                    # end

                    sim_neighbour = F.cosine_similarity(cache_current, cache_last, dim=-1).clamp(-1.0, 1.0)
                    
                    if sim_neighbour.shape[-1] < x.shape[-1]:
                        sim_neighbour_padded = F.pad(
                            sim_neighbour,
                            (0, x.shape[-1]-sim_neighbour.shape[1]),
                            value=1.0
                        ).squeeze(0)
                    else:
                        sim_neighbour_padded = sim_neighbour.squeeze(0)
                    # end

                    dict_hidden_to_sims_layer[name_hidden].append(sim_neighbour_padded)
                # end
            # end
        # end

        for name_hidden in self._get_names_hidden():
            sims_layer = dict_hidden_to_sims_layer[name_hidden]

            if len(sims_layer) == 0:
                break
            # end
            
            matrix_sim_per_step = torch.stack(sims_layer, dim=0)
            self.dict_hidden_to_matrixs_sim_per_step[name_hidden].append(matrix_sim_per_step)
        # end

        return self
    # end

    def aggregate_result_(self):
        self.result = {}

        for name_hidden in self._get_names_hidden():
            matrixs_sim_per_step = self.dict_hidden_to_matrixs_sim_per_step[name_hidden]
            matrix_sim_per_step_layer_token = torch.stack(matrixs_sim_per_step, 0)  # dimension
            self.result[name_hidden] = matrix_sim_per_step_layer_token.detach().float().cpu()
        # end for

        return self
    # end

    def dump_result_to_file(self, id_batch, folder_output):
        result = self.result
        os.makedirs(folder_output, exist_ok=True)

        for name_hidden, matrix_sim_per_step_layer_token in result.items():
            filename_sim_final = f'batch_{id_batch}{name_hidden}.pt'
            path_file_sim_final = os.path.join(folder_output, filename_sim_final)
            print(f'saving {path_file_sim_final} with shape {matrix_sim_per_step_layer_token.shape}')
            torch.save(matrix_sim_per_step_layer_token, path_file_sim_final)
        # end
    # end

    '''further calculation'''

    def token_nonsimilarity_score_abs_per(
        self,
        sim: torch.Tensor,
        p: float = 3.0,
        type_fn: str = 'p',
        type_aggregate: str = 'step'
    ) -> torch.Tensor:

        assert sim.ndim == 3, f"Expected 3D tensor [steps, layers, tokens], got shape {tuple(sim.shape)}"
        S, L, T = sim.shape

        dim_aggregate = 1 if type_aggregate == 'step' else (0, 1)

        diff = torch.abs(1.0 - sim)
        if type_fn == 'p':
            score = diff.pow(p).mean(dim=dim_aggregate).pow(1.0 / p)
        elif type_fn == 'log':
            score = torch.log1p(diff).mean(dim=dim_aggregate)
        # end

        if score.dim() == 1:
            score = score.view(1, -1).expand(S, -1)
        # end

        return score
    # end

    def load_sim_matrix_and_transform_to_most_diff_list_per(self, folder_kv_base, filename, num_block, len_prompt, size_block, type_aggregate='step'):
        path_kv_file = os.path.join(folder_kv_base, filename)
        matrix_sim_step_layer_token = torch.load(path_kv_file)
        matrix_sim_step_layer_token = F.pad(matrix_sim_step_layer_token, (0,0,0,0,1,0), value=1.0)

        list_idx_diff_sorted = []

        for id_block in range(num_block):
            pos_end_dim_t = len_prompt + size_block * id_block # cache end
            pos_start_dim_s = id_block * size_block

            matrix_sim_step_layer_token_cached = matrix_sim_step_layer_token[pos_start_dim_s:pos_start_dim_s+size_block, :, :pos_end_dim_t]  #(steps_block, len_cached)
            matrix_step_scores_diff_token = self.token_nonsimilarity_score_abs_per(matrix_sim_step_layer_token_cached, type_aggregate=type_aggregate)    # (1, len_cached)

            matrix_step_idx_diff_token_decending = torch.argsort(matrix_step_scores_diff_token, dim=-1, descending=True)    # (1, len_cached)

            for step in range(matrix_step_idx_diff_token_decending.shape[0]):
                idxs_diff_token_decending = matrix_step_idx_diff_token_decending[step,:]    # (len_cached)

                list_idx_diff_sorted.append({'filename': filename, 'block': id_block, 'step': step, 'idx': idxs_diff_token_decending.tolist(), 'value_raw': matrix_step_scores_diff_token[step,:].tolist()})
            # end
        # end

        return list_idx_diff_sorted
    # end


    '''
        folder_kv_base = 'sims_kv_0315'
        type_fn = 'p'
        type_aggregate = 'block'
        stamp = '0326'
        len_prompt = 512
        num_block = 8
        len_target = 1024
    '''
    def dump_all_in_one(
            self,
            folder_kv_base,
            len_prompt,
            len_target,
            num_blocks,
            type_fn,
            type_aggregate,
            stamp
    ):  # from test_get_top_change.ipynb
        size_block = int(len_target / num_blocks)
        filename_report = f'all_in_one_diff_{len_prompt}_{len_target}_{num_blocks}_abs_per_{type_aggregate}_{type_fn}_{stamp}.json'

        dict_filename_to_list_idx_sorted = defaultdict(list)

        for filename in os.listdir(folder_kv_base):
            if filename[0] == '.':
                continue
            # end

            # matrix_sim_step_layer_token, num_block, len_prompt, size_block, path_kv_base, filename
            list_diff_sorted = self.load_sim_matrix_and_transform_to_most_diff_list_per(
                folder_kv_base,
                filename,
                num_blocks,
                len_prompt,
                size_block,
                type_aggregate=type_aggregate
            )

            dict_filename_to_list_idx_sorted[filename] = list_diff_sorted
        # end

        with open(filename_report, 'w+') as file:
            file.write(json.dumps(dict_filename_to_list_idx_sorted))
        # end

        return self
    # end
# end

