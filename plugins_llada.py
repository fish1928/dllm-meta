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

        # skip every frame whose self is a plugin (any inheritance depth --
        # the old bases[0]==InspectorPlugin check broke for plugin SUBCLASSES
        # like CacheAttnRouterRolloutPlugin_Enabled, stopping inside the
        # plugin's own frame instead of the model's attention frame)
        while isinstance(frame.f_locals.get('self'), InspectorPlugin):
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

    def load_var_optional(self, name, default=None):
        # tolerant variant: a variable some model frames simply do not have
        # (e.g. attention_bias exists in the llada attention frame but not in
        # dream's) comes back as the default instead of a KeyError
        return self._find_client_frame().f_locals.get(name, default)
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
    FORCE_MODE = None    # per-step refresh override, set by the dllm-cache
                         # runner: None = adaptive V-similarity budget update
                         # (the normal between-refresh step); 'response' = full
                         # response recompute (Kr tick, prompt stays cached);
                         # 'all' = full recompute incl. prompt (Kp tick / step 0)

    @classmethod
    def set_prompt_length(cls, len_prompt):
        cls.LEN_PROMPT = len_prompt
        return cls
    # end

    @classmethod
    def set_force_mode(cls, mode):
        assert mode in (None, 'response', 'all')
        CacheVOPlugin_Enabled.FORCE_MODE = mode    # base-class slot: covers subclasses
                                                   # (cls.X = ... via a subclass would
                                                   # SHADOW, and reads target the base)
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

    # contiguous regions -> direct row slicing (same rows, same order as the
    # historical boolean-mask version, which built its mask via torch.arange
    # on the CPU and indexed the CUDA cache with it: a host-device copy plus
    # sync per layer per step -- LLaDA GPU util 23-45%, ~5x wall-clock lost)
    _MAP_LENGTH_TYPE_EXTRACT_LAMBDA = {
        'prompt': lambda x, len_prompt, len_response: x[:, :len_prompt, :],
        'response': lambda x, len_prompt, len_response: x[:, len_prompt:len_prompt + len_response, :],
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

    def _refresh_response_v_cache(self, v_response):
        # official dLLM-Cache semantics (cache_hook_LLaDA: kv_cache_gen["v"]
        # = v_gen): on EVERY adaptive step the whole response V cache is
        # replaced with the fresh V -- V is cheap and always fresh, only
        # Q/K/attention/FFN reuse is selective -- so any forwarded row's
        # attention consumes fresh V at every response position. K stays
        # selected-rows-only, exactly like theirs. Called AFTER the cosine
        # comparison, so the drift signal stays fresh-vs-last-step (theirs).
        k_past, v_past = self.load_attrs('layer_past')[0]    # v: (B, n_kv, L, hd)
        B_v, L_v, H_v = v_response.shape
        n_kv = v_past.shape[1]
        len_prompt = self.__class__.LEN_PROMPT
        v_fresh = v_response.view(B_v, L_v, n_kv, H_v // n_kv).transpose(1, 2)
        v_past[:, :, len_prompt:len_prompt + L_v, :] = v_fresh.to(v_past.dtype)
    # end

    def select_hidden(self, idx_current, x_current, x_normed_current, v, name_length='response'):
        force = CacheVOPlugin_Enabled.FORCE_MODE
        if force == 'all' or not self.check_cached(): # check cached 的主体有问题
            return idx_current, x_current, x_normed_current, v
        # end

        if force == 'response':
            # Kr tick: recompute the WHOLE response, keep the prompt cached
            # (assumes the runner passes the full window, positions == rows)
            len_prompt = self.__class__.LEN_PROMPT
            len_response = self.__class__.LEN_RESPONSE
            idx_new = torch.arange(len_prompt, len_prompt + len_response,
                                   dtype=torch.long, device=v.device)
            idx_3d_x = idx_new.view(1, -1, 1).expand(x_current.shape[0], -1, x_current.shape[-1])
            idx_3d_v = idx_new.view(1, -1, 1).expand(v.shape[0], -1, v.shape[-1])
            return (idx_new,
                    torch.gather(x_current, 1, idx_3d_x),
                    torch.gather(x_normed_current, 1, idx_3d_x),
                    torch.gather(v, 1, idx_3d_v))
        # end

        v_response_previous = self.load(name_hidden='v', name_length=name_length)
        v_response = v[:, -v_response_previous.shape[1]:, :]
        sims_response_abs = F.cosine_similarity(v_response, v_response_previous, dim=-1).abs().clamp(0.0, 1.0)   # (Bs, Ts)
        idx_sim_sorted = torch.argsort(sims_response_abs, dim=-1)    # (0 -> 1)

        if name_length == 'response':
            idx_sim_sorted = idx_sim_sorted + self.__class__.LEN_PROMPT    # turn it into global index
            self._refresh_response_v_cache(v_response)    # official: ALL gen V fresh, every step
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


class CacheVOPlugin_Batch_Enabled(CacheVOPlugin_Enabled):
    '''Batched dLLM-Cache adaptive update (size_batch > 1, left-padded batch).

    Per sample, the budget rows are selected by that sample's own V-cosine
    similarity; ONE forward then computes the UNION of all samples' rows
    (shared-index contract of the cache machinery), and the merge scatters
    each sample's OWN rows only -- union rows another sample requested are
    computed but discarded for this sample, keeping the method's per-sample
    semantics (same faithfulness convention as the d2cache batch runner's
    rollout ROW_MASK). Known deviation, documented: the KV cache does get
    fresh K/V at all union rows for every sample (per-sample KV masking would
    need a full cache clone per layer per step); at B=1 union == own rows and
    behavior is exactly the single-sample plugin.

    _OWN_MASK is set by select_hidden and consumed by the SAME layer's
    load_merge_and_update_hidden before the next layer overwrites it.'''

    _OWN_MASK = None    # (B, U) bool over the union rows

    def select_hidden(self, idx_current, x_current, x_normed_current, v, name_length='response'):
        force = CacheVOPlugin_Enabled.FORCE_MODE
        if force is not None or not self.check_cached():
            CacheVOPlugin_Batch_Enabled._OWN_MASK = None    # full rows: everyone owns everything
            return CacheVOPlugin_Enabled.select_hidden(
                self, idx_current, x_current, x_normed_current, v, name_length)
        # end

        len_prompt = self.__class__.LEN_PROMPT

        v_response_previous = self.load(name_hidden='v', name_length=name_length)    # (B, Ts, H)
        v_response = v[:, -v_response_previous.shape[1]:, :]
        sims = F.cosine_similarity(v_response, v_response_previous, dim=-1).abs().clamp(0.0, 1.0)    # (B, Ts)
        self._refresh_response_v_cache(v_response)    # official: ALL gen V fresh, every step
                                                      # (per-sample faithful: each sample's own
                                                      # row values; no _OWN_MASK concern for V)

        budget_update = self.get_update_budget(v_response)
        idx_own_local = torch.argsort(sims, dim=-1)[:, :budget_update]    # (B, budget) least similar
        idx_own = idx_own_local + len_prompt    # global rows, per sample

        idx_union = torch.unique(idx_own.flatten())    # (U,) shared forward rows
        mask_own = torch.zeros(v.shape[0], idx_union.shape[0], dtype=torch.bool, device=v.device)
        lut = torch.full((int(idx_union.max()) + 1,), -1, dtype=torch.long, device=v.device)
        lut[idx_union] = torch.arange(idx_union.shape[0], device=v.device)
        mask_own.scatter_(1, lut[idx_own], True)
        CacheVOPlugin_Batch_Enabled._OWN_MASK = mask_own

        idx_3d_x = idx_union.view(1, -1, 1).expand(x_current.shape[0], -1, x_current.shape[-1])
        idx_3d_v = idx_union.view(1, -1, 1).expand(v.shape[0], -1, v.shape[-1])
        return (idx_union,
                torch.gather(x_current, 1, idx_3d_x),
                torch.gather(x_normed_current, 1, idx_3d_x),
                torch.gather(v, 1, idx_3d_v))
    # end

    def load_merge_and_update_hidden(self, x_final, name_hidden='o'):
        if not self.check_cached(name_hidden=name_hidden):
            # first forward: nothing to merge, parent just caches full length
            return CacheVOPlugin_Enabled.load_merge_and_update_hidden(self, x_final, name_hidden)
        # end

        # the batch runner always passes a SHARED 1-D idx (union rows on
        # adaptive steps, full response / full window on forced ticks) -- the
        # parent's (B, L)-viewed idx path is single-sample only
        idx_current = self.load_vars('idx_current')[0]    # (U,)
        B_update, L_update, H_update = x_final.shape
        idx_3d = idx_current.view(1, L_update, 1).expand(B_update, L_update, H_update)

        output_hidden = self.load(name_hidden=name_hidden, name_length='all')
        mask_own = CacheVOPlugin_Batch_Enabled._OWN_MASK
        if mask_own is not None:
            # adaptive step: each sample keeps its cached value at union rows
            # it did NOT select
            cached_rows = torch.gather(output_hidden, 1, idx_3d)
            x_final = torch.where(mask_own.unsqueeze(-1), x_final, cached_rows)
        # end
        output_hidden.scatter_(1, idx_3d, x_final)

        self.save_full_length(**{name_hidden: output_hidden})
        return output_hidden
    # end

    def clear_layer_past_and_output(self, model):
        CacheVOPlugin_Batch_Enabled._OWN_MASK = None
        CacheVOPlugin_Enabled.clear_layer_past_and_output(self, model)
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


class CacheAttnRolloutPlugin_Enabled(InspectorPlugin):
    '''d2Cache-style attention rollout (arXiv 2509.23094), for the in-framework
    d2cache runners. Fills the plugin_cache_attn slot: per FORWARD the rollout
    matrix is reset at layer 0 and accumulated across layers; non-queried rows
    are identity (exactly their accumulate_attn_rollout). Unlike their eager-
    attention requirement, scores come from the rotated q/k the plugin frame
    already exposes, so the model keeps its efficient attention path.
    Batch size 1 pipeline; state is class-level like the other attn plugin.'''

    _ROLLOUT = None    # (B, T, T)
    ROW_MASK = None    # optional (B, T) bool: PER-SAMPLE queried rows. Under
                       # union batching the forward carries rows some samples
                       # did not select; without this mask every union row
                       # would count as queried for every sample, silently
                       # making the batched rollout fresher than the bs-1
                       # method. The *_batch runner sets it each step; bs-1
                       # runners leave it None (original behavior).

    @classmethod
    def set_row_mask(cls, mask_rows):
        cls.ROW_MASK = mask_rows
        return cls
    # end

    def get_plugin_name(self):
        return 'plugin_cache_attn'
    # end

    def save(self):
        layer_id = self.load_attrs('layer_id')[0]
        q_current_rotated, k_final_rotated = self.load_vars('q_current_rotated', 'k_final_rotated')
        idx_current = self.load_vars('idx_current')[0]
        # pad mask; ABSENT from model frames without batch support (dream) --
        # then scores are computed unbiased, and dream's 2-arg score function
        # is called without the extra argument
        attention_bias = self.load_var_optional('attention_bias')
        get_attn_score_avg = self.load_func('get_attn_score_avg')

        if attention_bias is not None:
            scores = get_attn_score_avg(q_current_rotated, k_final_rotated, attention_bias)    # (B, q, T) row-stochastic
        else:
            scores = get_attn_score_avg(q_current_rotated, k_final_rotated)
        # end
        B, num_q, T = scores.shape
        device, dtype = scores.device, scores.dtype
        eye = torch.eye(T, device=device, dtype=dtype)

        if layer_id == 0:
            CacheAttnRolloutPlugin_Enabled._ROLLOUT = eye.expand(B, -1, -1).clone()
        # end
        rollout = CacheAttnRolloutPlugin_Enabled._ROLLOUT
        if rollout is None or rollout.shape[-1] != T:
            # a forward outside the canvas loop (defensive); restart from identity
            rollout = eye.expand(B, -1, -1).clone()
        # end

        effective = eye.repeat(B, 1, 1)
        effective[:, idx_current, :] = scores    # only queried rows carry real attention

        mask_rows = CacheAttnRolloutPlugin_Enabled.ROW_MASK
        if mask_rows is not None and mask_rows.shape == (B, T):
            # keep identity at rows THIS sample did not select (union batching)
            effective = torch.where(mask_rows.to(device).unsqueeze(-1), effective,
                                    eye.expand(B, -1, -1))
        # end

        residual = effective + eye
        residual = residual / residual.sum(dim=-1, keepdim=True)
        CacheAttnRolloutPlugin_Enabled._ROLLOUT = residual @ rollout
    # end

    def get_global_importance(self):
        rollout = CacheAttnRolloutPlugin_Enabled._ROLLOUT
        return None if rollout is None else rollout.sum(dim=1)    # (B, T): end-to-end influence of column j
    # end

    def clear(self, model):
        CacheAttnRolloutPlugin_Enabled._ROLLOUT = None
        CacheAttnRolloutPlugin_Enabled.ROW_MASK = None
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
    ID_BLOCK_FORCED = None    # full-canvas DENSE forwards (oracle collectors) query
                              # the whole canvas, so the current decoding block cannot
                              # be inferred from idx_current[-1]; set it explicitly
                              # per block and reset to None afterwards
    SKIP_SAVE = False    # KV-only maintenance forwards over rows OUTSIDE the
                         # current block (e.g. the instruct runner's suffix
                         # refresh) would make reset_and_refresh_3d infer the
                         # wrong block from idx_current[-1] and wipe the
                         # current block's score table -- set this True around
                         # such forwards (prompt-row forwards need no guard:
                         # they early-return via the len_base check)

    @classmethod
    def set_len_prompt(cls, len_prompt):
        cls.LEN_PROMPT = len_prompt
        return cls
    # end

    @classmethod
    def set_id_block_forced(cls, id_block):
        cls.ID_BLOCK_FORCED = id_block
        return cls
    # end

    @classmethod
    def set_size_block(cls, size_block):
        cls.SIZE_BLOCK = size_block
        return cls
    # end

    @classmethod
    def set_skip_save(cls, skip):
        CacheAttnPlugin_Enabled.SKIP_SAVE = bool(skip)    # base-class slot: covers subclasses
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
        if self.__class__.ID_BLOCK_FORCED is not None:
            id_block_current = self.__class__.ID_BLOCK_FORCED
        else:
            id_block_current = self.get_block_id(idx_current[-1], len_block, len_base)
        # end

        idx_block_current_min = self.get_block_idx_min(id_block_current, len_block, len_base)

        # both bounds: dense full-canvas queries carry rows BEYOND the current
        # block too (future blocks); growing/sparse windows never do, so the
        # upper bound is a no-op there
        mask_row_current = (idx_current >= idx_block_current_min)\
                         & (idx_current < idx_block_current_min + len_block)
        idx_current = idx_current[mask_row_current]   # select by mask
        assert idx_current.shape[-1] > 0

        # idx_current 处理完成
        matrix_current = matrix_current[:, mask_row_current, :]   # keep rows aligned with filtered idx_current
        # keep the key columns of the CURRENT block. Columns are global positions
        # (keys are the merged full-window cache), so slice by block bounds:
        # under growing windows this equals the old [-len_block:] (the window ends
        # at the block), but under full-canvas windows (run_llada_instruct_mlp)
        # the current block sits mid-window and [-len_block:] would grab the
        # canvas's LAST block instead.
        matrix_current = matrix_current[:, :, idx_block_current_min:idx_block_current_min + len_block]

        B = matrix_current.shape[0]    # batch-general: shared idx, per-sample scores
        if id_block_current != id_block_origin:
            matrix_origin = torch.zeros((B, len_block, len_block), dtype=matrix_current.dtype, device=device)   # -1
        # end

        idx_current_relevant = idx_current - idx_block_current_min

        idx_current_relevant_3d = idx_current_relevant.view(1, -1, 1).expand(B, -1, len_block)

        matrix_origin = matrix_origin.scatter(1, idx_current_relevant_3d, matrix_current)
        return matrix_origin
    # end

    def save(self):

        if CacheAttnPlugin_Enabled.SKIP_SAVE:
            return
        # end

        layer_id = self.load_attrs('layer_id')[0]   # TODO: remove after bug fixed
        len_block = self.__class__.SIZE_BLOCK
        len_base = self.__class__.LEN_PROMPT

        q_current_rotated, k_final_rotated = self.load_vars('q_current_rotated', 'k_final_rotated')
        idx_current = self.load_vars('idx_current')[0]
        # pad mask; absent from model frames without batch support (dream)
        attention_bias = self.load_var_optional('attention_bias')

        get_attn_score_avg = self.load_func('get_attn_score_avg')

        if attention_bias is not None:
            scores_attn_current = get_attn_score_avg(q_current_rotated, k_final_rotated, attention_bias)
        else:
            scores_attn_current = get_attn_score_avg(q_current_rotated, k_final_rotated)    # dream: 2-arg signature
        # end
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

    def collect_attn_from_all_blocks_batched(self, model): # -> (num_layers, B, Q, K)
        # batch-preserving variant for the *_batch runners; the original method
        # keeps its batch-1 squeeze so existing runners stay untouched
        return torch.stack(
            [block_transformer.scores_attn_origin
             for block_transformer in model.model.transformer.blocks[:]], dim=0)
    # end

    def clear(self, model):
        CacheAttnPlugin_Enabled.SKIP_SAVE = False
        for block_transformer in model.model.transformer.blocks[:]:
            if hasattr(block_transformer, 'scores_attn_origin'):
                del block_transformer.scores_attn_origin
            # end
            if hasattr(block_transformer, 'idx_origin'):
                del block_transformer.idx_origin
            # end
        # end
    # end
# end



class CacheAttnRouterRolloutPlugin_Enabled(CacheAttnPlugin_Enabled):
    '''both consumers of the attention hook in ONE plugin slot: the block-local
    attention rows the router features need (parent behavior) AND the d2Cache
    attention rollout (for rollout-based refresh of unmasked/prompt tokens).
    The per-layer scores are computed twice (once per consumer) -- negligible
    for sparse query widths. Used by the router+rollout hybrid runner.'''

    def save(self):
        super().save()    # block-local rows -> router features
        CacheAttnRolloutPlugin_Enabled.save(self)    # rollout accumulation
    # end

    def get_global_importance(self):
        return CacheAttnRolloutPlugin_Enabled.get_global_importance(self)
    # end

    def clear(self, model):
        super().clear(model)
        CacheAttnRolloutPlugin_Enabled._ROLLOUT = None
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

    ROW_MASK = None    # optional (B, T) bool: PER-SAMPLE selected rows. Under
                       # union batching a forward carries rows some samples did
                       # not select; without this mask their K/V would be
                       # refreshed for every sample, making the batched cache
                       # fresher than the bs-1 method. True = this sample takes
                       # the fresh row; False = it keeps its cached row. The
                       # *_batch runners set it per union forward (None for
                       # shared-row forwards); bs-1 runners never touch it.

    @classmethod
    def set_row_mask(cls, mask_rows):
        cls.ROW_MASK = mask_rows
        return cls
    # end

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

        mask_rows = CachePastKVPlugin_Enabled.ROW_MASK
        keep_k = keep_v = idx_old = None
        if mask_rows is not None:
            # snapshot the pre-merge rows (concat_and_replace writes in place);
            # rows beyond the old cache length are NEW and stay fresh for all
            len_old = k_previous.shape[-2]
            idx_old = idx_current[idx_current < len_old]
            keep_k = k_previous[:, :, idx_old, :].clone()
            keep_v = v_previous[:, :, idx_old, :].clone()
        # end

        k_final = concat_and_replace(k_previous, k_current, idx_current, shape_target)
        v_final = concat_and_replace(v_previous, v_current, idx_current, shape_target)

        if mask_rows is not None and idx_old.numel() > 0:
            fresh = mask_rows[:, idx_old][:, None, :, None]    # (B, 1, n, 1)
            k_final[:, :, idx_old, :] = torch.where(fresh, k_final[:, :, idx_old, :], keep_k)
            v_final[:, :, idx_old, :] = torch.where(fresh, v_final[:, :, idx_old, :], keep_v)
        # end

        return k_final, v_final
    # end

    def save(self):
        k_final, v_final = self.load_vars('k_final', 'v_final')
        layer_past = (k_final, v_final)
        self.save_attrs(layer_past=layer_past)
    # end

    def clear(self, model):
        CachePastKVPlugin_Enabled.ROW_MASK = None
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

