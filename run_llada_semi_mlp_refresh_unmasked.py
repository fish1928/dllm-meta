#################################################
# v2 (filename historical). ABLATION arm: NO refresh of any kind + d2Cache's
# masked-token selection with a width hyperparameter.
#
#   selection: top-WIDTH still-masked positions by conf x certainty_density
#              (their score; width = d2c_k, default 32; sigma = d2c_sigma)
#   refresh:   none -- no prompt re-forward, no periodic block re-sync; only
#              the just-unmasked tokens are re-queried once (mask-KV -> token-KV)
#   conf:      live lazy table (refreshed at queried candidates each step)
#
# This configuration is exactly run_llada_d2cache with the attention rollout
# disabled, so it is implemented as a thin subclass pinning those defaults
# (rollout off, conf live) instead of duplicating the loop -- a runner name
# for the ablation that cannot drift from the validated implementation.
# d2c_k / d2c_sigma remain overridable via model_args (the "width" knob:
# d2c_k=8/16/32...).
#
# v1 of this file (periodic refresh of unmasked gen tokens, router selection)
# scored ~0.25 on gsm8k smoke -- superseded by this arm.
#################################################

from run_llada_d2cache import RunModel as RunModelD2C


class RunModel(RunModelD2C):

    def config_plugin_(self, config):
        # pin the ablation's defaults BEFORE plugin selection (config_plugin_
        # is where the rollout plugin would otherwise be enabled); explicit
        # model_args still win
        if config.d2c_rollout_p is None:
            config.d2c_rollout_p = 0.0    # no rollout: extras = just-unmasked only
        # end
        if config.d2c_conf_mode is None:
            config.d2c_conf_mode = 'live'
        # end
        return super().config_plugin_(config)
    # end
# end
