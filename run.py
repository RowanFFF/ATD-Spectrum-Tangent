import argparse
import os
import torch
import torch.backends
from utils.print_args import print_args
import random
import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='TimesNet')

    # basic config
    parser.add_argument('--task_name', type=str, required=True, default='long_term_forecast',
                        help='task name, options:[long_term_forecast, short_term_forecast, imputation, classification, anomaly_detection]')
    parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
    parser.add_argument('--model_id', type=str, required=True, default='test', help='model id')
    parser.add_argument('--model', type=str, required=True, default='Autoformer',
                        help='model name, options: [Autoformer, Transformer, TimesNet]')

    # data loader
    parser.add_argument('--data', type=str, required=True, default='ETTh1', help='dataset type')
    parser.add_argument(
        '--dataset_tag',
        type=str,
        default=None,
        help='optional unambiguous dataset label used in experiment settings',
    )
    parser.add_argument('--root_path', type=str, default='./data/ETT/', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')
    parser.add_argument('--checkpoint_path', type=str, default='',
                        help='explicit checkpoint file to load for testing or zero-shot transfer')
    parser.add_argument('--data_scale', dest='data_scale', action='store_true',
                        help='apply the training-set StandardScaler to model inputs')
    parser.add_argument('--no_data_scale', dest='data_scale', action='store_false',
                        help='keep model inputs in their original scale')
    parser.set_defaults(data_scale=None)

    # forecasting task
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=48, help='start token length')
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
    parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='subset for M4')
    parser.add_argument('--inverse', action='store_true', help='inverse output data', default=False)

    # inputation task
    parser.add_argument('--mask_rate', type=float, default=0.25, help='mask ratio')

    # anomaly detection task
    parser.add_argument('--anomaly_ratio', type=float, default=0.25, help='prior anomaly ratio (%%)')

    # model define
    parser.add_argument('--expand', type=int, default=2, help='expansion factor for Mamba')
    parser.add_argument('--d_conv', type=int, default=4, help='conv kernel size for Mamba')
    parser.add_argument('--tv_dt', type=int, default=0, help='whether to use time variant dt for MambaSL')
    parser.add_argument('--tv_B', type=int, default=0, help='whether to use time variant B for MambaSL')
    parser.add_argument('--tv_C', type=int, default=0, help='whether to use time variant C for MambaSL')
    parser.add_argument('--use_D', type=int, default=0, help='whether to use D for MambaSL')
    parser.add_argument('--top_k', type=int, default=5, help='for TimesBlock')
    parser.add_argument('--num_kernels', type=int, default=6, help='for Inception')
    parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=7, help='output size')
    parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
    parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
    parser.add_argument('--factor', type=int, default=1, help='attn factor')
    parser.add_argument('--distil', action='store_false',
                        help='whether to use distilling in encoder, using this argument means not using distilling',
                        default=True)
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--channel_independence', type=int, default=1,
                        help='0: channel dependence 1: channel independence for FreTS model')
    parser.add_argument('--decomp_method', type=str, default='moving_avg',
                        help='method of series decompsition, only support moving_avg or dft_decomp')
    parser.add_argument('--use_norm', type=int, default=1, help='whether to use normalize; True 1 False 0')
    parser.add_argument('--down_sampling_layers', type=int, default=0, help='num of down sampling layers')
    parser.add_argument('--down_sampling_window', type=int, default=1, help='down sampling window size')
    parser.add_argument('--down_sampling_method', type=str, default=None,
                        help='down sampling method, only support avg, max, conv')
    parser.add_argument('--seg_len', type=int, default=96,
                        help='the length of segmen-wise iteration of SegRNN')

    # optimization
    parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
    parser.add_argument('--loader_seed', type=int, default=-1,
                        help='independent DataLoader RNG seed; negative uses the global RNG')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
    parser.add_argument(
        '--eval_batch_size',
        type=int,
        default=0,
        help='validation/test batch size; non-positive uses batch_size')
    parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
    parser.add_argument('--checkpoint_selection', type=str, default='vali',
                        choices=['vali', 'test_oracle'],
                        help='checkpoint metric; test_oracle is diagnostic test leakage')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    parser.add_argument(
        '--sam_rho',
        type=float,
        default=0.0,
        help=(
            'SAM neighborhood radius; zero keeps ordinary Adam and a '
            'positive value adds a second sharpness-aware training pass'),
    )
    parser.add_argument('--des', type=str, default='test', help='exp description')
    parser.add_argument('--loss', type=str, default='MSE', help='loss function')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

    # GPU
    parser.add_argument('--use_gpu', action='store_true', default=True, help='use gpu (default: on)')
    parser.add_argument('--no_use_gpu', action='store_false', dest='use_gpu', help='disable gpu (force cpu)')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--gpu_type', type=str, default='cuda', help='gpu type')  # cuda or mps
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')

    # de-stationary projector params
    parser.add_argument('--p_hidden_dims', type=int, nargs='+', default=[128, 128],
                        help='hidden layer dimensions of projector (List)')
    parser.add_argument('--p_hidden_layers', type=int, default=2, help='number of hidden layers in projector')

    # metrics (dtw)
    parser.add_argument('--use_dtw', action='store_true', default=False,
                        help='enable dtw metric (time consuming; default: off)')
    parser.add_argument('--stream_metrics', action='store_true', default=False,
                        help='accumulate metrics without saving pred/true arrays')
    parser.add_argument('--report_horizons', type=int, nargs='*', default=[],
                        help='additional prefix horizons reported during streaming evaluation')
    parser.add_argument('--skip_epoch_test', action='store_true', default=False,
                        help='skip diagnostic test scoring during training epochs')
    parser.add_argument(
        '--skip_final_test',
        action='store_true',
        default=False,
        help=(
            'train and select a checkpoint without evaluating the final test '
            'split; useful for validation-only model or scale selection'))
    parser.add_argument(
        '--include_initial_checkpoint',
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            'include the untrained model as epoch-zero candidate in '
            'validation-based checkpoint selection'
        ),
    )
    parser.add_argument(
        '--direct_patch_validation',
        action='store_true',
        default=False,
        help=(
            'select direct-patch checkpoints with their short training '
            'objective instead of a full autoregressive rollout'))
    parser.add_argument(
        '--direct_patch_train_channels',
        type=int,
        default=0,
        help=(
            'randomly sample at most this many channels per training batch '
            'for channel-independent direct-patch models'))
    parser.add_argument(
        '--direct_patch_validation_channels',
        type=int,
        default=-1,
        help=(
            'channel limit for direct-patch validation; -1 reuses the '
            'training limit and 0 evaluates every channel'))

    # Augmentation
    parser.add_argument('--augmentation_ratio', type=int, default=0, help="How many times to augment")
    parser.add_argument('--seed', type=int, default=2021, help="Randomization seed")
    parser.add_argument('--jitter', default=False, action="store_true", help="Jitter preset augmentation")
    parser.add_argument('--scaling', default=False, action="store_true", help="Scaling preset augmentation")
    parser.add_argument('--permutation', default=False, action="store_true",
                        help="Equal Length Permutation preset augmentation")
    parser.add_argument('--randompermutation', default=False, action="store_true",
                        help="Random Length Permutation preset augmentation")
    parser.add_argument('--magwarp', default=False, action="store_true", help="Magnitude warp preset augmentation")
    parser.add_argument('--timewarp', default=False, action="store_true", help="Time warp preset augmentation")
    parser.add_argument('--windowslice', default=False, action="store_true", help="Window slice preset augmentation")
    parser.add_argument('--windowwarp', default=False, action="store_true", help="Window warp preset augmentation")
    parser.add_argument('--rotation', default=False, action="store_true", help="Rotation preset augmentation")
    parser.add_argument('--spawner', default=False, action="store_true", help="SPAWNER preset augmentation")
    parser.add_argument('--dtwwarp', default=False, action="store_true", help="DTW warp preset augmentation")
    parser.add_argument('--shapedtwwarp', default=False, action="store_true", help="Shape DTW warp preset augmentation")
    parser.add_argument('--wdba', default=False, action="store_true", help="Weighted DBA preset augmentation")
    parser.add_argument('--discdtw', default=False, action="store_true",
                        help="Discrimitive DTW warp preset augmentation")
    parser.add_argument('--discsdtw', default=False, action="store_true",
                        help="Discrimitive shapeDTW warp preset augmentation")
    parser.add_argument('--extra_tag', type=str, default="", help="Anything extra")

    # TimeXer
    parser.add_argument('--patch_len', type=int, default=16, help='patch length')

    # DenseAR / TailAR
    parser.add_argument('--ar_patch_len', type=int, default=24,
                        help='non-overlapping patch length for DenseAR and TailAR')
    parser.add_argument(
        '--granularity_decoder_len',
        type=int,
        default=0,
        help=(
            'causal points decoded and committed from each full parent '
            'state by GranularityDecoupledAR; zero uses ar_patch_len'),
    )
    parser.add_argument(
        '--granularity_multi_commit_pretrained',
        type=str,
        default='',
        help='frozen P96-state/fine-commit parent checkpoint',
    )
    parser.add_argument(
        '--granularity_multi_commit_hidden',
        type=int,
        default=1024,
        help='hidden width of the protected fine-commit macro head',
    )
    parser.add_argument(
        '--granularity_multi_commit_patches',
        type=int,
        default=4,
        help='fine patches emitted by one protected macro call',
    )
    parser.add_argument(
        '--granularity_multi_commit_eval_patches',
        type=int,
        default=0,
        help='fine patches committed at evaluation; zero uses trained K',
    )
    parser.add_argument(
        '--granularity_multi_commit_target',
        type=str,
        default='composition',
        choices=['composition', 'truth'],
        help='recursive-parent DPOD target or direct real-future target',
    )
    parser.add_argument(
        '--granularity_multi_commit_readout',
        type=str,
        default='last',
        choices=['last', 'innovation'],
        help='frozen parent state exposed to the macro head',
    )
    parser.add_argument(
        '--q1_moe_pretrained',
        type=str,
        default='',
        help='strong frozen Q1 checkpoint used by residual MoE placement',
    )
    parser.add_argument(
        '--q1_moe_placement',
        type=str,
        default='ffn_last',
        choices=['stem', 'ffn_first', 'ffn_last', 'ffn_all', 'output'],
        help='interface receiving protected residual experts',
    )
    parser.add_argument(
        '--q1_moe_experts',
        type=int,
        default=4,
        help='number of residual Q1 experts',
    )
    parser.add_argument(
        '--q1_moe_top_k',
        type=int,
        default=2,
        help='experts selected per Q1 token',
    )
    parser.add_argument(
        '--q1_moe_hidden',
        type=int,
        default=128,
        help='hidden width of every residual Q1 expert',
    )
    parser.add_argument(
        '--q1_moe_expert_inputs',
        type=str,
        default='state',
        choices=['state', 'temporal_roles'],
        help='homogeneous state experts or heterogeneous temporal roles',
    )
    parser.add_argument(
        '--q1_moe_balance_weight',
        type=float,
        default=0.01,
        help='assignment-aware router balancing weight',
    )
    parser.add_argument('--dense_ar_roll_patches', type=int, default=1,
                        help='number of patches predicted per DenseAR rollout')
    parser.add_argument(
        '--dense_ar_eval_commit_patches',
        type=int,
        default=0,
        help=(
            'patches committed per evaluation step; zero commits every '
            'predicted query patch'),
    )
    parser.add_argument('--dense_ar_loss_space', type=str,
                        default='normalized',
                        choices=['normalized', 'point'],
                        help='compute DenseAR next-patch loss before or after instance denormalization')
    parser.add_argument('--dense_ar_loss_type', type=str,
                        default='mse', choices=['mse', 'huber'],
                        help='regression objective used by DenseAR training')
    parser.add_argument('--band_norm_json', type=str, default='',
                        help='path to train-set per-band DCT signal energies; '
                             'when set, the training objective is the '
                             'band-normalized MSE (mechanism 3) instead of '
                             'plain point-space MSE')
    parser.add_argument('--point_space_train', action='store_true',
                        help='supervise the AR forecast output in point space '
                             '(model forward == forecast) instead of the '
                             'patch-space DenseAR objective; matched control '
                             'for --band_norm_json')
    parser.add_argument('--lora_rank', type=int, default=0,
                        help='low-rank adapter rank on the block projections '
                             '(attention qkv/out, FFN in/out); 0 disables '
                             'LoRA entirely (r=0 identity)')
    parser.add_argument('--lora_freeze_backbone', action='store_true',
                        help='freeze every parameter except the lora bypasses, '
                             'so training moves only the adapter')
    parser.add_argument('--lora_init_from', type=str, default='',
                        help='checkpoint whose backbone weights initialize '
                             'the frozen shared model; the fresh lora '
                             'bypasses stay zero (missing keys accepted)')
    parser.add_argument('--expand_init_from', type=str, default='',
                        help='narrower trained checkpoint (same family) '
                             'zero-padded into this wider model: the wide '
                             'model starts from the narrow model\'s '
                             'behavior, so extra capacity is inert until '
                             'the data activates it')
    parser.add_argument('--dense_ar_huber_delta', type=float, default=1.0,
                        help='Huber transition point when dense_ar_loss_type=huber')
    parser.add_argument(
        '--dense_ar_far_patch_weight',
        type=float,
        default=1.0,
        help=(
            'relative loss weight of future patches 2..K in DenseAR; '
            'future patch 1 remains in the dense anchor objective'),
    )
    parser.add_argument(
        '--template_closure_mode',
        type=str,
        default='off',
        choices=['off', 'learn', 'struct'],
        help=(
            'two-predictor shrinkage built into training: off = plain '
            'parent; learn = learnable band multipliers alpha(k) (init 0); '
            'struct = band multipliers frozen at the theoretical shape'),
    )
    parser.add_argument(
        '--template_closure_scales',
        type=str,
        default='48,96',
        help='comma-separated template scales (phase lengths)',
    )
    parser.add_argument(
        '--template_closure_betas',
        type=str,
        default='0.03125,0.03125',
        help='comma-separated pull betas per template scale',
    )
    parser.add_argument(
        '--template_closure_bands',
        type=str,
        default='0.25,0.5,1.0,2.0,3.0',
        help='comma-separated band multipliers over {h96,h192,h336,h672,post} '
             '(frozen in struct mode; initial value in learn mode)',
    )
    parser.add_argument(
        '--dense_ar_onpolicy_train_patches',
        type=int,
        default=1,
        help=(
            'generated-history Q1 steps used for on-policy transition '
            'training; one disables the auxiliary rollout'),
    )
    parser.add_argument(
        '--dense_ar_onpolicy_loss_weight',
        type=float,
        default=0.0,
        help='weight of generated-history Q1 transition losses',
    )
    parser.add_argument(
        '--dense_ar_structural_loss_weight',
        type=float,
        default=0.0,
        help=(
            'weight of gradient-balanced patch correlation, '
            'distribution, and mean training losses'),
    )
    parser.add_argument(
        '--dense_ar_onpolicy_detach_history',
        dest='dense_ar_onpolicy_detach_history',
        action='store_true',
        help='stop gradients through generated history during on-policy loss',
    )
    parser.add_argument(
        '--no_dense_ar_onpolicy_detach_history',
        dest='dense_ar_onpolicy_detach_history',
        action='store_false',
        help='backpropagate through generated on-policy history',
    )
    parser.set_defaults(
        dense_ar_onpolicy_detach_history=True)
    parser.add_argument(
        '--dense_ar_occupancy_ages',
        type=str,
        default='',
        help=(
            'comma-separated generated-history ages sampled during Q1 '
            'pushforward training; empty disables the objective'),
    )
    parser.add_argument(
        '--dense_ar_occupancy_unroll_patches',
        type=int,
        default=2,
        help='differentiable Q1 transitions after occupancy roll-in',
    )
    parser.add_argument(
        '--dense_ar_occupancy_loss_weight',
        type=float,
        default=0.0,
        help='weight of the age-stratified generated-occupancy loss',
    )
    parser.add_argument(
        '--dense_ar_occupancy_detach_local_history',
        action='store_true',
        help=(
            'detach locally generated patches; matched semi-gradient '
            'control for occupancy training'),
    )
    parser.add_argument(
        '--trajectory_native_mode',
        choices=[
            'point', 'multiscale', 'trend', 'slow_plan', 'pareto_anchor',
            'pareto_anchor_balanced',
        ],
        default='point',
        help=(
            'train a checkpoint-initialized Q1 on its complete generated '
            'trajectory, optionally with coarse losses, slow-loss-only trend '
            'training, immutable-memory slow-plan tokens, or parent-anchored '
            'early/far PCGrad, with optional far-gradient norm balancing'),
    )
    parser.add_argument(
        '--trajectory_native_memory_mode',
        choices=['history', 'terminal', 'shuffled', 'none'],
        default='history',
        help='aligned immutable memory or a matched slow-plan control',
    )
    parser.add_argument(
        '--trajectory_native_multiscale_weight', type=float, default=0.0,
        help='weight of blockwise-constant trajectory error projections',
    )
    parser.add_argument(
        '--trajectory_native_drift_weight', type=float, default=0.0,
        help='weight of cumulative prefix-bias errors',
    )
    parser.add_argument(
        '--trajectory_native_plan_aux_weight', type=float, default=0.0,
        help='weight of origin-normalized slow-plan level supervision',
    )
    parser.add_argument(
        '--trajectory_native_plan_gate_limit', type=float, default=0.25,
        help='maximum hidden-state displacement scale from the slow plan',
    )
    parser.add_argument(
        '--trajectory_native_plan_heads', type=int, default=4,
        help='cross-attention heads used by immutable slow-plan queries',
    )
    parser.add_argument(
        '--trajectory_native_scales', type=str, default='1,2,4',
        help='comma-separated parent-commit multipliers for coarse losses',
    )
    parser.add_argument(
        '--trajectory_native_drift_horizons',
        type=str, default='96,192,336,720',
        help='comma-separated cumulative horizons used by the drift loss',
    )
    parser.add_argument(
        '--trajectory_native_detach_history',
        action=argparse.BooleanOptionalAction,
        default=False,
        help='detach generated writeback for a semi-gradient control',
    )
    parser.add_argument(
        '--trajectory_native_deterministic_rollout',
        action=argparse.BooleanOptionalAction,
        default=True,
        help='disable candidate dropout during deployment-matched rollout',
    )
    parser.add_argument(
        '--trajectory_native_anchor_early_horizon', type=int, default=96,
        help='protected early behavioral horizon for pareto-anchor training',
    )
    parser.add_argument(
        '--trajectory_native_anchor_far_start', type=int, default=336,
        help='first point in the far rollout objective',
    )
    parser.add_argument(
        '--trajectory_native_anchor_weight', type=float, default=1.0,
        help='weight of smooth positive early excess risk versus the parent',
    )
    parser.add_argument(
        '--trajectory_native_anchor_temperature', type=float, default=0.01,
        help='soft positive-part temperature for early excess risk',
    )
    parser.add_argument(
        '--trajectory_native_anchor_margin', type=float, default=0.0,
        help='desired parent-relative early-risk improvement margin',
    )
    parser.add_argument(
        '--trajectory_native_anchor_mid_weight', type=float, default=0.5,
        help='middle-band weight inside the far behavioral objective',
    )
    parser.add_argument(
        '--trajectory_native_anchor_trust_weight', type=float, default=0.0,
        help='relative native-parameter drift penalty in the anchor objective',
    )
    parser.add_argument(
        '--native_successor_placement',
        choices=['causal_all', 'rolling_terminal', 'immutable_terminal'],
        default='immutable_terminal',
        help=(
            'inject shifted-successor reads at every causal token, at the '
            'rolling terminal, or from immutable observed transition memory'),
    )
    parser.add_argument(
        '--native_successor_mode',
        choices=[
            'anchor_centered', 'anchor_only', 'centered_only',
            'full_attention', 'same_patch', 'shuffled_successor',
            'no_memory_mlp', 'none',
        ],
        default='anchor_centered',
        help='uniform-anchored successor branch or a matched information control',
    )
    parser.add_argument(
        '--native_successor_heads', type=int, default=4,
        help='heads used to address predecessor-key/successor-value edges',
    )
    parser.add_argument(
        '--native_successor_conditional_mix', type=float, default=0.25,
        help='learned-attention deviation mixed around the uniform value anchor',
    )
    parser.add_argument(
        '--native_successor_max_norm_ratio', type=float, default=0.5,
        help='maximum branch RMS relative to its receiving parent token',
    )
    parser.add_argument(
        '--native_successor_gate_limit', type=float, default=0.25,
        help='maximum signed occupancy-gated native branch strength',
    )
    parser.add_argument(
        '--native_successor_gate_init', type=float, default=0.0,
        help='initial signed branch strength; zero is exact Q1',
    )
    parser.add_argument(
        '--native_successor_occupancy_power', type=float, default=1.0,
        help='power applied to generated-history occupancy before branch gating',
    )
    parser.add_argument(
        '--native_successor_train_steps', type=int, default=4,
        help='short differentiable Q1 commits used for native pushforward training',
    )
    parser.add_argument(
        '--native_successor_aux_weight', type=float, default=0.25,
        help='weight of the observed-transition structural auxiliary',
    )
    parser.add_argument(
        '--native_successor_pushforward_weight', type=float, default=1.0,
        help='weight of native short closed-loop rollout risk',
    )
    parser.add_argument(
        '--native_successor_teacher_weight', type=float, default=0.0,
        help='weight of ordinary teacher-forced Q1 next-patch training',
    )
    parser.add_argument(
        '--native_successor_aux_strength', type=float, default=0.125,
        help='fixed branch strength used only by the structural auxiliary',
    )
    parser.add_argument(
        '--native_successor_trend_weight', type=float, default=0.0,
        help='weight of multi-resolution trajectory-mean errors',
    )
    parser.add_argument(
        '--native_successor_drift_weight', type=float, default=0.0,
        help='weight of cumulative prefix-bias errors',
    )
    parser.add_argument(
        '--native_successor_trust_weight', type=float, default=0.0,
        help='relative candidate-Q1 parameter-drift penalty',
    )
    parser.add_argument(
        '--native_successor_update_weight', type=float, default=0.0,
        help='quadratic penalty on the realized latent branch displacement',
    )
    parser.add_argument(
        '--native_successor_candidate_lr_scale', type=float, default=1.0,
        help='Q1 learning-rate multiplier; zero freezes the checkpoint parent path',
    )
    parser.add_argument(
        '--native_successor_branch_lr_scale', type=float, default=1.0,
        help='learning-rate multiplier for the native transition branch',
    )
    parser.add_argument(
        '--native_successor_detach_history',
        action=argparse.BooleanOptionalAction,
        default=False,
        help='detach generated writeback for a matched semi-gradient control',
    )
    parser.add_argument(
        '--native_successor_deterministic_rollout',
        action=argparse.BooleanOptionalAction,
        default=True,
        help='disable candidate dropout during short training rollouts',
    )
    parser.add_argument(
        '--native_provenance_mode',
        choices=[
            'split', 'observed_only', 'generated_only', 'merged', 'swapped',
            'rolling_split', 'no_source_mlp', 'none',
        ],
        default='split',
        help=(
            'fixed observed/generated head routing or a matched source-mask '
            'control in the strong-Q1 provenance branch'),
    )
    parser.add_argument(
        '--native_provenance_heads', type=int, default=4,
        help='attention heads divided between observed and generated sources',
    )
    parser.add_argument(
        '--native_provenance_max_norm_ratio', type=float, default=0.5,
        help='maximum provenance-branch RMS relative to the terminal token',
    )
    parser.add_argument(
        '--native_koopman_mode',
        choices=[
            'rotation_slow', 'decay_slow', 'identity', 'fast_decay',
            'shuffled_modes', 'no_spectrum_mlp', 'none',
        ],
        default='rotation_slow',
        help='stable slow-spectrum branch or a matched transition control',
    )
    parser.add_argument(
        '--native_koopman_rank', type=int, default=8,
        help='even dimension of the orthogonal Koopman observable subspace',
    )
    parser.add_argument(
        '--native_koopman_rho_min', type=float, default=0.9,
        help='minimum modulus of learned slow eigenvalues',
    )
    parser.add_argument(
        '--native_koopman_rho_max', type=float, default=0.999,
        help='strict upper bound on learned slow eigenvalue modulus',
    )
    parser.add_argument(
        '--native_koopman_rho_init', type=float, default=0.97,
        help='initial slow eigenvalue modulus',
    )
    parser.add_argument(
        '--native_koopman_theta_max', type=float, default=1.5707963267948966,
        help='maximum absolute phase rotation per parent-patch transition',
    )
    parser.add_argument(
        '--native_koopman_fast_rho_max', type=float, default=0.5,
        help='spectral-radius ceiling for the matched fast-decay control',
    )
    parser.add_argument(
        '--native_koopman_max_norm_ratio', type=float, default=0.5,
        help='maximum Koopman displacement RMS relative to terminal token',
    )
    parser.add_argument(
        '--native_koopman_consistency_weight', type=float, default=0.25,
        help='weight of observed one-step eigenfunction consistency',
    )
    parser.add_argument(
        '--location_state_lags',
        type=str,
        default='1,2,4,7,14',
        help='comma-separated patch-level velocity lags for location closure',
    )
    parser.add_argument(
        '--location_state_hidden',
        type=int,
        default=0,
        help='location transition hidden width; zero uses a linear head',
    )
    parser.add_argument(
        '--location_state_max_scale',
        type=float,
        default=0.5,
        help='maximum next-patch mean residual in rolling-std units',
    )
    parser.add_argument(
        '--location_state_velocity_control',
        choices=['true', 'zero', 'negated'],
        default='true',
        help='causal velocity features or matched information controls',
    )
    parser.add_argument(
        '--location_state_decay',
        type=float,
        default=0.0,
        help='retention of the explicit gauge location state',
    )
    parser.add_argument(
        '--location_state_anchor_weight',
        type=float,
        default=1.0,
        help='weight of the clean next-patch location objective',
    )
    parser.add_argument(
        '--location_state_loss',
        choices=['patch', 'level'],
        default='patch',
        help='fit the complete emitted patch or only its scalar mean',
    )
    parser.add_argument('--ablation_attention', type=str, default='softmax',
                        choices=['softmax', 'shrinkage', 'lag_bias',
                                 'sigmoid_norm', 'relu_norm', 'raw', 'uniform',
                                 'static', 'identity', 'none'],
                        help='token mixer in TransformerAblationAR')
    parser.add_argument('--ablation_ffn', type=str, default='mlp',
                        choices=['mlp', 'swiglu', 'linear', 'none'],
                        help='feed-forward sublayer in TransformerAblationAR')
    parser.add_argument(
        '--ablation_shrinkage_type',
        type=str,
        default='fixed',
        choices=['fixed', 'hybrid', 'learned_head', 'learned_query'],
        help='how shrinkage attention allocates dynamic QK weight',
    )
    parser.add_argument(
        '--ablation_shrinkage_alpha',
        type=float,
        default=1.0,
        help='fixed or initial dynamic-QK weight in shrinkage attention',
    )
    parser.add_argument(
        '--ablation_dynamic_heads',
        type=int,
        default=None,
        help='number of content-dependent heads in hybrid attention',
    )
    parser.add_argument(
        '--ablation_qk_norm',
        type=str,
        default='dot',
        choices=['dot', 'cosine'],
        help='dot-product or cosine QK scores',
    )
    parser.add_argument(
        '--ablation_attention_temperature',
        type=float,
        default=1.0,
        help='multiplicative attention-score temperature',
    )
    parser.add_argument(
        '--ablation_attention_softcap',
        type=float,
        default=0.0,
        help='smooth absolute cap for QK logits; zero disables soft-capping',
    )
    parser.add_argument(
        '--ablation_learnable_temperature',
        action='store_true',
        help='learn the attention-score temperature',
    )
    parser.add_argument(
        '--ablation_norm_kind',
        type=str,
        default='layer',
        choices=['layer', 'rms'],
        help='LayerNorm or RMSNorm in TransformerAblationAR',
    )
    parser.add_argument(
        '--ablation_layer_scale_init',
        type=float,
        default=-1.0,
        help='negative disables LayerScale; otherwise its initial value',
    )
    parser.add_argument(
        '--ablation_local_kernel',
        type=int,
        default=0,
        help='zero or an odd causal local-residual kernel width >= 3',
    )
    parser.add_argument(
        '--ablation_local_no_bias',
        action='store_true',
        help='remove bias from the optional local token mixer',
    )
    parser.add_argument(
        '--ablation_local_mode',
        type=str,
        default='state',
        choices=['state', 'innovation', 'state_innovation',
                 'attention', 'state_attention'],
        help=(
            'filter hidden states, innovations, attention outputs, or a '
            'two-path combination locally'),
    )
    parser.add_argument(
        '--ablation_last_query_inference',
        action='store_true',
        help='use the exact one-layer final-query path during AR inference',
    )
    parser.add_argument(
        '--predictive_value_residual',
        action='store_true',
        help=(
            'add gated first-layer values to later attention layers in '
            'PredictiveStateResidualAR'),
    )
    parser.add_argument(
        '--predictive_value_gate_init',
        type=float,
        default=0.0,
        help='initial cross-layer value-residual strength',
    )
    parser.add_argument(
        '--predictive_value_gate_limit',
        type=float,
        default=1.0,
        help='absolute bound on cross-layer value-residual strengths',
    )
    parser.add_argument(
        '--predictive_layerwise_loss_weight',
        type=float,
        default=0.0,
        help=(
            'weight of shared-head intermediate-layer Q1 supervision in '
            'PredictiveStateResidualAR'),
    )
    parser.add_argument(
        '--predictive_auxiliary_layers',
        type=str,
        default='all',
        choices=['all', 'last'],
        help='supervise all intermediate layers or only the penultimate one',
    )
    parser.add_argument(
        '--predictive_detach_auxiliary_head',
        action='store_true',
        help=(
            'block auxiliary-loss gradients into the shared Q1 head while '
            'retaining its ordinary final-layer gradient'),
    )
    parser.add_argument(
        '--adaptive_patch_fusion_atom_len',
        type=int,
        default=12,
        help='fixed atomic patch length for adaptive encoder fusion',
    )
    parser.add_argument(
        '--adaptive_patch_fusion_max_atoms',
        type=int,
        default=8,
        help='largest power-of-two atomic fusion span',
    )
    parser.add_argument(
        '--adaptive_patch_fusion_atom_dim',
        type=int,
        default=0,
        help='shared P12 basis width; zero keeps the full atomic dimension',
    )
    parser.add_argument(
        '--adaptive_patch_fusion_router',
        type=str,
        default='content',
        choices=['static', 'content'],
        help='dataset-global or token-content adaptive fusion routing',
    )
    parser.add_argument(
        '--adaptive_patch_fusion_rank_router',
        type=str,
        default='none',
        choices=['none', 'static', 'content'],
        help='optional within-parent routing over nested atomic ranks',
    )
    parser.add_argument(
        '--adaptive_patch_fusion_output',
        type=str,
        default='protected',
        choices=['fixed', 'replace', 'protected'],
        help='fixed atomic control, routed replacement, or protected residual',
    )
    parser.add_argument(
        '--adaptive_patch_fusion_parent_logit',
        type=float,
        default=2.0,
        help='initial router preference for the current parent span',
    )
    parser.add_argument(
        '--adaptive_patch_fusion_rank_parent_logit',
        type=float,
        default=2.0,
        help='initial router preference for the full atomic rank',
    )
    parser.add_argument(
        '--adaptive_patch_fusion_gate_init',
        type=float,
        default=0.25,
        help='initial bounded protected fusion gate',
    )
    parser.add_argument(
        '--ablation_raw_patch_skip_kernel',
        type=int,
        default=0,
        help='causal raw-patch linear skip width; zero disables it',
    )
    parser.add_argument(
        '--ablation_raw_patch_skip_mode',
        type=str,
        default='full',
        choices=['full', 'phase', 'dyadic_phase'],
        help='full, phase-aligned, or sparse dyadic phase raw-patch skip',
    )
    parser.add_argument(
        '--ablation_recurrent_state_gate_init',
        type=float,
        default=-2.0,
        help='negative disables the parallel causal recurrent state path',
    )
    parser.add_argument(
        '--ablation_patch_shape_gate_init',
        type=float,
        default=-2.0,
        help=(
            'negative disables the standardized within-patch shape '
            'embedding; otherwise its bounded residual gate init'),
    )
    parser.add_argument(
        '--ablation_patch_moment_gate_init',
        type=float,
        default=-2.0,
        help=(
            'negative disables the local mean/log-RMS patch embedding; '
            'otherwise its bounded residual gate init'),
    )
    parser.add_argument(
        '--ablation_patch_shape_address_gate_init',
        type=float,
        default=-2.0,
        help=(
            'negative disables standardized shape-only Q/K addressing; '
            'otherwise its bounded residual gate init'),
    )
    parser.add_argument(
        '--ablation_patch_shape_value_gate_init',
        type=float,
        default=-2.0,
        help=(
            'negative disables standardized shape augmentation of V; '
            'otherwise its bounded residual gate init'),
    )
    parser.add_argument(
        '--ablation_patch_shape_sidecar_gate_init',
        type=float,
        default=-2.0,
        help=(
            'negative disables the separately projected standardized '
            'shape retrieval sidecar; otherwise its bounded gate init'),
    )
    parser.add_argument(
        '--ablation_patch_moment_sidecar_gate_init',
        type=float,
        default=-2.0,
        help=(
            'negative disables the separately projected local moment '
            'retrieval sidecar; otherwise its bounded gate init'),
    )
    parser.add_argument(
        '--ablation_causal_low_value_gate_init',
        type=float,
        default=-2.0,
        help=(
            'negative disables the causal four-token low-pass value '
            'sidecar; otherwise its bounded gate init'),
    )
    parser.add_argument(
        '--ablation_causal_detail_value_gate_init',
        type=float,
        default=-2.0,
        help=(
            'negative disables the causal four-token detail value '
            'sidecar; otherwise its bounded gate init'),
    )
    parser.add_argument(
        '--ablation_value_topology',
        type=str,
        default='state',
        choices=['state', 'past_state', 'past_anchor',
                 'successor', 'successor_delta'],
        help='which observed token is retrieved by each attention key',
    )
    parser.add_argument(
        '--ablation_patch_assembly',
        type=str,
        default='none',
        choices=[
            'none', 'chronological', 'reverse', 'mean_sorted', 'factorized',
            'fd'],
        help=(
            'composition of shared fine-patch representations into each '
            'Transformer token'),
    )
    parser.add_argument(
        '--ablation_patch_assembly_fine_len',
        type=int,
        default=12,
        help='fine observation length used by direct patch assembly',
    )
    parser.add_argument(
        '--ablation_patch_assembly_atom_dim',
        type=int,
        default=0,
        help=(
            'fixed output width of the shared fine-patch projection; zero '
            'keeps the legacy d_model / assembly_slots parameterization'),
    )
    parser.add_argument(
        '--ablation_patch_assembly_basis_dim',
        type=int,
        default=8,
        help=(
            'number of shared DCT-position maps used by factorized patch '
            'assembly; must cover the active fine-patch slots'),
    )
    parser.add_argument(
        '--ablation_patch_assembly_stem',
        type=str,
        default='linear',
        choices=['linear', 'diagonal', 'identity'],
        help=(
            'shared fine-patch stem before direct concatenation; diagonal '
            'and identity require atom_dim to equal the fine-patch length'),
    )
    parser.add_argument(
        '--ablation_patch_assembly_interface_dim',
        type=int,
        default=0,
        help=(
            'allow the direct-concat width (slots x fine dim) to differ '
            'from d_model by projecting to d_model through a fixed-width '
            'interface layer; zero keeps the direct packing test'),
    )
    parser.add_argument(
        '--ablation_patch_assembly_fd_widths',
        type=str,
        default='',
        help=(
            'per-band widths (level,macro,patch_scale,fine) of the '
            'frequency-band assembly; must sum to d_model'),
    )
    parser.add_argument(
        '--ablation_output_head',
        type=str,
        default='full',
        choices=[
            'full', 'split_independent', 'split_shared', 'split_haar',
            'factorized_atoms'],
        help=(
            'decode a full parent patch monolithically or from ordered '
            'd_model subspaces while retaining one canonical AR commit'),
    )
    parser.add_argument(
        '--ablation_output_fine_patch_len',
        type=int,
        default=12,
        help='fine output slot length used by structured readout heads',
    )
    parser.add_argument(
        '--ablation_output_basis_dim',
        type=int,
        default=8,
        help=(
            'number of shared DCT-position maps used by the factorized '
            'fine-patch output head'),
    )
    parser.add_argument(
        '--ablation_pretrained',
        type=str,
        default='',
        help=(
            'optional TransformerAblationAR checkpoint used to initialize '
            'all matching parent parameters'
        ),
    )
    parser.add_argument(
        '--ablation_trainable_scope',
        type=str,
        default='all',
        choices=['all', 'local_mixer', 'variable_mixer'],
        help=(
            'parameters optimized after loading an ablation checkpoint; '
            'restricted modes freeze the parent and fit only one residual'
        ),
    )
    parser.add_argument(
        '--cross_variable_memory_mode',
        type=str,
        default='innovation',
        choices=['level', 'innovation', 'shuffled_innovation'],
        help=(
            'source used by the protected low-rank cross-variable memory'
        ),
    )
    parser.add_argument(
        '--cross_variable_memory_rank',
        type=int,
        default=4,
        help='rank of the shared cross-variable factor state',
    )
    parser.add_argument(
        '--cross_variable_memory_heads',
        type=int,
        default=4,
        help='attention heads used to read the factor memory',
    )
    parser.add_argument(
        '--cross_variable_memory_bound',
        type=float,
        default=0.1,
        help='absolute normalized next-patch correction bound',
    )
    parser.add_argument(
        '--cross_variable_memory_penalty',
        type=float,
        default=0.0,
        help='optional squared correction penalty',
    )
    parser.add_argument(
        '--future_phase_query_mode',
        type=str,
        default='phase',
        choices=['phase', 'shuffled_phase', 'history_only'],
        help='future timestamp input used by the protected phase query',
    )
    parser.add_argument(
        '--future_phase_query_heads',
        type=int,
        default=4,
        help='attention heads used by the future-phase history read',
    )
    parser.add_argument(
        '--future_phase_query_bound',
        type=float,
        default=0.1,
        help='absolute normalized future-phase patch correction bound',
    )
    parser.add_argument(
        '--future_phase_query_penalty',
        type=float,
        default=0.0,
        help='optional squared future-phase correction penalty',
    )
    parser.add_argument(
        '--terminal_reread_memory_mode',
        type=str,
        default='history',
        choices=['history', 'terminal', 'shuffled', 'uniform'],
        help='memory intervention used by TerminalHistoryRereadAR',
    )
    parser.add_argument(
        '--terminal_reread_correction_space',
        type=str,
        default='output',
        choices=['output', 'state'],
        help='apply the bounded reread residual before or after the decoder',
    )
    parser.add_argument(
        '--terminal_reread_train_steps',
        type=int,
        default=1,
        help='number of generated-history commits supervised during training',
    )
    parser.add_argument(
        '--terminal_reread_rollin_steps',
        type=int,
        default=0,
        help='unsupervised generated commits before the short training loss',
    )
    parser.add_argument(
        '--terminal_reread_pushforward_mode',
        type=str,
        default='detached',
        choices=['detached', 'full'],
        help='whether later losses backpropagate through earlier Q1 commits',
    )
    parser.add_argument(
        '--terminal_reread_loss_decay',
        type=float,
        default=1.0,
        help='geometric weight applied to successive short-rollout losses',
    )
    parser.add_argument(
        '--terminal_reread_refinement_steps',
        type=int,
        default=1,
        help='weight-tied latent rereads applied inside one Q1 commit',
    )
    parser.add_argument(
        '--terminal_reread_step_scale',
        type=float,
        default=1.0,
        help='bounded latent displacement scale per recurrent reread',
    )
    parser.add_argument(
        '--working_state_fine_patch_len',
        type=int,
        default=12,
        help='immutable fine observation length used by the working-state read',
    )
    parser.add_argument(
        '--working_state_bundle_size',
        type=int,
        default=4,
        help='number of ordered P12 observations exposed by each memory value',
    )
    parser.add_argument(
        '--working_state_steps',
        type=int,
        default=2,
        help='shared recurrent query/read/update iterations per Q1 call',
    )
    parser.add_argument(
        '--working_state_value_mode',
        type=str,
        default='ordered',
        choices=['singleton', 'mean', 'ordered', 'shuffled'],
        help='matched-capacity value supplied at each addressed fine token',
    )
    parser.add_argument(
        '--working_state_read_mode',
        type=str,
        default='dynamic',
        choices=['dynamic', 'repeat_first'],
        help='re-address history after each update or reuse the first context',
    )
    parser.add_argument(
        '--working_state_heads',
        type=int,
        default=4,
        help='attention heads used by the recurrent working-state reader',
    )
    parser.add_argument(
        '--working_state_bound',
        type=float,
        default=0.1,
        help='absolute normalized next-patch correction bound',
    )
    parser.add_argument(
        '--working_state_penalty',
        type=float,
        default=0.0,
        help='optional squared working-state correction penalty',
    )
    parser.add_argument(
        '--ablation_self_logit_bias',
        type=float,
        default=0.0,
        help='fixed additive bias on causal attention diagonal logits',
    )
    parser.add_argument(
        '--ablation_self_edge_heads',
        type=int,
        default=-1,
        help='-1 keeps all self edges; otherwise number of heads with them',
    )
    parser.add_argument(
        '--ablation_self_history_mode',
        type=str,
        default='joint',
        choices=['joint', 'fixed', 'learned_scalar',
                 'learned_head', 'learned_query',
                 'evidence_sum', 'evidence_count025',
                 'evidence_count05', 'evidence_count075',
                 'evidence_mean', 'evidence_max'],
        help='joint softmax or factorized current-state/history routing',
    )
    parser.add_argument(
        '--ablation_self_history_init',
        type=float,
        default=0.5,
        help='initial or fixed current-value mass for factorized routing',
    )
    parser.add_argument(
        '--ablation_self_history_pattern',
        type=str,
        default='all',
        choices=['all', 'first', 'last'],
        help='layers receiving non-joint self/history routing',
    )
    parser.add_argument(
        '--ablation_qk_input_mode',
        type=str,
        default='state',
        choices=['state', 'delta', 'detrended',
                 'query_delta', 'key_delta',
                 'query_detrended', 'key_detrended'],
        help='token representation used by Q/K while V retains state',
    )
    parser.add_argument(
        '--ablation_head_sharing',
        type=str,
        default='none',
        choices=['none', 'query', 'key', 'value',
                 'query_key', 'key_value'],
        help='average selected projected roles across attention heads',
    )
    parser.add_argument(
        '--ablation_attention_window',
        type=int,
        default=0,
        help='zero uses the full causal prefix; otherwise token span',
    )
    parser.add_argument(
        '--ablation_coarse_memory_group',
        type=int,
        default=0,
        choices=[0, 2, 4, 8],
        help='non-overlapping causal key/value pooling width; zero disables',
    )
    parser.add_argument(
        '--ablation_coarse_memory_gate_init',
        type=float,
        default=0.0,
        help='initial bounded residual gate for coarse memory',
    )
    parser.add_argument(
        '--ablation_address_geometry',
        type=str,
        default='single',
        choices=['single', 'level_change', 'query_change',
                 'query_change_shared', 'query_change_fixed',
                 'query_change_eval_shared',
                 'query_change_distribution',
                 'query_change_independent',
                 'query_change_distribution_independent',
                 'query_change_dynamic', 'query_multiscale',
                 'query_local_change', 'query_short_multiscale',
                 'query_reference_mix'],
        help='ordinary state address or learned level/change score residual',
    )
    parser.add_argument(
        '--ablation_dyadic_router_mode',
        type=str,
        default='none',
        choices=['none', 'count', 'innovation',
                 'soft_count', 'soft_innovation'],
        help='factor attention into dyadic lag groups and route group mass',
    )
    parser.add_argument(
        '--ablation_dyadic_router_gate_init',
        type=float,
        default=0.0,
        help='initial bounded count/innovation dyadic-routing gate',
    )
    parser.add_argument(
        '--ablation_geometry_gate_init',
        type=float,
        default=0.0,
        help='initial bounded residual gates for level/change scores',
    )
    parser.add_argument(
        '--ablation_cross_channel_mixer',
        action='store_true',
        help='mix normalized channel predictions with a zero-init residual',
    )
    parser.add_argument(
        '--ablation_variable_mixer',
        type=str,
        default='none',
        choices=[
            'none', 'full', 'row_softmax', 'low_rank',
            'low_rank_orthogonal', 'mean', 'dynamic_slots'],
        help=(
            'protected per-patch hidden interaction across variables; '
            'dynamic_slots uses content-routed latent variable summaries'),
    )
    parser.add_argument(
        '--ablation_variable_mixer_source',
        type=str,
        default='state',
        choices=['state', 'temporal_innovation', 'channel_centered'],
        help='hidden quantity passed through the variable interaction',
    )
    parser.add_argument(
        '--ablation_variable_mixer_placement',
        type=str,
        default='post_attention',
        choices=['pre_attention', 'post_attention', 'post_ffn'],
        help='Transformer sublayer boundary receiving variable interaction',
    )
    parser.add_argument(
        '--ablation_variable_mixer_pattern',
        type=str,
        default='all_shared',
        choices=['first', 'last', 'all_shared', 'all_independent'],
        help='layers using the static variable interaction operator',
    )
    parser.add_argument(
        '--ablation_variable_mixer_rank',
        type=int,
        default=8,
        help='factor rank for the O(CR) low-rank variable mixer',
    )
    parser.add_argument(
        '--ablation_variable_mixer_gate_init',
        type=float,
        default=0.0,
        help='initial bounded residual gate; zero is exact CI fallback',
    )
    parser.add_argument(
        '--ablation_variable_mixer_gate_limit',
        type=float,
        default=1.0,
        help='maximum absolute residual gate for variable interaction',
    )
    parser.add_argument(
        '--ablation_channel_specific_head',
        action='store_true',
        help='add zero-init channel-specific patch decoding residuals',
    )
    parser.add_argument(
        '--ablation_persistence_gate_init',
        type=float,
        default=-2.0,
        help='enable next-patch persistence blending with this gate init',
    )
    parser.add_argument('--ablation_norm', type=str, default='post',
                        choices=['pre', 'post', 'none'],
                        help='normalization placement in TransformerAblationAR')
    parser.add_argument('--ablation_position_encoding', type=str,
                        default='rope', choices=['rope', 'none'],
                        help='position encoding in TransformerAblationAR')
    parser.add_argument('--ablation_attention_residual',
                        dest='ablation_attention_residual',
                        action='store_true',
                        help='retain the attention residual connection')
    parser.add_argument('--no_ablation_attention_residual',
                        dest='ablation_attention_residual',
                        action='store_false',
                        help='remove the attention residual connection')
    parser.add_argument('--ablation_ffn_residual',
                        dest='ablation_ffn_residual',
                        action='store_true',
                        help='retain the FFN residual connection')
    parser.add_argument('--no_ablation_ffn_residual',
                        dest='ablation_ffn_residual',
                        action='store_false',
                        help='remove the FFN residual connection')
    parser.set_defaults(
        ablation_attention_residual=True,
        ablation_ffn_residual=True,
    )
    parser.add_argument(
        '--external_autotimes_llm_checkpoint',
        type=str,
        default='',
        help='local Hugging Face GPT-2 checkpoint for AutoTimesOperatorAR',
    )
    parser.add_argument(
        '--external_autotimes_pretrained',
        type=str,
        default='',
        help='accepted AutoTimesOperatorAR Q1 or protected hierarchy checkpoint',
    )
    parser.add_argument(
        '--external_autotimes_mode',
        type=str,
        choices=['q1', 'clean', 'generated', 'staged'],
        default='q1',
        help='AutoTimes parent, clean/Generated-K control, or protected DPOD stage',
    )
    parser.add_argument(
        '--external_autotimes_token_len',
        type=int,
        default=96,
        help='native AutoTimes AR writeback length',
    )
    parser.add_argument(
        '--external_autotimes_mlp_hidden',
        type=int,
        default=512,
        help='AutoTimes continuous tokenizer and detokenizer hidden width',
    )
    parser.add_argument(
        '--external_autotimes_mlp_layers',
        type=int,
        default=2,
        help='AutoTimes continuous tokenizer and detokenizer layer count',
    )
    parser.add_argument(
        '--external_timer_checkpoint',
        type=str,
        default='',
        help='local official Timer checkpoint directory for TimerOperatorAR',
    )
    parser.add_argument(
        '--external_timer_mode',
        type=str,
        choices=['q1', 'generated'],
        default='q1',
        help='frozen Timer parent or Generated-K compiler mode',
    )
    parser.add_argument(
        '--external_timesfm_source',
        type=str,
        default='',
        help='local official TimesFM source directory (the src/ folder)',
    )
    parser.add_argument(
        '--external_timesfm_checkpoint',
        type=str,
        default='',
        help='local official TimesFM 2.5 checkpoint directory or safetensors file',
    )
    parser.add_argument(
        '--external_timesfm_mode',
        type=str,
        choices=['q1', 'generated'],
        default='q1',
        help='frozen TimesFM parent or Generated-K native-block compiler mode',
    )
    parser.add_argument(
        '--distilled_multi_commit_pretrained',
        type=str,
        default='',
        help='frozen Q1 checkpoint used as recursive distillation teacher',
    )
    parser.add_argument(
        '--distilled_multi_commit_hidden',
        type=int,
        default=256,
        help='hidden width of the isolated second-commit residual head',
    )
    parser.add_argument(
        '--distilled_multi_commit_patches',
        type=int,
        default=2,
        help='number of recursively distilled patches committed per forward',
    )
    parser.add_argument(
        '--distilled_multi_commit_eval_patches',
        type=int,
        default=0,
        help='prefix patches committed per evaluation call; zero uses K',
    )
    parser.add_argument(
        '--distilled_multi_commit_readout',
        type=str,
        default='last',
        choices=['last', 'innovation', 'tail_mean'],
        help='hidden-state summary exposed to macro residual heads',
    )
    parser.add_argument(
        '--distilled_multi_commit_readout_tail',
        type=int,
        default=4,
        help='tail width used by the tail-mean macro readout',
    )
    parser.add_argument(
        '--distilled_multi_commit_truth_weight',
        type=float,
        default=0.0,
        help='mix of real second-patch loss versus recursive-teacher loss',
    )
    parser.add_argument(
        '--distilled_multi_commit_band_betas',
        type=str,
        default=None,
        help='per-band beta schedule for the target-side contraction family '
             'y_beta = y_hat + beta[band(t)] * (y - y_hat); mutually '
             'exclusive with distilled_multi_commit_truth_weight',
    )
    parser.add_argument(
        '--distilled_multi_commit_geometry_components',
        type=str,
        default='',
        help='comma-separated level/shape/phase components retained when '
             'projecting truth residual into the Generated far target',
    )
    parser.add_argument(
        '--distilled_multi_commit_geometry_period',
        type=int,
        default=0,
        help='history-aligned period for geometric far targets; required '
             'when geometry components are enabled',
    )
    parser.add_argument(
        '--distilled_multi_commit_geometry_strength',
        type=float,
        default=1.0,
        help='fraction of the selected oracle geometry projection retained '
             'in the Generated far label',
    )
    parser.add_argument(
        '--distilled_multi_commit_geometry_component_strengths',
        type=str,
        default='',
        help='optional level,shape,phase projection fractions; mutually '
             'exclusive with a non-default global geometry strength',
    )
    parser.add_argument(
        '--distilled_multi_commit_truth_loss',
        type=str,
        choices=['mse', 'huber'],
        default='mse',
        help='robustness of the real-future macro objective',
    )
    parser.add_argument(
        '--geometric_coordinate_base_checkpoint',
        type=str,
        default='',
        help='trained Generated checkpoint wrapped by geometric coordinates',
    )
    parser.add_argument(
        '--geometric_coordinate_period',
        type=int,
        default=0,
        help='history-aligned period used by geometric coordinate templates',
    )
    parser.add_argument(
        '--closure_anchor_gamma',
        type=float,
        default=1.0,
        help='global multiplier applied to the fixed closure-anchor betas',
    )
    parser.add_argument(
        '--closure_anchor_preserve_first_patch',
        action=argparse.BooleanOptionalAction,
        default=False,
        help='hard-preserve the Q1 slot in every closure-anchored proposal',
    )
    parser.add_argument(
        '--tokenwise_frozen_direct_target',
        type=str,
        choices=['truth', 'q1_teacher_forced', 'q1_autoregressive'],
        default='truth',
        help='second-successor target for the shared token-wise frozen K2 head',
    )
    parser.add_argument(
        '--dense_offset_update_mode',
        type=str,
        choices=['frozen', 'joint'],
        default='frozen',
        help='freeze Q1 or jointly update it with dense clean offset losses',
    )
    parser.add_argument(
        '--dense_offset_decoder_block_patches',
        type=int,
        default=1,
        help=(
            'encoded-patch widths sharing one monolithic decoder MLP; one '
            'keeps every far offset independent'),
    )
    parser.add_argument(
        '--dense_offset_capacity_match',
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            'scale monolithic block hidden widths to the parameter budget of '
            'the corresponding independent offset heads'),
    )
    parser.add_argument(
        '--dense_offset_anchor_weight',
        type=float,
        default=1.0,
        help='dense next-patch anchor weight when Q1 is jointly updated',
    )
    parser.add_argument(
        '--dense_offset_far_weight',
        type=float,
        default=1.0,
        help='mean clean far-offset loss weight',
    )
    parser.add_argument(
        '--dense_offset_backbone_lr_scale',
        type=float,
        default=0.1,
        help='Q1 learning-rate multiplier in joint dense-offset training',
    )
    parser.add_argument(
        '--latent_closure_hidden',
        type=int,
        default=256,
        help='hidden width of the successor-state transition',
    )
    parser.add_argument(
        '--latent_closure_state_weight',
        type=float,
        default=0.0,
        help='weight of the recursive successor hidden-state target',
    )
    parser.add_argument(
        '--latent_closure_truth_weight',
        type=float,
        default=0.3,
        help='real-future mix in latent-closure output training',
    )
    parser.add_argument(
        '--latent_closure_transition_bound',
        type=float,
        default=2.0,
        help='componentwise bound on the successor hidden residual',
    )
    parser.add_argument(
        '--behavioral_closure_steps',
        type=int,
        default=4,
        help='decoded future steps supervising the recurrent closure state',
    )
    parser.add_argument(
        '--rebased_multi_commit_include_cocycle',
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            'append exact successor mean-shift and log-scale coordinates '
            'to the rebased K2 macro readout'
        ),
    )
    parser.add_argument(
        '--history_read_multi_commit_heads',
        type=int,
        default=0,
        help=(
            'heads in the protected K2 single-query history read; zero '
            'uses the frozen parent head count'
        ),
    )
    parser.add_argument(
        '--history_read_multi_commit_mode',
        type=str,
        choices=['attention', 'dot', 'projected_mean', 'mean', 'last'],
        default='attention',
        help='dynamic history read or equally wide non-addressing controls',
    )
    parser.add_argument(
        '--history_read_multi_commit_temperature',
        type=float,
        default=0.25,
        help='softmax temperature for parameter-free dot history reads',
    )
    parser.add_argument(
        '--history_read_local_kernel',
        type=int,
        default=0,
        help=(
            'zero disables the bounded causal local sidecar over frozen '
            'history-read memory'
        ),
    )
    parser.add_argument(
        '--history_read_local_bound',
        type=float,
        default=0.1,
        help='componentwise bound on the history-read local state residual',
    )
    parser.add_argument(
        '--history_read_local_input',
        type=str,
        choices=['state', 'innovation'],
        default='state',
        help='normalized state or first difference seen by the local sidecar',
    )
    parser.add_argument(
        '--history_read_local_pretrained',
        type=str,
        default='',
        help=(
            'protected DHR-K2 checkpoint; only a zero-initialized local '
            'history sidecar remains trainable'
        ),
    )
    parser.add_argument(
        '--history_read_raw_input',
        type=str,
        choices=['none', 'patch', 'shape'],
        default='none',
        help=(
            'zero-init raw normalized patch or within-patch shape sidecar '
            'for the protected history reader'
        ),
    )
    parser.add_argument(
        '--history_read_raw_bound',
        type=float,
        default=0.1,
        help='componentwise bound on the raw-patch history residual',
    )
    parser.add_argument(
        '--history_read_raw_pretrained',
        type=str,
        default='',
        help=(
            'protected DHR-K2 checkpoint; only the zero-initialized raw '
            'patch projection remains trainable'
        ),
    )
    parser.add_argument(
        '--history_read_patch_token_kernel',
        type=int,
        default=0,
        help=(
            'zero disables the causal local sidecar over frozen '
            'pre-Transformer patch embeddings'
        ),
    )
    parser.add_argument(
        '--history_read_patch_token_bound',
        type=float,
        default=0.1,
        help='componentwise bound on the patch-token local residual',
    )
    parser.add_argument(
        '--history_read_patch_token_input',
        type=str,
        choices=['state', 'innovation'],
        default='state',
        help='patch embedding state or first difference seen by the sidecar',
    )
    parser.add_argument(
        '--history_read_patch_token_pretrained',
        type=str,
        default='',
        help=(
            'protected DHR-K2 checkpoint; only the zero-init patch-token '
            'local mixer remains trainable'
        ),
    )
    parser.add_argument(
        '--fine_slot_decoder_patch_len',
        type=int,
        default=12,
        help=(
            'ordered fine-patch slot length used by the protected shared '
            'macro calibration decoder'
        ),
    )
    parser.add_argument(
        '--coadaptive_history_read_pretrained',
        type=str,
        default='',
        help='exact C-DHR checkpoint used to initialize macro co-adaptation',
    )
    parser.add_argument(
        '--coadaptive_fine_patch_len',
        type=int,
        default=12,
        help='fixed atom length for the co-adaptive fine history memory',
    )
    parser.add_argument(
        '--coadaptive_fine_mode',
        type=str,
        choices=['none', 'state', 'innovation'],
        default='none',
        help='reader-only control or P12 state/innovation history memory',
    )
    parser.add_argument(
        '--coadaptive_fine_pool',
        type=str,
        choices=['all', 'parent_mean'],
        default='all',
        help='retain every P12 token or mean-pool within each parent patch',
    )
    parser.add_argument(
        '--coadaptive_fine_integration',
        type=str,
        choices=['cross_attention', 'aligned_residual'],
        default='cross_attention',
        help='use a second attention or inject aligned P12 memory into DHR',
    )
    parser.add_argument(
        '--coadaptive_fine_bound',
        type=float,
        default=0.1,
        help='componentwise bound on the fine-memory context residual',
    )
    parser.add_argument(
        '--coadaptive_reader_lr_scale',
        type=float,
        default=0.1,
        help='learning-rate multiplier for the pretrained macro reader',
    )
    parser.add_argument(
        '--coadaptive_trust_pretrained',
        type=str,
        default='',
        help='trained fine-history candidate wrapped by protected scalar trust',
    )
    parser.add_argument(
        '--coadaptive_trust_max',
        type=float,
        default=1.0,
        help='maximum absolute candidate interpolation coefficient',
    )
    parser.add_argument(
        '--coadaptive_trust_lr_scale',
        type=float,
        default=10.0,
        help='learning-rate multiplier for the single trust parameter',
    )
    parser.add_argument(
        '--coadaptive_trust_schedule',
        type=str,
        choices=['constant', 'linear', 'frontload', 'backload'],
        default='constant',
        help='constant, linear, or endpoint-pulsed trust over rollout calls',
    )
    parser.add_argument(
        '--coadaptive_trust_schedule_start',
        type=float,
        default=0.5,
        help='first rollout trust used by a linear schedule',
    )
    parser.add_argument(
        '--coadaptive_trust_schedule_end',
        type=float,
        default=0.5,
        help='last rollout trust used by a linear schedule',
    )
    parser.add_argument(
        '--coadaptive_trust_schedule_steps',
        type=int,
        default=2,
        help='number of early/late rollout calls receiving pulsed trust',
    )
    parser.add_argument(
        '--progressive_history_read_pretrained',
        type=str,
        default='',
        help='protected learned-attention K2 checkpoint expanded to K4',
    )
    parser.add_argument(
        '--progressive_history_read_patches',
        type=int,
        default=4,
        help='far exit width for protected progressive history reads',
    )
    parser.add_argument(
        '--pushforward_history_read_pretrained',
        type=str,
        default='',
        help='protected progressive history-read K4 corrected on occupancy',
    )
    parser.add_argument(
        '--pushforward_history_read_hidden',
        type=int,
        default=32,
        help='hidden width of isolated K3/K4 occupancy corrections',
    )
    parser.add_argument(
        '--pushforward_history_read_bound',
        type=float,
        default=0.05,
        help='componentwise normalized bound of occupancy corrections',
    )
    parser.add_argument(
        '--pushforward_history_read_origin_weight',
        type=float,
        default=0.5,
        help='mix of origin versus one-K4-roll-in truth correction',
    )
    parser.add_argument(
        '--pushforward_history_read_penalty',
        type=float,
        default=0.0,
        help='quadratic correction penalty',
    )
    parser.add_argument(
        '--clean_target_onpolicy_pretrained',
        type=str,
        default='',
        help='immutable calibrated DHR-K2 checkpoint used by clean K4',
    )
    parser.add_argument(
        '--clean_target_onpolicy_far_pretrained',
        type=str,
        default='',
        help='progressive DHR-K4 checkpoint initializing the far exits',
    )
    parser.add_argument(
        '--clean_target_onpolicy_hidden',
        type=int,
        default=256,
        help='hidden width of the retrained K3/K4 heads',
    )
    parser.add_argument(
        '--clean_target_onpolicy_visited_weight',
        type=float,
        default=0.5,
        help='mixture weight for detached K4 visited-state supervision',
    )
    parser.add_argument(
        '--clean_target_onpolicy_teacher_weight',
        type=float,
        default=0.7,
        help='Q1 true-prefix versus real-future Huber target weight',
    )
    parser.add_argument(
        '--clean_target_onpolicy_rollin_steps',
        type=int,
        default=1,
        help='detached K4 blocks before visited-state supervision',
    )
    parser.add_argument(
        '--clean_target_onpolicy_huber_delta',
        type=float,
        default=1.0,
        help='Huber transition point for clean and visited far targets',
    )
    parser.add_argument(
        '--calibrated_history_cocycle_bound',
        type=float,
        default=0.05,
        help=(
            'componentwise normalized correction bound for the '
            'C-DHR history-conditioned K3/K4 cocycle'
        ),
    )
    parser.add_argument(
        '--calibrated_history_cocycle_teacher',
        type=str,
        choices=['true_prefix', 'protected_composition'],
        default='true_prefix',
        help=(
            'true-history Q1 or protected C-DHR composition target for '
            'the history-conditioned K3/K4 cocycle'
        ),
    )
    parser.add_argument(
        '--clean_target_shrink_pretrained',
        type=str,
        default='',
        help='trained clean-target K4 endpoint used by protected shrinkage',
    )
    parser.add_argument(
        '--clean_target_shrink_alpha',
        type=float,
        default=0.25,
        help='fixed interpolation strength from original to clean far exits',
    )
    parser.add_argument(
        '--reliability_gate_clean_pretrained',
        type=str,
        default='',
        help='trained clean-target K4 endpoint used by reliability gating',
    )
    parser.add_argument(
        '--reliability_gate_max_alpha',
        type=float,
        default=0.25,
        help='absolute trust bound around the original K4 far exits',
    )
    parser.add_argument(
        '--reliability_gate_visited_weight',
        type=float,
        default=0.0,
        help='mixture weight for detached-rollout gate supervision',
    )
    parser.add_argument(
        '--reliability_gate_penalty',
        type=float,
        default=0.0,
        help='quadratic penalty on the state-dependent clean trust',
    )
    parser.add_argument(
        '--reliability_gate_rollin_steps',
        type=int,
        default=1,
        help='detached K4 blocks before visited gate supervision',
    )
    parser.add_argument(
        '--reliability_gate_huber_delta',
        type=float,
        default=1.0,
        help='Huber transition point for gate risk',
    )
    parser.add_argument(
        '--reliability_gate_hidden',
        type=int,
        default=16,
        help='hidden width of the low-capacity reliability gate',
    )
    parser.add_argument(
        '--reliability_gate_mode',
        type=str,
        choices=['state', 'scalar'],
        default='state',
        help='state-dependent gate or one learned global trust scalar',
    )
    parser.add_argument(
        '--residual_history_read_pretrained',
        type=str,
        default='',
        help='trained last-hidden K2 protected by a history-read residual',
    )
    parser.add_argument(
        '--residual_history_read_heads',
        type=int,
        default=8,
        help='heads in the dynamic addressing residual',
    )
    parser.add_argument(
        '--residual_history_read_hidden',
        type=int,
        default=128,
        help='hidden width of the bounded history-read correction',
    )
    parser.add_argument(
        '--residual_history_read_bound',
        type=float,
        default=0.05,
        help='componentwise normalized bound around the trained last-K2',
    )
    parser.add_argument(
        '--constrained_correction_pretrained',
        type=str,
        default='',
        help='frozen pure PRCD checkpoint corrected toward real labels',
    )
    parser.add_argument(
        '--constrained_correction_hidden',
        type=int,
        default=256,
        help='hidden width of the isolated truth-correction head',
    )
    parser.add_argument(
        '--constrained_correction_max_delta',
        type=float,
        default=0.25,
        help='componentwise normalized correction bound',
    )
    parser.add_argument(
        '--constrained_correction_comp_epsilon',
        type=float,
        default=0.1,
        help='relative composition-defect budget above pure PRCD',
    )
    parser.add_argument(
        '--constrained_correction_gain_limit',
        type=float,
        default=0.0,
        help='local sensitivity ratio budget versus pure PRCD; zero disables',
    )
    parser.add_argument(
        '--constrained_correction_perturb_scale',
        type=float,
        default=0.01,
        help='history perturbation scale for the local gain probe',
    )
    parser.add_argument(
        '--constrained_correction_dual_lr',
        type=float,
        default=0.05,
        help='projected dual-ascent rate for correction constraints',
    )
    parser.add_argument(
        '--constrained_correction_dual_max',
        type=float,
        default=100.0,
        help='maximum projected Lagrange multiplier',
    )
    parser.add_argument(
        '--constrained_correction_penalty',
        type=float,
        default=1.0,
        help='quadratic augmented-Lagrangian penalty',
    )
    parser.add_argument(
        '--constrained_correction_truth_loss',
        type=str,
        default='mse',
        choices=['mse', 'huber'],
        help='robust loss used to learn the real-label correction',
    )
    parser.add_argument(
        '--constrained_correction_huber_delta',
        type=float,
        default=1.0,
        help='Huber transition point for constrained truth correction',
    )
    parser.add_argument(
        '--occupancy_composition_pretrained',
        type=str,
        default='',
        help='pure PRCD checkpoint continued on macro-visited states',
    )
    parser.add_argument(
        '--occupancy_composition_rollin_steps',
        type=int,
        default=1,
        help='detached macro commits before querying the frozen teacher',
    )
    parser.add_argument(
        '--occupancy_composition_weight',
        type=float,
        default=0.5,
        help='mixture weight for visited-state composition loss',
    )
    parser.add_argument(
        '--internal_coarse_pretrained',
        type=str,
        default='',
        help='optional pure PRCD checkpoint used to initialize joint training',
    )
    parser.add_argument(
        '--internal_coarse_composition_weight',
        type=float,
        default=1.0,
        help='recursive-composition objective weight in joint training',
    )
    parser.add_argument(
        '--internal_coarse_truth_weight',
        type=float,
        default=0.0,
        help='real second-patch objective weight in joint training',
    )
    parser.add_argument(
        '--internal_coarse_gradient_mode',
        type=str,
        default='anchor_pcgrad',
        choices=[
            'sum',
            'anchor_pcgrad',
            'anchor_norm_pcgrad_amp',
        ],
        help='ordinary loss sum or Q1-anchor-safe macro gradient projection',
    )
    parser.add_argument(
        '--internal_coarse_backbone_lr_scale',
        type=float,
        default=0.1,
        help='shared Q1 learning-rate multiplier relative to the macro head',
    )
    parser.add_argument(
        '--internal_coarse_reference_weight',
        type=float,
        default=0.0,
        help='function-space trust-region weight toward the initial Q1 map',
    )
    parser.add_argument(
        '--internal_coarse_prefix_balanced',
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            'average macro regression risk across every '
            'deployable commit prefix'
        ),
    )
    parser.add_argument(
        '--progressive_nested_pretrained',
        type=str,
        default='',
        help=(
            'trained K2 checkpoint whose first two exits '
            'remain immutable during Kmax expansion'
        ),
    )
    parser.add_argument(
        '--progressive_nested_teacher',
        type=str,
        default='q1',
        choices=['q1', 'protected_k2', 'protected_prefix'],
        help=(
            'compose either Q1 or the immutable K-patch prefix '
            'when supervising new exits'
        ),
    )
    parser.add_argument(
        '--progressive_nested_protected_patches',
        type=int,
        default=2,
        help=(
            'number of immutable prefix exits in a progressive '
            'dyadic expansion'
        ),
    )
    parser.add_argument(
        '--progressive_nested_state_weight',
        type=float,
        default=0.0,
        help=(
            'weight for matching the frozen backbone state after '
            'student and composed-teacher block commits'
        ),
    )
    parser.add_argument(
        '--progressive_nested_canonical_weight',
        type=float,
        default=0.0,
        help=(
            'weight for matching the affine-canonicalized next '
            'history induced by a macro commit to the real successor'
        ),
    )
    parser.add_argument(
        '--progressive_nested_canonical_context_only',
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            'apply canonical-successor matching only to retained '
            'observed context, isolating normalization-state drift'
        ),
    )
    parser.add_argument(
        '--progressive_nested_occupancy_weight',
        type=float,
        default=0.0,
        help=(
            'mixture weight for protected composition distillation '
            'on detached histories visited by the macro student'
        ),
    )
    parser.add_argument(
        '--progressive_nested_occupancy_rollin_steps',
        type=int,
        default=1,
        help='detached macro commits before occupancy distillation',
    )
    parser.add_argument(
        '--progressive_nested_occupancy_average_rollins',
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            'average composition loss over every detached rollout '
            'depth from 1 through occupancy_rollin_steps'
        ),
    )
    parser.add_argument(
        '--progressive_nested_parent_function_weight',
        type=float,
        default=0.0,
        help=(
            'function-space proximal weight toward the loaded '
            'macro parent on observed histories'
        ),
    )
    parser.add_argument(
        '--progressive_nested_tangent_weight',
        type=float,
        default=0.0,
        help=(
            'weight for finite-difference Jacobian matching between the '
            'new 2K exits and the protected K-composition teacher'
        ),
    )
    parser.add_argument(
        '--progressive_nested_tangent_mode',
        type=str,
        choices=['match', 'excess_gain'],
        default='match',
        help=(
            'match the protected teacher tangent, or only penalize '
            'student directional gain above the protected teacher'
        ),
    )
    parser.add_argument(
        '--progressive_nested_tangent_epsilon',
        type=float,
        default=0.01,
        help='relative history perturbation used for tangent matching',
    )
    parser.add_argument(
        '--progressive_nested_tangent_floor',
        type=float,
        default=1e-3,
        help='energy floor in the relative tangent mismatch',
    )
    parser.add_argument(
        '--protected_pushforward_pretrained',
        type=str,
        default='',
        help=(
            'trained progressive Kmax checkpoint continued with '
            'protected pushforward correction heads'
        ),
    )
    parser.add_argument(
        '--protected_pushforward_rollin_steps',
        type=int,
        default=1,
        help='detached Kmax blocks before the on-policy truth loss',
    )
    parser.add_argument(
        '--protected_pushforward_rollin_commit_patches',
        type=int,
        default=0,
        help=(
            'patches committed per detached pushforward roll-in; '
            'non-positive uses the full trained macro width'
        ),
    )
    parser.add_argument(
        '--protected_pushforward_weight',
        type=float,
        default=0.5,
        help='mixture weight of visited-state versus clean-state loss',
    )
    parser.add_argument(
        '--protected_pushforward_truth_weight',
        type=float,
        default=0.1,
        help='real-future weight on the visited macro state',
    )
    parser.add_argument(
        '--protected_pushforward_correction_penalty',
        type=float,
        default=0.0,
        help='quadratic penalty on the new pushforward correction',
    )
    parser.add_argument(
        '--protected_pushforward_correction_bound',
        type=float,
        default=0.0,
        help=(
            'absolute normalized correction bound; zero leaves '
            'the correction unbounded'
        ),
    )
    parser.add_argument(
        '--protected_pushforward_truth_loss',
        type=str,
        default='mse',
        choices=['mse', 'huber'],
        help='risk used for aligned real-future pushforward targets',
    )
    parser.add_argument(
        '--protected_pushforward_truth_loss_space',
        type=str,
        default='normalized',
        choices=['normalized', 'point'],
        help=(
            'coordinate system for aligned truth risk; point '
            'matches the globally standardized test MSE'
        ),
    )
    parser.add_argument(
        '--protected_pushforward_huber_delta',
        type=float,
        default=1.0,
        help='Huber transition point in the normalized patch space',
    )
    parser.add_argument(
        '--protected_pushforward_closed_loop_gradient',
        action=argparse.BooleanOptionalAction,
        default=False,
        help='retain gradients through generated roll-in states',
    )
    parser.add_argument(
        '--protected_pushforward_closed_loop_base_jacobian',
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            'also retain the frozen far-step base-model input '
            'Jacobian during closed-loop training'
        ),
    )
    parser.add_argument(
        '--protected_pushforward_closed_loop_gradient_scale',
        type=float,
        default=1.0,
        help=(
            'backward-only discount applied per generated roll-in '
            'transition; forward predictions are unchanged'
        ),
    )
    parser.add_argument(
        '--protected_pushforward_cached_training',
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            'materialize frozen clean/roll-in states once and train '
            'only the correction heads from the resulting cache'
        ),
    )
    parser.add_argument(
        '--protected_pushforward_cache_refreshes',
        type=int,
        default=0,
        help=(
            'number of fitted-policy cache refreshes; zero uses only '
            'the initial parent occupancy'
        ),
    )
    parser.add_argument(
        '--protected_pushforward_cache_epochs',
        type=int,
        default=0,
        help=(
            'cached head-only epochs before switching to online '
            'pushforward training; zero never switches'
        ),
    )
    parser.add_argument(
        '--protected_pushforward_cache_shuffle_unit',
        type=str,
        default='window',
        choices=['window', 'channel'],
        help='shuffle cached examples by complete window or by channel',
    )
    parser.add_argument(
        '--protected_pushforward_cache_fraction',
        type=float,
        default=1.0,
        help='random fraction of training batches materialized per cache',
    )
    parser.add_argument(
        '--direct_patch_cached_training',
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            'materialize a model-defined frozen-state regression cache '
            'for generic direct-patch training'
        ),
    )
    parser.add_argument(
        '--direct_patch_cache_refreshes',
        type=int,
        default=0,
        help='number of generic direct-patch cache refreshes',
    )
    parser.add_argument(
        '--direct_patch_cache_epochs',
        type=int,
        default=0,
        help=(
            'cached epochs before switching back to online '
            'direct-patch training; zero never switches'
        ),
    )
    parser.add_argument(
        '--direct_patch_cache_shuffle_unit',
        type=str,
        default='window',
        choices=['window', 'channel'],
        help='shuffle generic cached examples by window or channel',
    )
    parser.add_argument(
        '--direct_patch_cache_fraction',
        type=float,
        default=1.0,
        help='fraction of training batches used by the generic cache',
    )
    parser.add_argument(
        '--protected_pushforward_head_type',
        type=str,
        default='mlp',
        choices=[
            'mlp', 'linear', 'mean_contrast', 'dct_mlp',
            'secant_mlp', 'secant_full_mlp',
            'gated_secant_mlp', 'kinematic_mlp',
            'kinematic_extrapolation_mlp',
        ],
        help='capacity of each protected far-exit correction',
    )
    parser.add_argument(
        '--protected_pushforward_hidden',
        type=int,
        default=0,
        help=(
            'correction MLP width; zero reuses the macro-head width'
        ),
    )
    parser.add_argument(
        '--protected_pushforward_common_fraction',
        type=float,
        default=0.75,
        help=(
            'fraction of the two-exit mean-contrast hidden budget '
            'assigned to the common correction'
        ),
    )
    parser.add_argument(
        '--protected_pushforward_basis_rank',
        type=int,
        default=8,
        help='number of low-frequency DCT correction modes',
    )
    parser.add_argument(
        '--continuation_ppmc_pretrained',
        type=str,
        default='',
        help='trained PPMC checkpoint for continuation correction',
    )
    parser.add_argument(
        '--continuation_correction_hidden',
        type=int,
        default=32,
        help='hidden width of continuation-only protected-prefix heads',
    )
    parser.add_argument(
        '--continuation_correction_bound',
        type=float,
        default=0.02,
        help='pointwise normalized bound for continuation correction',
    )
    parser.add_argument(
        '--continuation_correction_penalty',
        type=float,
        default=0.0,
        help='quadratic penalty on continuation correction',
    )
    parser.add_argument(
        '--behavioral_cocycle_mode',
        type=str,
        default='recurrent_residual',
        choices=[
            'recurrent_residual',
            'memory_residual',
            'shared_decoder',
            'memory_shared_decoder',
        ],
        help=(
            'shared predictive-state transition topology for '
            'protected dyadic macro far exits'
        ),
    )
    parser.add_argument(
        '--behavioral_cocycle_hidden',
        type=int,
        default=32,
        help='hidden width of the shared behavioral correction readout',
    )
    parser.add_argument(
        '--behavioral_cocycle_memory_heads',
        type=int,
        default=8,
        help='heads used to re-read frozen history during internal steps',
    )
    parser.add_argument(
        '--behavioral_cocycle_memory_kernel',
        type=str,
        default='softmax',
        choices=['softmax', 'elu'],
        help=(
            'softmax read or cacheable ELU linear-attention sufficient '
            'statistics for the immutable history'
        ),
    )
    parser.add_argument(
        '--behavioral_cocycle_memory_ablation',
        type=str,
        default='dynamic',
        choices=[
            'dynamic',
            'centered_dynamic',
            'zero',
            'uniform',
            'shuffled_value',
        ],
        help='diagnostic intervention on the internal history read',
    )
    parser.add_argument(
        '--behavioral_cocycle_cache_training',
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            'reuse projected history keys/values in the training '
            'graph; inference caching is controlled separately'
        ),
    )
    parser.add_argument(
        '--behavioral_cocycle_cache_inference',
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            'reuse projected history keys/values across internal '
            'future steps at inference'
        ),
    )
    parser.add_argument(
        '--behavioral_cocycle_static_pretrained',
        type=str,
        default='',
        help=(
            'optional trained PPMC checkpoint used as the bounded '
            'static anchor before behavioral correction'
        ),
    )
    parser.add_argument(
        '--conditional_memory_baseline_hidden',
        type=int,
        default=32,
        help='state-only baseline width for conditional memory innovation',
    )
    parser.add_argument(
        '--conditional_memory_neighbors',
        type=int,
        default=4,
        help='counterfactual memories used per state for baseline targets',
    )
    parser.add_argument(
        '--conditional_memory_neighbor_mode',
        type=str,
        choices=['nearest', 'cyclic'],
        default='nearest',
        help='state-matched or cyclic counterfactual memory control',
    )
    parser.add_argument(
        '--conditional_memory_baseline_weight',
        type=float,
        default=1.0,
        help='weight of the amortized conditional-read baseline loss',
    )
    parser.add_argument(
        '--behavioral_query_distill_pretrained',
        type=str,
        default='',
        help=(
            'trained HCBC checkpoint used as the exact query-trajectory '
            'continuation endpoint'
        ),
    )
    parser.add_argument(
        '--behavioral_query_distill_weight',
        type=float,
        default=0.1,
        help=(
            'weight of free-Q1 address-distribution trajectory '
            'distillation'
        ),
    )
    parser.add_argument(
        '--strong_parent_target_patches',
        type=int,
        default=8,
        help=(
            '2K commit width distilled from a frozen behavioral K parent'
        ),
    )
    parser.add_argument(
        '--strong_parent_pretrained',
        type=str,
        default='',
        help='trained behavioral K parent kept as the exact prefix',
    )
    parser.add_argument(
        '--strong_parent_k2_pretrained',
        type=str,
        default='',
        help='matched DPOD-2K checkpoint used to initialize new far exits',
    )
    parser.add_argument(
        '--strong_parent_truth_weight',
        type=float,
        default=0.0,
        help=(
            'mixture weight of real far labels versus behavioral composition'
        ),
    )
    parser.add_argument(
        '--residual_state_pretrained',
        type=str,
        default='',
        help=(
            'trained PPMC checkpoint used as the zero-state anchor '
            'for protected residual-state dynamics'
        ),
    )
    parser.add_argument(
        '--residual_state_mode',
        type=str,
        default='state',
        choices=[
            'state',
            'memory',
            'innovation',
            'uniform_memory',
        ],
        help=(
            'forcing supplied to the protected residual-state '
            'transition'
        ),
    )
    parser.add_argument(
        '--residual_state_dim',
        type=int,
        default=32,
        help='dimension of the stable residual predictive state',
    )
    parser.add_argument(
        '--residual_state_max_decay',
        type=float,
        default=0.95,
        help=(
            'strict upper bound on the diagonal homogeneous '
            'residual-state dynamics'
        ),
    )
    parser.add_argument(
        '--residual_state_memory_heads',
        type=int,
        default=8,
        help='heads used by residual-state history addressing',
    )
    parser.add_argument(
        '--residual_state_orthogonality_weight',
        type=float,
        default=0.1,
        help=(
            'weight for the state-only regression baseline used '
            'to residualize memory reads'
        ),
    )
    parser.add_argument(
        '--residual_state_update_bound',
        type=float,
        default=0.0,
        help=(
            'optional pointwise trust region around the PPMC '
            'anchor in normalized output coordinates'
        ),
    )
    parser.add_argument(
        '--residual_state_cache_inference',
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            'cache immutable history key/value projections across '
            'internal residual-state steps'
        ),
    )
    parser.add_argument(
        '--scale_consistent_budget',
        type=float,
        default=0.05,
        help=(
            'reference per-point magnitude used to construct one '
            'global L2 correction budget'
        ),
    )
    parser.add_argument(
        '--scale_consistent_reference_exits',
        type=int,
        default=2,
        help=(
            'number of corrected exits at which the global budget '
            'equals the reference pointwise budget'
        ),
    )
    parser.add_argument(
        '--scale_consistent_point_bound',
        type=float,
        default=0.05,
        help=(
            'optional pointwise trust region composed with the '
            'global correction projection; zero disables it'
        ),
    )
    parser.add_argument(
        '--scale_consistent_projection_eps',
        type=float,
        default=1e-8,
        help='numerical epsilon for global correction projection',
    )
    parser.add_argument(
        '--scale_consistent_correction_gain',
        type=float,
        default=1.0,
        help=(
            'inference-time multiplier for globally projected '
            'corrections; zero exactly recovers the parent'
        ),
    )
    parser.add_argument(
        '--scale_consistent_truth_risk',
        type=str,
        default='absolute',
        choices=[
            'absolute',
            'relative',
            'log_ratio',
            'clipped_relative',
        ],
        help=(
            'aligned-future risk: ordinary MSE or dimensionless '
            'excess risk relative to the frozen parent'
        ),
    )
    parser.add_argument(
        '--scale_consistent_relative_floor',
        type=float,
        default=1.0,
        help=(
            'positive multiple of mean parent risk used to '
            'stabilize dimensionless truth risk'
        ),
    )
    parser.add_argument(
        '--scale_consistent_risk_quantile',
        type=float,
        default=0.5,
        help=(
            'parent-risk quantile defining the transition from '
            'absolute to relative excess-risk influence'
        ),
    )
    parser.add_argument(
        '--parallel_history_correction_hidden',
        type=int,
        default=32,
        help='shared decoder width for parallel protected corrections',
    )
    parser.add_argument(
        '--parallel_history_correction_heads',
        type=int,
        default=8,
        help='heads in the batched history read for all new exits',
    )
    parser.add_argument(
        '--parallel_history_correction_kernel',
        type=str,
        default='softmax',
        choices=['none', 'softmax', 'elu'],
        help='parallel history-read kernel for protected correction queries',
    )
    parser.add_argument(
        '--parallel_history_correction_use_proposal_prefix',
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            'condition each parallel correction query on the cumulative '
            'frozen macro proposal preceding that exit'
        ),
    )
    parser.add_argument(
        '--parallel_scan_cocycle_hidden',
        type=int,
        default=32,
        help='hidden width of the associative behavioral scan decoder',
    )
    parser.add_argument(
        '--parallel_scan_cocycle_heads',
        type=int,
        default=8,
        help='heads in the batched history forcing read',
    )
    parser.add_argument(
        '--parallel_scan_cocycle_use_history',
        action=argparse.BooleanOptionalAction,
        default=True,
        help='add one batched frozen-history read before the affine scan',
    )
    parser.add_argument(
        '--parallel_scan_cocycle_spectral_radius',
        type=float,
        default=0.9,
        help='strict upper bound on every diagonal scan transition',
    )
    parser.add_argument(
        '--moment_closure_pretrained',
        type=str,
        default='',
        help='trained TRIC checkpoint used to initialize MomentClosureAR',
    )
    parser.add_argument(
        '--moment_closure_use_history_stats',
        action=argparse.BooleanOptionalAction,
        default=True,
        help='condition macro moment correction on explicit history moments',
    )
    parser.add_argument(
        '--moment_closure_mean_bound',
        type=float,
        default=0.25,
        help='maximum normalized mean correction of the macro patch',
    )
    parser.add_argument(
        '--moment_closure_log_scale_bound',
        type=float,
        default=0.25,
        help='maximum absolute log-scale correction of the macro patch',
    )
    parser.add_argument(
        '--moment_closure_hidden',
        type=int,
        default=128,
        help='hidden width of the macro moment head',
    )
    parser.add_argument(
        '--moment_closure_freeze_base',
        action=argparse.BooleanOptionalAction,
        default=True,
        help='train only the moment head or jointly adapt the TRIC base',
    )
    parser.add_argument(
        '--reference_q1_pretrained',
        type=str,
        default='',
        help=(
            'pretrained Q1 checkpoint for the protected '
            'continuation control'),
    )
    parser.add_argument(
        '--reference_q1_weight',
        type=float,
        default=10.0,
        help=(
            'function-space reference penalty for protected '
            'Q1 continuation'),
    )
    parser.add_argument(
        '--reference_q1_lr_scale',
        type=float,
        default=0.01,
        help=(
            'learning-rate multiplier for protected '
            'Q1 continuation'),
    )
    parser.add_argument('--tail_ar_patches', type=int, default=1,
                        help='number of repeated TailAR query patches supervised during training')
    parser.add_argument('--tail_ar_roll_patches', type=int, default=1,
                        help='number of TailAR query patches committed per inference rollout')
    parser.add_argument('--tail_placeholder_type', type=str, default='zero',
                        choices=['zero', 'learned_patch', 'learned_patches',
                                 'coordinate_patch', 'learned_token', 'copy_last'],
                        help='TailAR placeholder before or after patch projection')
    parser.add_argument('--tail_separate_projection', action='store_true', default=False,
                        help='use a separate projection for TailAR forecast placeholders')
    parser.add_argument('--tail_independent_heads', action='store_true', default=False,
                        help='use one TailAR output head per supervised future patch')
    parser.add_argument('--tail_loss_sampling', type=str, default='all',
                        choices=['all', 'uniform', 'anchor'],
                        help='sample supervised TailAR future patches per batch')
    parser.add_argument('--tail_loss_sample_count', type=int, default=4,
                        help='number of TailAR patch losses sampled per batch')
    parser.add_argument('--tail_history_loss_weight', type=float, default=0.2,
                        help='historical next-patch auxiliary weight in HybridTailAR')
    parser.add_argument(
        '--block_recurrent_history_weight',
        type=float,
        default=-1.0,
        help=(
            'BlockRecurrentAR history-loss weight; negative keeps the '
            'token-count-weighted combined loss'
        ),
    )
    parser.add_argument(
        '--addressable_delta_mode',
        choices=['none', 'output', 'state'],
        default='output',
        help=(
            'future-state Delta memory topology: recurrent control, '
            'output-only read, or bounded state write-back'
        ),
    )
    parser.add_argument(
        '--addressable_delta_key_dim',
        type=int,
        default=64,
        help='key width of the shifted state-to-successor Delta memory',
    )
    parser.add_argument(
        '--addressable_delta_train_patches',
        type=int,
        default=0,
        help=(
            'future recurrent states supervised in training; zero uses '
            'dense_ar_roll_patches'
        ),
    )
    parser.add_argument(
        '--addressable_delta_beta_init',
        type=float,
        default=0.9,
        help='initial error-correcting Delta write strength',
    )
    parser.add_argument(
        '--addressable_delta_output_max_scale',
        type=float,
        default=0.25,
        help='maximum convex successor-patch correction',
    )
    parser.add_argument(
        '--addressable_delta_state_max_scale',
        type=float,
        default=0.1,
        help='maximum addressed successor-state write-back',
    )
    parser.add_argument(
        '--addressable_delta_gate_init',
        type=float,
        default=0.1,
        help='initial gate fraction within each maximum correction',
    )
    parser.add_argument(
        '--addressable_delta_read_multiplier',
        type=float,
        default=1.0,
        help='diagnostic multiplier for addressed output/state reads',
    )
    parser.add_argument(
        '--delta_placeholder_fine_patch_len',
        type=int,
        default=12,
        help='fine patch transferred through DeltaPlaceholderAR',
    )
    parser.add_argument(
        '--delta_placeholder_history_fusion',
        choices=['concat', 'delta_boundary'],
        default='delta_boundary',
        help='history representation exposed to the second Transformer',
    )
    parser.add_argument(
        '--delta_placeholder_attention_scope',
        choices=['none', 'placeholders', 'queries', 'all'],
        default='placeholders',
        help='tokens participating in post-Delta causal attention',
    )
    parser.add_argument(
        '--delta_placeholder_attention_fusion',
        choices=['replace', 'gated_residual'],
        default='replace',
        help='how the second Transformer updates Delta future states',
    )
    parser.add_argument(
        '--delta_placeholder_attention_gate_init',
        type=float,
        default=0.03,
        help='initial bounded Transformer correction applied to Delta state',
    )
    parser.add_argument(
        '--delta_placeholder_attention_gate_max',
        type=float,
        default=0.25,
        help='maximum bounded Transformer correction to Delta state',
    )
    parser.add_argument(
        '--delta_placeholder_history_weight',
        type=float,
        default=0.2,
        help='dense historical P12 loss weight; negative combines all tokens',
    )
    parser.add_argument(
        '--delta_placeholder_transformer_layers',
        type=int,
        default=1,
        help='Transformer layers after Delta state transfer',
    )
    parser.add_argument(
        '--delta_block_rollout_placeholder_blocks',
        type=int,
        choices=[0, 1],
        default=1,
        help='P48 placeholder blocks appended after observed Delta blocks',
    )
    parser.add_argument(
        '--delta_block_rollout_commit_blocks',
        type=int,
        default=0,
        help='predicted P48 blocks committed per outer rollout; zero uses all',
    )
    parser.add_argument(
        '--delta_block_rollout_train_target_blocks',
        type=int,
        default=0,
        help='P48 labels loaded in training; zero matches predicted blocks',
    )
    parser.add_argument(
        '--delta_block_rollout_eval_commit_blocks',
        type=int,
        default=0,
        help='inference-only commit override excluded from checkpoint setting',
    )
    parser.add_argument(
        '--delta_block_rollout_second_block_weight',
        type=float,
        default=1.0,
        help='relative training weight of the closure-decoded second block',
    )
    parser.add_argument(
        '--delta_block_rollout_state_consistency_weight',
        type=float,
        default=0.0,
        help='weight for counterfactual teacher-successor state consistency',
    )
    parser.add_argument(
        '--delta_block_rollout_conditioned_placeholder',
        action='store_true',
        default=False,
        help='condition the closure token on the final observed block state',
    )
    parser.add_argument(
        '--delta_block_rollout_predicted_feedback',
        action='store_true',
        default=False,
        help=(
            'replace the learned closure placeholder by a stop-gradient '
            'encoding of the first predicted block'),
    )
    parser.add_argument(
        '--delta_block_rollout_state_feedback_bound',
        type=float,
        default=0.0,
        help=(
            'elementwise bound rho for the state residual '
            'rho*tanh(residual/rho); zero leaves it unbounded'),
    )
    parser.add_argument(
        '--delta_block_rollout_independent_closure_head',
        action='store_true',
        default=False,
        help='decode the closure block with a separate paired-initialized head',
    )
    parser.add_argument(
        '--delta_block_rollout_isolate_closure_gradient',
        action='store_true',
        default=False,
        help='route the second-block objective only to closure parameters',
    )
    parser.add_argument(
        '--delta_block_rollout_anchor_selection',
        action='store_true',
        default=False,
        help='optimize both blocks but select checkpoints on the anchor loss',
    )
    parser.add_argument(
        '--delta_block_rollout_project_closure_gradient',
        action='store_true',
        default=False,
        help='project conflicting closure gradients off the anchor gradient',
    )
    parser.add_argument(
        '--delta_block_rollout_predictive_consistency_weight',
        type=float,
        default=0.0,
        help='weight for decoder-space observable closure consistency',
    )
    parser.add_argument(
        '--delta_block_rollout_predictive_consistency_steps',
        type=int,
        choices=[1, 2],
        default=1,
        help='number of observable successor coordinates to match',
    )
    parser.add_argument(
        '--delta_block_rollout_alternate_closure_period',
        type=int,
        default=0,
        help='use one closure-only batch per period; zero disables',
    )
    parser.add_argument(
        '--multiscale_coarse_aux_weight',
        type=float,
        default=0.25,
        help='isolated coarse-scale auxiliary loss weight',
    )
    parser.add_argument(
        '--multiscale_haar_contrast',
        action='store_true',
        default=False,
        help=(
            'add a protected half-block Haar contrast correction beside '
            'the block-mean correction'),
    )
    parser.add_argument(
        '--progressive_scale_factors',
        type=int,
        nargs='+',
        default=[1, 2, 4],
        help='increasing AR span multipliers from near to far horizon',
    )
    parser.add_argument(
        '--progressive_scale_boundaries',
        type=int,
        nargs='*',
        default=[96, 288],
        help='generated-point origins at which the AR scale increases',
    )
    parser.add_argument(
        '--progressive_scale_inference_factor',
        type=int,
        default=0,
        help='zero follows the progressive schedule; positive fixes one scale',
    )
    parser.add_argument(
        '--progressive_scale_interpolation',
        choices=['linear', 'repeat'],
        default='linear',
        help='reconstruction from coarse knots to the committed span',
    )
    parser.add_argument(
        '--progressive_scale_shared_backbone',
        action='store_true',
        default=False,
        help='share one Transformer across scales (legacy screen)',
    )
    parser.add_argument(
        '--progressive_scale_dense_tail_weight',
        type=float,
        default=1.0,
        help='coarse interpolated-tail loss added to dense transition loss',
    )
    parser.add_argument(
        '--recurrent_query_steps',
        type=int,
        default=2,
        help='shared history-read/update cycles for RecurrentQueryDirectAR',
    )
    parser.add_argument(
        '--recurrent_query_position_mode',
        type=str,
        default='monotone_adjacency',
        choices=['none', 'monotone', 'monotone_adjacency'],
        help='fixed relative relation used by recurrent query reads',
    )
    parser.add_argument(
        '--recurrent_query_read_mode',
        type=str,
        default='dynamic',
        choices=['dynamic', 'repeat_first'],
        help='re-address history each cycle or repeat the first read context',
    )
    parser.add_argument(
        '--recurrent_query_output_mode',
        type=str,
        default='last',
        choices=['last', 'mixture'],
        help='decode only the last query state or a convex anytime mixture',
    )
    parser.add_argument(
        '--recurrent_query_temperature',
        type=float,
        default=8.0,
        help='initial cosine-attention temperature for recurrent queries',
    )
    parser.add_argument(
        '--persistent_successor_mode',
        type=str,
        default='local',
        choices=[
            'local',
            'persistent_output',
            'persistent_residual',
            'persistent_transition',
            'rolling_output',
            'rolling_residual',
            'rolling_transition',
        ],
        help='working-window and shifted-successor topology',
    )
    parser.add_argument(
        '--persistent_successor_history_weight',
        type=float,
        default=0.0,
        help='strict shifted-causal historical successor loss weight',
    )
    parser.add_argument(
        '--persistent_successor_residual_scale',
        type=float,
        default=0.25,
        help='bounded raw-successor residual scale',
    )
    parser.add_argument(
        '--persistent_successor_residual_max_scale',
        type=float,
        default=None,
        help='residual scale after the rolling window loses all real history',
    )
    parser.add_argument(
        '--persistent_successor_residual_gate',
        type=str,
        default='real_fraction',
        choices=[
            'real_fraction',
            'confidence',
            'confidence_detached',
            'agreement_detached',
        ],
        help='schedule residual scale by real-history fraction or address confidence',
    )
    parser.add_argument(
        '--persistent_successor_temperature',
        type=float,
        default=8.0,
        help='cosine successor-address temperature',
    )
    parser.add_argument(
        '--persistent_successor_null_margin',
        type=float,
        default=1.0,
        help='NULL successor-address logit margin',
    )
    parser.add_argument(
        '--persistent_successor_loss_mode',
        type=str,
        default='mean',
        choices=[
            'mean',
            'anchor',
            'anchor_pcgrad',
            'sampled_far',
            'isolated_far',
        ],
        help='Direct-K horizon aggregation and gradient protection',
    )
    parser.add_argument(
        '--persistent_successor_far_weight',
        type=float,
        default=1.0,
        help='weight of each grouped far-horizon objective',
    )
    parser.add_argument(
        '--persistent_successor_gradient_group',
        type=int,
        default=2,
        help='number of adjacent far horizons per projected objective',
    )
    parser.add_argument(
        '--persistent_successor_far_sample_count',
        type=int,
        default=1,
        help='far horizons sampled per optimizer step in sampled_far mode',
    )
    parser.add_argument(
        '--persistent_successor_stats_mode',
        type=str,
        default='rolling',
        choices=['rolling', 'initial', 'expanding'],
        help='normalization statistics used during block-AR rollout',
    )
    parser.add_argument(
        '--persistent_successor_retained_real_patches',
        type=int,
        default=0,
        help='minimum recent real patches retained in the rolling window',
    )
    parser.add_argument('--recon_tail_fine_patch_len', type=int, default=12,
                        help='fine patch length in StateReconTailAR')
    parser.add_argument('--recon_tail_group_patches', type=int, default=4,
                        help='fine patches assembled into each reconstructed state')
    parser.add_argument('--recon_tail_future_points', type=int, default=96,
                        help='learned future points supervised in StateReconTailAR')
    parser.add_argument('--recon_tail_roll_groups', type=int, default=1,
                        help='decoded state groups committed per rollout')
    parser.add_argument('--recon_tail_train_rollouts', type=int, default=1,
                        help='number of differentiable fixed-window direct rollouts trained')
    parser.add_argument('--recon_tail_commit_points', type=int, default=0,
                        help='points committed from decoded groups; 0 uses all')
    parser.add_argument('--recon_tail_history_weight', type=float, default=0.1,
                        help='same-block historical reconstruction loss weight')
    parser.add_argument('--recon_tail_history_mode', type=str, default='same',
                        choices=['same', 'masked', 'aligned_query'],
                        help='same-block, masked, or same-position causal-query historical supervision')
    parser.add_argument('--recon_tail_history_mask_ratio', type=float,
                        default=0.25,
                        help='fraction of historical states masked for reconstruction')
    parser.add_argument('--recon_tail_query_context', type=str,
                        default='causal',
                        choices=['causal', 'self'],
                        help='aligned queries read prior queries or only their own query token')
    parser.add_argument('--recon_tail_query_layer_aggregation', type=str,
                        default='last',
                        choices=['last', 'mean'],
                        help='decode the final aligned-query layer or average predictions from every layer')
    parser.add_argument('--recon_tail_loss_space', type=str,
                        default='normalized',
                        choices=['normalized', 'point'],
                        help='compute StateReconTailAR losses before or after instance denormalization')
    parser.add_argument('--ar_norm_eps', type=float, default=1e-5,
                        help='variance epsilon for rolling instance normalization')
    parser.add_argument(
        '--endogenous_stats_pretrained',
        type=str,
        default='',
        help='pretrained Q1 checkpoint for EndogenousStatStateAR',
    )
    parser.add_argument(
        '--endogenous_stats_frame',
        type=str,
        default='local',
        choices=['origin', 'local'],
        help='immutable coordinate frame assigned to newly written patches',
    )
    parser.add_argument(
        '--endogenous_stats_conditioning',
        action=argparse.BooleanOptionalAction,
        default=True,
        help='embed moment drift into the recurrent token state',
    )
    parser.add_argument(
        '--endogenous_stats_conditioning_scope',
        type=str,
        default='both',
        choices=['token', 'output', 'both'],
        help='apply moment conditioning to cached tokens, output, or both',
    )
    parser.add_argument(
        '--endogenous_stats_train_steps',
        type=int,
        default=4,
        help='teacher-forced future origins used to train statistical state',
    )
    parser.add_argument(
        '--endogenous_stats_rollout_weight',
        type=float,
        default=1.0,
        help='weight of future-origin transition losses',
    )
    parser.add_argument(
        '--endogenous_stats_composition_weight',
        type=float,
        default=0.0,
        help='weight for matching a full rolling-recompute Q1 teacher',
    )
    parser.add_argument(
        '--endogenous_stats_rollout_source',
        type=str,
        default='truth',
        choices=['truth', 'prediction'],
        help='values used to advance the training-time statistical state',
    )
    parser.add_argument(
        '--endogenous_stats_reference_weight',
        type=float,
        default=10.0,
        help='function-space Q1 reference weight',
    )
    parser.add_argument(
        '--endogenous_stats_backbone_lr_scale',
        type=float,
        default=0.01,
        help='backbone learning-rate multiplier during statistical adaptation',
    )
    parser.add_argument(
        '--endogenous_stats_hidden',
        type=int,
        default=256,
        help='hidden width of the moment-conditioned residual head',
    )
    parser.add_argument(
        '--endogenous_stats_cache_policy',
        type=str,
        default='unbounded',
        choices=['unbounded', 'window'],
        help='retain all streaming tokens or evict to the initial window size',
    )
    parser.add_argument(
        '--endogenous_stats_embedding_bound',
        type=float,
        default=0.0,
        help='positive tanh bound for the statistical token embedding',
    )
    parser.add_argument(
        '--endogenous_stats_residual_bound',
        type=float,
        default=0.0,
        help='positive tanh bound for statistical output correction',
    )
    parser.add_argument(
        '--endogenous_stats_recompute_eval',
        action='store_true',
        default=False,
        help='recompute immutable token prefixes instead of using KV cache',
    )
    parser.add_argument('--ar_transformer_norm', type=str, default='post',
                        choices=['pre', 'post'],
                        help='residual normalization order for DenseAR and TailAR')
    parser.add_argument('--ar_mamba_d_state', type=int, default=16,
                        help='state width per expanded channel in Mamba AR')
    parser.add_argument('--ssm_retrieval_mode', type=str, default='none',
                        choices=['none', 'identity', 'uniform', 'static',
                                 'lag', 'softmax', 'static_only',
                                 'softmax_only'],
                        help='controlled retrieval side path for RetrievalMambaAR')
    parser.add_argument('--ssm_retrieval_lags', type=int, nargs='*',
                        default=[1, 2, 4, 7, 14],
                        help='relative token lags used by the lag retrieval path')
    parser.add_argument('--ssm_retrieval_scale_init', type=float, default=0.1,
                        help='initial residual scale of the retrieval side path')
    parser.add_argument('--ssm_mamba_scale_init', type=float, default=1.0,
                        help='initial learnable Mamba scale when retrieval is active')
    parser.add_argument(
        '--addressable_mamba_mode',
        type=str,
        default='none',
        choices=[
            'none',
            'snapshot_output',
            'snapshot_state',
            'snapshot_blend',
            'assoc_output',
            'assoc_write',
            'assoc_select',
            'token_output',
            'token_write',
            'token_select',
            'token_blend',
            'token_only',
            'state_query',
            'block_token_only',
            'block_state_query',
            'block_state_memory',
            'block_enriched_query',
            'block_enriched_memory',
            'block_gated_query',
            'block_gated_memory',
            'block_bounded_query',
            'block_bounded_memory',
            'block_router',
            'block_bounded_router',
            'block_content_residual',
            'block_state_residual',
            'block_state_dynamic_residual',
            'block_state_bounded_residual',
            'block_state_self_residual',
            'block_state_innovation_residual',
        ],
        help='where content-addressed history enters AddressableMambaAR',
    )
    parser.add_argument('--addressable_mamba_key_dim', type=int, default=16,
                        help='content-address width per Mamba head')
    parser.add_argument('--addressable_mamba_scale_init', type=float, default=0.1,
                        help='initial content-memory correction scale')
    parser.add_argument('--addressable_mamba_gate_init', type=float, default=0.1,
                        help='initial content-memory gate probability')
    parser.add_argument('--ar_mamba_fine_patch_len', type=int, default=12,
                        help='fine Mamba token length inside a coarse AR patch')
    parser.add_argument('--ar_mamba_lag_points', type=int, default=48,
                        help='point shift between a Mamba patch and its AR label')
    parser.add_argument('--ar_gru_slots', type=int, default=2,
                        help='recurrently generated future slots in GRUSlotAR')
    parser.add_argument('--direct_roll_schedule', type=int, nargs='+',
                        default=[1, 2, 4],
                        help='successive committed patch counts in ProgressiveDirectAR')
    parser.add_argument('--direct_history_patch_stride', type=int, default=24,
                        help='observed-history patch stride in OverlapDirectAR')
    parser.add_argument('--direct_position_encoding', type=str, default='none',
                        choices=['none', 'rope'],
                        help='position encoding used by OverlapDirectAR')
    parser.add_argument('--history_only_fine_patch_len', type=int, default=12,
                        help='shared fine-patch length in HistoryOnlyDirectAR')
    parser.add_argument('--history_only_group_patches', type=int, default=2,
                        help='fine patches concatenated into one direct token')
    parser.add_argument(
        '--history_only_position_mode',
        type=str,
        default='none',
        choices=[
            'none',
            'standard_rope',
            'dyadic_rope',
            'gaussian_bias',
            'adjacency_bias',
            'monotone_bias',
            'monotone_adjacency',
            'standard_rope_gaussian',
            'dyadic_rope_gaussian',
        ],
        help='relative address structure in HistoryOnlyDirectAR',
    )
    parser.add_argument(
        '--bounded_memory_mode',
        type=str,
        default='persistent',
        choices=['rolling_only', 'persistent'],
        help='whether Direct-K queries retain dropped original history',
    )
    parser.add_argument(
        '--bounded_memory_granularity',
        type=str,
        default='coarse',
        choices=['coarse', 'fine'],
        help='temporal resolution retained by the immutable history bank',
    )
    parser.add_argument(
        '--bounded_memory_stats_mode',
        type=str,
        default='initial',
        choices=['initial', 'rolling'],
        help='normalization frame for bounded-memory Direct-K rollout',
    )
    parser.add_argument(
        '--bounded_memory_position_mode',
        type=str,
        default='none',
        choices=['none', 'monotone', 'monotone_adjacency'],
        help='absolute patch relation in bounded-memory Direct-K rollout',
    )
    parser.add_argument(
        '--bounded_memory_fusion_mode',
        type=str,
        default='joint_attention',
        choices=['joint_attention', 'isolated_residual'],
        help='joint memory attention or a gradient-isolated residual read',
    )
    parser.add_argument(
        '--bounded_memory_far_context_mode',
        type=str,
        default='teacher_forced',
        choices=['teacher_forced', 'detached_rollout'],
        help='context used to train sampled far rollout origins',
    )
    parser.add_argument(
        '--bounded_memory_residual_bound',
        type=float,
        default=0.0,
        help='normalized tanh bound; zero leaves memory residual unbounded',
    )
    parser.add_argument(
        '--bounded_memory_sampled_far_origins',
        type=int,
        default=1,
        help='teacher-forced far rollout origins sampled beside the anchor',
    )
    parser.add_argument(
        '--bounded_memory_far_loss_weight',
        type=float,
        default=1.0,
        help='relative weight of sampled far-origin loss versus anchor loss',
    )
    parser.add_argument(
        '--bounded_memory_gradient_mode',
        type=str,
        default='joint',
        choices=[
            'joint',
            'joint_fp32',
            'anchor_pcgrad',
            'anchor_pcgrad_amp',
            'anchor_norm_pcgrad_amp',
        ],
        help='how anchor and sampled far-origin gradients are combined',
    )
    parser.add_argument('--semigroup_base_patch_len', type=int, default=12,
                        help='time unit decoded by SemigroupFlow')
    parser.add_argument('--semigroup_history_patch_len', type=int, default=24,
                        help='overlapping observed patch length in SemigroupFlow')
    parser.add_argument('--semigroup_history_stride', type=int, default=12,
                        help='observed patch stride in SemigroupFlow')
    parser.add_argument('--semigroup_state_dim', type=int, default=64,
                        help='structured latent state width in SemigroupFlow')
    parser.add_argument(
        '--semigroup_flow_mode',
        choices=[
            'diagonal',
            'rotation',
            'rotation_trend',
            'gru',
            'residual_gru',
        ],
        default='rotation_trend',
        help='stable transition family used by SemigroupFlow',
    )
    parser.add_argument(
        '--semigroup_anchor',
        choices=['last', 'mean'],
        default='last',
        help='raw last value or instance mean equilibrium for SemigroupFlow',
    )
    parser.add_argument(
        '--semigroup_loss_space',
        choices=['normalized', 'point'],
        default='normalized',
        help='training objective coordinates for SemigroupFlow',
    )
    parser.add_argument('--horizon_cross_channel_embedding',
                        dest='horizon_cross_channel_embedding',
                        action='store_true',
                        help='use a learned channel coordinate in HorizonCrossTransformer')
    parser.add_argument('--no_horizon_cross_channel_embedding',
                        dest='horizon_cross_channel_embedding',
                        action='store_false',
                        help='share one channel coordinate in HorizonCrossTransformer')
    parser.set_defaults(horizon_cross_channel_embedding=True)
    parser.add_argument('--horizon_cross_anchor', type=str,
                        default='raw_last',
                        choices=['raw_last', 'encoded_last'],
                        help='forecast-origin anchor in HorizonCrossTransformer')
    parser.add_argument(
        '--transformer_state_anchor',
        type=str,
        default='residual_last',
        choices=['encoded_last', 'residual_last'],
        help=(
            'history-state anchor in TransformerStateDirect; residual_last '
            'protects the projected final observed patch'),
    )
    parser.add_argument(
        '--transformer_state_read_gate_init',
        type=float,
        default=-2.0,
        help=(
            'negative disables state-addressed history read in '
            'TransformerStateDirect'),
    )
    parser.add_argument('--segrnn_memory_local_segments', type=int, default=4,
                        help='recent segments encoded into the protected SegRNN state')
    parser.add_argument('--segrnn_memory_scale_init', type=float, default=0.01,
                        help='initial gated Transformer-memory residual scale')
    parser.add_argument('--segrnn_memory_max_scale', type=float, default=0.0,
                        help='tanh bound for memory scale; zero is unbounded')
    parser.add_argument('--segrnn_memory_pretrained', type=str, default='',
                        help='SegRNN checkpoint used to initialize the local base')
    parser.add_argument('--segrnn_memory_freeze_base', action='store_true',
                        default=False,
                        help='freeze the pretrained local SegRNN predictor')
    parser.add_argument(
        '--addressable_state_mode',
        type=str,
        default='independent',
        choices=[
            'independent',
            'recurrent',
            'read_input',
            'read_output',
            'read_state',
            'successor_mean_output',
            'successor_output',
        ],
        help='future-state and history-read topology in AddressableStateDirect',
    )
    parser.add_argument(
        '--addressable_state_fine_patch_len',
        type=int,
        default=12,
        help='shared fine-patch length before coarse state concatenation',
    )
    parser.add_argument(
        '--addressable_state_temperature',
        type=float,
        default=8.0,
        help='initial cosine-address temperature',
    )
    parser.add_argument(
        '--addressable_state_null_margin',
        type=float,
        default=1.0,
        help='initial rejection margin of the NULL history address',
    )
    parser.add_argument(
        '--addressable_state_max_residual',
        type=float,
        default=0.25,
        help='maximum coordinate-wise history-read state correction',
    )
    parser.add_argument(
        '--addressable_state_pretrained',
        type=str,
        default='',
        help='state-only checkpoint used to initialize the read experiment',
    )
    parser.add_argument(
        '--addressable_state_freeze_base',
        action='store_true',
        default=False,
        help='freeze the pretrained state predictor and train only the read',
    )
    parser.add_argument('--state_read_mode', type=str, default='none',
                        choices=[
                            'none',
                            'read',
                            'bounded_read',
                            'analog_read',
                            'analog_linear_read',
                            'analog_phase_read',
                            'stable_analog_read',
                            'innovation_read',
                            'state_delta_read',
                            'state_context_read',
                            'linear_read',
                            'phase_linear_read',
                            'channel_phase_linear_read',
                            'no_null',
                        ],
                        help='rejectable history read for StateReadDirect')
    parser.add_argument('--state_read_null_margin', type=float, default=1.0,
                        help='initial NULL score margin over an average history address')
    parser.add_argument('--state_read_temperature', type=float, default=8.0,
                        help='initial cosine-address temperature for StateReadDirect')
    parser.add_argument('--state_read_max_residual', type=float, default=0.25,
                        help='maximum normalized point correction for bounded read')
    parser.add_argument('--state_read_history_weight', type=float, default=0.0,
                        help='weight for dense historical next-patch state supervision')
    parser.add_argument('--state_read_history_loss_space', type=str,
                        default='point', choices=['point', 'normalized'],
                        help='coordinate system for StateReadDirect history supervision')
    parser.add_argument('--state_read_encoder_patch_len', type=int, default=0,
                        help='fine Mamba address patch; zero reuses seg_len')
    parser.add_argument('--internal_read_mode', type=str, default='internal',
                        choices=[
                            'none', 'internal', 'gated', 'split',
                            'fast_weight', 'horizon_state', 'spectral_state',
                            'spectral_linear',
                        ],
                        help='addressable recurrent-state read in InternalReadSSM')
    parser.add_argument('--orthogonal_state_modes', type=int, default=16,
                        help='fixed observable time-basis modes in InternalReadSSM')
    parser.add_argument('--state_read_pretrained', type=str, default='',
                        help='state-only StateReadDirect checkpoint used before residual fitting')
    parser.add_argument('--state_read_freeze_base', action='store_true',
                        default=False,
                        help='freeze the pretrained state predictor and fit only read residual')
    parser.add_argument('--ar_mamba_use_ffn',
                        dest='ar_mamba_use_ffn',
                        action='store_true',
                        help='retain the Transformer-matched FFN after each Mamba mixer')
    parser.add_argument('--no_ar_mamba_use_ffn',
                        dest='ar_mamba_use_ffn',
                        action='store_false',
                        help='use a standard Mamba-only residual block')
    parser.set_defaults(ar_mamba_use_ffn=True)
    parser.add_argument('--delta_expand_k', type=float, default=1.0,
                        help='DeltaNet key-width expansion ratio')
    parser.add_argument('--delta_expand_v', type=float, default=1.0,
                        help='DeltaNet value-width expansion ratio')
    parser.add_argument('--delta_conv_size', type=int, default=4,
                        help='DeltaNet Q/K/V causal convolution width')
    parser.add_argument('--delta_qk_activation', type=str, default='silu',
                        choices=['silu', 'relu', 'elu', 'identity'],
                        help='DeltaNet query/key feature activation')
    parser.add_argument('--delta_qk_norm', type=str, default='l2',
                        choices=['l2', 'sum', 'none'],
                        help='DeltaNet query/key feature normalization')
    parser.add_argument('--delta_use_beta', dest='delta_use_beta',
                        action='store_true',
                        help='learn per-token DeltaNet write strengths')
    parser.add_argument('--no_delta_use_beta', dest='delta_use_beta',
                        action='store_false',
                        help='fix every DeltaNet write strength to one')
    parser.set_defaults(delta_use_beta=True)
    parser.add_argument('--delta_use_short_conv',
                        dest='delta_use_short_conv',
                        action='store_true',
                        help='apply short causal convolutions to DeltaNet Q/K/V')
    parser.add_argument('--no_delta_use_short_conv',
                        dest='delta_use_short_conv',
                        action='store_false',
                        help='disable DeltaNet short causal convolutions')
    parser.set_defaults(delta_use_short_conv=True)
    parser.add_argument('--delta_use_output_gate',
                        dest='delta_use_output_gate',
                        action='store_true',
                        help='gate normalized DeltaNet state readouts')
    parser.add_argument('--no_delta_use_output_gate',
                        dest='delta_use_output_gate',
                        action='store_false',
                        help='disable the DeltaNet output gate')
    parser.set_defaults(delta_use_output_gate=False)
    parser.add_argument('--delta_allow_negative_eigenvalues',
                        action='store_true', default=False,
                        help='allow DeltaNet beta values up to two')
    parser.add_argument('--delta_use_ffn', dest='delta_use_ffn',
                        action='store_true',
                        help='retain the FFN after each DeltaNet mixer')
    parser.add_argument('--no_delta_use_ffn', dest='delta_use_ffn',
                        action='store_false',
                        help='remove the FFN after each DeltaNet mixer')
    parser.set_defaults(delta_use_ffn=True)
    parser.add_argument(
        '--controlled_parent_mixer',
        type=str,
        default='transformer',
        choices=[
            'transformer',
            'transformer_ablation',
            'delta',
            'delta_address_gram',
            'gated_delta',
            'gated_delta_readonly_placeholder',
            'gated_delta_transformer',
            'gated_delta_query_attention',
        ],
        help=(
            'sole causal-mixer substitution in ControlledParentMixerAR; '
            'the P12 concat, placeholder, loss, and rollout stay fixed'),
    )
    parser.add_argument(
        '--controlled_state_decay_init',
        type=float,
        default=0.98,
        help='initial alpha gate for the controlled Gated-Delta update',
    )
    parser.add_argument(
        '--controlled_address_gram_ridge',
        type=float,
        default=1.0,
        help=(
            'ridge prior for the controlled address-Gram/RLS Delta state'),
    )
    parser.add_argument(
        '--terminal_latent_steps',
        type=int,
        default=3,
        help=(
            'number of causal scratch patches before the sole terminal '
            'next-patch readout'),
    )
    parser.add_argument(
        '--terminal_successor_mode',
        type=str,
        default='local',
        choices=['local', 'observed_successor'],
        help=(
            'use only the working state or add an immutable real-transition '
            'successor read at the terminal scratch state'),
    )
    parser.add_argument(
        '--terminal_successor_value_mode',
        type=str,
        default='successor',
        choices=[
            'successor',
            'same',
            'shuffled_successor',
            'permuted_successor',
            'random_permuted_successor',
            'mean_successor',
        ],
        help=(
            'store true next tokens, same-time tokens, or shuffled next '
            'tokens as a transition-memory identity control'),
    )
    parser.add_argument(
        '--terminal_successor_ridge',
        type=float,
        default=1.0,
        help='mean-risk ridge penalty for terminal successor memory',
    )
    parser.add_argument(
        '--terminal_successor_max_correction',
        type=float,
        default=0.0,
        help=(
            'smooth absolute bound on each normalized successor correction; '
            'zero keeps the unconstrained residual'),
    )
    parser.add_argument(
        '--terminal_successor_pretrained',
        type=str,
        default='',
        help='optional local terminal-model checkpoint used as stable base',
    )
    parser.add_argument(
        '--terminal_successor_freeze_base',
        action='store_true',
        default=False,
        help='train only the successor reader on the pretrained terminal base',
    )
    parser.add_argument(
        '--controlled_query_attention_gate_init',
        type=float,
        default=0.1,
        help=(
            'initial residual weight of query-only attention after '
            'Gated-Delta'),
    )
    parser.add_argument(
        '--controlled_eval_read_only_placeholder',
        action='store_true',
        default=False,
        help=(
            'train Delta placeholders with ordinary writes but make them '
            'read-only during validation and forecast rollout'),
    )
    parser.add_argument(
        '--addressable_dense_read_mode',
        choices=['softmax', 'ridge', 'ridge_mean'],
        default='softmax',
        help=(
            'read-only successor address state on the strong Delta AR base'),
    )
    parser.add_argument(
        '--addressable_dense_pretrained',
        type=str,
        default='',
        help='pure-Delta checkpoint used to initialize address-read AR',
    )
    parser.add_argument(
        '--addressable_dense_freeze_base',
        action='store_true',
        default=False,
        help='freeze the pure-Delta backbone and fit only its read state',
    )
    parser.add_argument(
        '--addressable_dense_supervision',
        choices=['dense_history', 'terminal_only'],
        default='dense_history',
        help=(
            'fit the address reader from every observed transition or only '
            'from the terminal forecast query'),
    )
    parser.add_argument(
        '--addressable_dense_rollout_train_patches',
        type=int,
        default=1,
        help=(
            'number of differentiable generated patches used to train the '
            'protected reader; one keeps dense one-step identification'),
    )
    parser.add_argument(
        '--addressable_dense_rollout_loss_weight',
        type=float,
        default=1.0,
        help='weight of multi-step closed-loop reader loss',
    )
    parser.add_argument(
        '--addressable_dense_read_only_placeholder',
        action='store_true',
        default=False,
        help=(
            'make the extra Q2 parent placeholder query each Delta cache '
            'without writing to it'),
    )
    parser.add_argument(
        '--addressable_dense_generated_second_query',
        action='store_true',
        default=False,
        help=(
            'replace the learned Q2 placeholder with the encoded first '
            'prediction before producing the second parent query'),
    )
    parser.add_argument(
        '--addressable_dense_temperature',
        type=float,
        default=8.0,
        help='initial cosine temperature of exact successor addressing',
    )
    parser.add_argument(
        '--addressable_dense_null_margin',
        type=float,
        default=1.0,
        help='initial NULL-address margin for exact successor addressing',
    )
    parser.add_argument(
        '--addressable_dense_ridge',
        type=float,
        default=1.0,
        help='ridge prior of the fixed-size successor address state',
    )
    parser.add_argument(
        '--addressable_dense_gate_init',
        type=float,
        default=0.03,
        help='initial bounded successor-read output correction',
    )
    parser.add_argument(
        '--addressable_dense_gate_max',
        type=float,
        default=0.25,
        help='maximum successor-read output correction',
    )
    parser.add_argument(
        '--addressable_dense_output_mode',
        choices=[
            'convex_successor',
            'additive_residual',
            'latent_residual',
        ],
        default='convex_successor',
        help=(
            'bounded successor interpolation or ungated conditional '
            'residual operator'),
    )
    parser.add_argument(
        '--addressable_dense_value_mode',
        choices=[
            'successor',
            'same',
            'reverse_successor',
            'permuted_successor',
            'random_permuted_successor',
            'mean_successor',
        ],
        default='successor',
        help=(
            'identity control for values paired with fixed observed keys in '
            'the address operator'),
    )
    parser.add_argument(
        '--addressable_dense_transition_shrinkage',
        type=float,
        default=1.0,
        help=(
            'shrink successor deviations toward their within-window mean; '
            'zero is a marginal-moment operator and one is the full '
            'transition operator'),
    )
    parser.add_argument(
        '--addressable_dense_learn_transition_shrinkage',
        action='store_true',
        default=False,
        help=(
            'learn the bounded transition-covariance shrinkage scalar from '
            'dense next-transition supervision'),
    )
    parser.add_argument(
        '--addressable_dense_compute_leverage',
        action='store_true',
        default=False,
        help=(
            'compute ridge leverage diagnostics during immutable-memory '
            'forecasting'),
    )
    parser.add_argument(
        '--addressable_dense_correction_scale',
        type=float,
        default=1.0,
        help=(
            'post-training multiplier on the protected transition '
            'correction; values below one damp closed-loop gain'),
    )
    parser.add_argument(
        '--addressable_dense_address_source',
        choices=[
            'state',
            'token',
            'hybrid_heads',
            'motif',
            'hybrid_motif_heads',
        ],
        default='state',
        help=(
            'derive successor addresses from recurrent states, local parent '
            'tokens, or separate state/token head groups'),
    )
    parser.add_argument(
        '--addressable_dense_motif_length',
        type=int,
        default=3,
        help=(
            'number of recent parent tokens concatenated into a local '
            'trajectory address'),
    )
    parser.add_argument(
        '--addressable_dense_tie_query_key',
        action='store_true',
        default=False,
        help=(
            'use one shared metric projection for successor queries and '
            'historical keys'),
    )
    parser.add_argument(
        '--addressable_dense_observed_only_rollout_memory',
        action='store_true',
        default=False,
        help=(
            'during AR rollout, prevent generated patches from becoming '
            'successor-memory key/value writes'),
    )
    parser.add_argument(
        '--addressable_dense_anchor_observed_memory',
        action='store_true',
        default=False,
        help=(
            'keep the original real-history successor bank immutable across '
            'fixed-window AR rollout and re-encode it under current stats'),
    )
    parser.add_argument(
        '--addressable_dense_eval_commit_patches',
        type=int,
        choices=[0, 1, 2],
        default=0,
        help=(
            'evaluation-only commit width; zero uses the trained Q width'),
    )
    parser.add_argument(
        '--addressable_dense_eval_r2_steps',
        type=int,
        default=-1,
        help=(
            'evaluation-only stopping policy: commit Q2 for this many '
            'outer steps, then R1; negative disables the policy'),
    )
    parser.add_argument(
        '--addressable_dense_phase_periods',
        type=float,
        nargs='*',
        default=[],
        help=(
            'optional parent-token periods for deterministic rotary address '
            'binding; an empty list disables phase binding'),
    )
    parser.add_argument('--ar_attention_mode', type=str, default='causal',
                        choices=['causal', 'bidirectional', 'block_causal'],
                        help='TailAR attention visibility; DenseAR remains causal')
    parser.add_argument('--conv_tail_kernel_size', type=int, default=2,
                        help='causal projected-patch convolution kernel for ConvTailAR')
    parser.add_argument('--value_mixer_hidden_multiplier', type=int, default=1,
                        help='per-head hidden multiplier for ValueMixTailAR composition')
    parser.add_argument('--swin_tail_window_size', type=int, default=4,
                        help='local patch-token window for SwinTailAR stages')
    parser.add_argument('--swin_tail_shift_size', type=int, default=0,
                        help='history-window shift in merged SwinTailAR states')
    parser.add_argument('--swin_tail_history_prepass', action='store_true',
                        default=False,
                        help='update merged history before forecast queries read it')
    parser.add_argument('--swin_tail_stage1_dim', type=int, default=32,
                        help='fine p12 hidden width before SwinTailAR patch merge')
    parser.add_argument('--swin_tail_stage1_heads', type=int, default=4,
                        help='attention heads in the fine SwinTailAR stage')
    parser.add_argument('--split_concat_fine_patch_len', type=int, default=12,
                        help='fine patch projected before adjacent concatenation')
    parser.add_argument('--sliding_window_fine_patch_len', type=int, default=12,
                        help='fine-patch stride of overlapping AR state windows')
    parser.add_argument('--sliding_window_projection', type=str,
                        default='whole', choices=['whole', 'slots'],
                        help='project each complete window or its fine-patch slots')
    parser.add_argument('--progressive_concat_fine_patch_len', type=int,
                        default=12,
                        help='shared fine-patch input length in ProgressiveConcatAR')
    parser.add_argument('--progressive_concat_sizes', type=int, nargs='+',
                        default=[6, 4, 2],
                        help='one right-aligned concat window per Transformer layer')
    parser.add_argument('--progressive_concat_stride', type=int, default=2,
                        help='common fine-patch stride for every concat scale')
    parser.add_argument('--progressive_concat_max_patches', type=int, default=6,
                        help='number of aligned slots in the fixed concat canvas')
    parser.add_argument('--ssm_fuse_fine_patch_len', type=int, default=12,
                        help='fine patch length for segmented SSM fusion')
    parser.add_argument('--finite_ssm_memory_patches', type=int, default=4,
                        help='strict maximum p12 memory in FiniteSSMDenseAR')
    parser.add_argument('--finite_ssm_fine_patch_len', type=int, default=12,
                        help='encoder patch length for AR24 finite-memory SSM')
    parser.add_argument('--finite_ssm_length_mode', type=str,
                        default='adaptive', choices=['adaptive', 'full'],
                        help='adaptively select or always retain the complete finite window')
    parser.add_argument('--latent_query_states', type=int, default=4,
                        help='learned future state queries per LatentQueryAR block')
    parser.add_argument('--latent_history_state_weight', type=float, default=0.5,
                        help='weight of adjacent history-state transition loss')
    parser.add_argument('--latent_future_state_weight', type=float, default=1.0,
                        help='weight of learned-query future-state loss')
    parser.add_argument('--latent_point_weight', type=float, default=1.0,
                        help='weight of decoded future-point loss')
    parser.add_argument('--adaptive_fine_patch_len', type=int, default=12,
                        help='fine patch length for AdaptiveTailAR (half of ar_patch_len)')
    parser.add_argument('--adaptive_router_bias_rate', type=float, default=0.01,
                        help='Kairos-style router load-balancing bias update rate')
    parser.add_argument('--adaptive_router_target_dist', type=float, nargs=3,
                        default=[0.2, 0.6, 0.2],
                        metavar=('COARSE', 'FINE', 'NULL'),
                        help='target router mass for coarse, fine, and null experts')
    parser.add_argument('--entropy_model_checkpoint', type=str,
                        default='assets/entrope/ETTh1_entropy_gpt.pt',
                        help='frozen EntroPE conditional-entropy model checkpoint')
    parser.add_argument('--entropy_patch_quantile', type=float, default=0.95,
                        help='per-instance conditional-entropy boundary quantile')
    parser.add_argument('--entropy_max_patch_len', type=int, default=32,
                        help='maximum EntroPE history patch length')
    parser.add_argument('--entropy_encoder_type', type=str, default='ape',
                        choices=['ape', 'resample'],
                        help='encode entropy segments with APE or ordered resampling')
    parser.add_argument('--entropy_quant_low', type=float,
                        default=-2.169745445251465,
                        help='lower ETTh1 tokenizer center from the training split')
    parser.add_argument('--entropy_quant_high', type=float,
                        default=1.8946701288223267,
                        help='upper ETTh1 tokenizer center from the training split')
    parser.add_argument('--entropy_monotonic', action='store_true', default=False,
                        help='place boundaries on entropy increases instead of high entropy')

    # GCN
    parser.add_argument('--node_dim', type=int, default=10, help='each node embbed to dim dimentions')
    parser.add_argument('--gcn_depth', type=int, default=2, help='')
    parser.add_argument('--gcn_dropout', type=float, default=0.3, help='')
    parser.add_argument('--propalpha', type=float, default=0.3, help='')
    parser.add_argument('--conv_channel', type=int, default=32, help='')
    parser.add_argument('--skip_channel', type=int, default=32, help='')

    parser.add_argument('--individual', action='store_true', default=False,
                        help='DLinear: a linear layer for each variate(channel) individually')

    # TimeFilter
    parser.add_argument('--alpha', type=float, default=0.1, help='KNN for Graph Construction')
    parser.add_argument('--top_p', type=float, default=0.5, help='Dynamic Routing in MoE')
    parser.add_argument('--pos', type=int, choices=[0, 1], default=1, help='Positional Embedding. Set pos to 0 or 1')

    args = parser.parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    if args.data_scale is None:
        # DenseAR and TailAR consume raw windows by default. Their evaluator can
        # still use train-only scaler statistics for benchmark-compatible metrics.
        args.data_scale = args.model not in {
            'DenseAR', 'TailAR', 'HybridTailAR', 'StateReconTailAR',
            'AdaptiveTailAR', 'EntropyTailAR',
            'ConvTailAR', 'ValueMixTailAR', 'SwinTailAR',
            'SplitConcatTailAR', 'SplitConcatDenseAR',
            'SlidingWindowDenseAR', 'ProgressiveConcatAR',
            'SSMFuseTailAR', 'SSMFuseDenseAR',
            'FiniteSSMDenseAR', 'FiniteSSM24DenseAR', 'LatentQueryAR',
            'TransformerAblationAR', 'PredictiveStateResidualAR',
            'AdaptivePatchFusionAR',
            'DistilledMultiCommitAR', 'LatentClosureMultiCommitAR',
            'BehavioralClosureMultiCommitAR', 'RebasedMultiCommitAR',
            'HistoryReadMultiCommitAR',
            'CalibratedHistoryReadMultiCommitAR',
            'ProgressiveHistoryReadMultiCommitAR',
            'CleanTargetOnPolicyHistoryReadAR',
            'CalibratedHistoryBehavioralCocycleAR',
            'ShrunkCleanTargetHistoryReadAR',
            'ReliabilityGatedCleanTargetHistoryReadAR',
            'ProtectedPushforwardHistoryReadAR',
            'ResidualHistoryReadMultiCommitAR',
            'InternalCoarseGrainAR', 'ProgressiveNestedCoarseGrainAR',
            'ProtectedPushforwardCoarseGrainAR',
            'ContinuationCorrectedPPMC',
            'ProtectedBehavioralCocycleAR',
            'ConditionalMemoryInnovationPPMC',
            'QueryTrajectoryBehavioralCocycleAR',
            'StrongParentDyadicDistillationAR',
            'ProtectedResidualStateSpaceAR',
            'ScaleConsistentBudgetedCorrectionAR',
            'ParallelHistoryBudgetedCorrectionAR',
            'ParallelScanBehavioralCocycleAR',
            'EndogenousStatStateAR',
            'MomentClosureAR', 'ReferenceQ1ContinuationAR',
            'MambaDenseAR',
            'SplitConcatMambaAR', 'RetrievalMambaAR', 'AddressableMambaAR',
            'FineMambaCoarseAR', 'LaggedMambaAR',
            'LearnedPatchAR', 'GRUSlotAR', 'ProgressiveDirectAR',
            'OverlapDirectAR', 'HistoryOnlyDirectAR',
            'BoundedMemoryDirectAR', 'SlidingStateDirectAR',
            'PersistentSuccessorAR', 'BlockRecurrentAR',
            'RecurrentQueryDirectAR',
            'SemigroupFlow', 'TerminalLatentSuccessorAR'}
    if torch.cuda.is_available() and args.use_gpu:
        args.device = torch.device('cuda:{}'.format(args.gpu))
        print('Using GPU')
    else:
        if hasattr(torch.backends, "mps"):
            args.device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
        else:
            args.device = torch.device("cpu")
        print('Using cpu or mps')

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    print('Args in experiment:')
    if args.checkpoint_selection == 'test_oracle':
        print('WARNING: test_oracle checkpoint selection leaks test data; '
              'report results as diagnostic upper bounds only.')
    print_args(args)


    if args.task_name == 'long_term_forecast':
        from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
        Exp = Exp_Long_Term_Forecast
    elif args.task_name == 'short_term_forecast':
        from exp.exp_short_term_forecasting import Exp_Short_Term_Forecast
        Exp = Exp_Short_Term_Forecast
    elif args.task_name == 'imputation':
        from exp.exp_imputation import Exp_Imputation
        Exp = Exp_Imputation
    elif args.task_name == 'anomaly_detection':
        from exp.exp_anomaly_detection import Exp_Anomaly_Detection
        Exp = Exp_Anomaly_Detection
    elif args.task_name == 'classification':
        from exp.exp_classification import Exp_Classification
        Exp = Exp_Classification
    elif args.task_name == 'zero_shot_forecast':
        from exp.exp_zero_shot_forecasting import Exp_Zero_Shot_Forecast
        Exp = Exp_Zero_Shot_Forecast
    else:
        from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
        Exp = Exp_Long_Term_Forecast

    if args.is_training:
        for ii in range(args.itr):
            # setting record of experiments
            exp = Exp(args)  # set experiments
            setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_expand{}_dc{}_fc{}_eb{}_dt{}_{}_{}'.format(
                args.task_name,
                args.model_id,
                args.model,
                args.data,
                args.features,
                args.seq_len,
                args.label_len,
                args.pred_len,
                args.d_model,
                args.n_heads,
                args.e_layers,
                args.d_layers,
                args.d_ff,
                args.expand,
                args.d_conv,
                args.factor,
                args.embed,
                args.distil,
                args.des, ii)
            
            # Override setting for specific model to ensure proper checkpoint naming and logging
            if args.model == 'MambaSingleLayer' and args.task_name == 'classification':
                setting = f'{args.task_name}_CLS_{args.model_id}_{args.model}_{args.data}_ft{args.features}' \
                        + f'_sl{args.seq_len}_ll{args.label_len}_pl{args.pred_len}_dm{args.d_model}_ds{args.d_ff}' \
                        + f'_expand{args.expand}_dc{args.d_conv}_nk{args.num_kernels}' \
                        + f'_tvdt{int(args.tv_dt)}_tvB{int(args.tv_B)}_tvC{int(args.tv_C)}_useD{int(args.use_D)}_{args.des}_{ii}'
            elif args.model == 'DenseAR':
                setting += f'_arp{args.ar_patch_len}_dense{args.dense_ar_roll_patches}' \
                           + f'_tail{args.tail_ar_patches}' \
                           + f'_ls{args.dense_ar_loss_space}' \
                           + f'_norm{args.ar_transformer_norm}' \
                           + f'_sep{int(args.tail_separate_projection)}' \
                           + f'_datascale{int(args.data_scale)}_s{args.seed}'
            elif args.model == 'TransformerAblationAR':
                setting = f'ltf_TAB_{args.model_id}_{args.data}' \
                          + f'_sl{args.seq_len}_pl{args.pred_len}' \
                          + f'_p{args.ar_patch_len}_dm{args.d_model}' \
                          + f'h{args.n_heads}l{args.e_layers}f{args.d_ff}' \
                          + f'_q{args.dense_ar_roll_patches}' \
                          + f'c{args.dense_ar_eval_commit_patches}' \
                          + f'_a{args.ablation_attention}' \
                          + (
                              f'{args.ablation_shrinkage_type}'
                              f'{args.ablation_shrinkage_alpha:g}'
                              f'dh{args.ablation_dynamic_heads}'
                              if args.ablation_attention == 'shrinkage'
                              else ''
                          ) \
                          + f'_f{args.ablation_ffn}_n{args.ablation_norm}' \
                          + f'_p{args.ablation_position_encoding}' \
                          + f'_r{int(args.ablation_attention_residual)}' \
                          + f'{int(args.ablation_ffn_residual)}' \
                          + f'_as{args.ablation_patch_assembly}' \
                          + f'{args.ablation_patch_assembly_fine_len}' \
                          + f'{args.ablation_patch_assembly_stem}' \
                          + (
                              f'a{args.ablation_patch_assembly_atom_dim}'
                              f'b{args.ablation_patch_assembly_basis_dim}'
                              if args.ablation_patch_assembly == 'factorized'
                              else ''
                          ) \
                          + (
                              f'w{args.ablation_patch_assembly_fd_widths}'
                              if args.ablation_patch_assembly == 'fd'
                              else ''
                          ) \
                          + f'_oh{args.ablation_output_head}' \
                          + f'{args.ablation_output_fine_patch_len}' \
                          + (
                              f'b{args.ablation_output_basis_dim}'
                              if args.ablation_output_head
                              == 'factorized_atoms'
                              else ''
                          ) \
                          + f'_e{args.ar_norm_eps:g}' \
                          + f'_ds{int(args.data_scale)}_s{args.seed}'
            elif args.model == 'TailAR':
                setting += f'_arp{args.ar_patch_len}_tail{args.tail_ar_patches}' \
                           + f'_roll{args.tail_ar_roll_patches}' \
                           + f'_ph{args.tail_placeholder_type}' \
                           + f'_norm{args.ar_transformer_norm}' \
                           + f'_sep{int(args.tail_separate_projection)}' \
                           + f'_ih{int(args.tail_independent_heads)}_datascale{int(args.data_scale)}'
            elif args.model == 'HybridTailAR':
                setting += f'_arp{args.ar_patch_len}_tail{args.tail_ar_patches}' \
                           + f'_roll{args.tail_ar_roll_patches}' \
                           + f'_ph{args.tail_placeholder_type}' \
                           + f'_norm{args.ar_transformer_norm}' \
                           + f'_hw{args.tail_history_loss_weight:g}' \
                           + f'_sep{int(args.tail_separate_projection)}' \
                           + f'_ih{int(args.tail_independent_heads)}_datascale{int(args.data_scale)}'
            elif args.model == 'LearnedPatchAR':
                setting += f'_arp{args.ar_patch_len}' \
                           + f'_tail{args.tail_ar_patches}' \
                           + f'_roll{args.tail_ar_roll_patches}' \
                           + f'_ph{args.tail_placeholder_type}' \
                           + f'_norm{args.ar_transformer_norm}' \
                           + f'_eps{args.ar_norm_eps:g}' \
                           + f'_datascale{int(args.data_scale)}_s{args.seed}'
            elif args.model == 'GRUSlotAR':
                setting += f'_arp{args.ar_patch_len}' \
                           + f'_g{args.ar_gru_slots}' \
                           + f'_roll{args.dense_ar_roll_patches}' \
                           + f'_ls{args.dense_ar_loss_space}' \
                           + f'_norm{args.ar_transformer_norm}' \
                           + f'_eps{args.ar_norm_eps:g}' \
                           + f'_datascale{int(args.data_scale)}_s{args.seed}'
            elif args.model == 'AddressableDeltaAR':
                setting = f'ltf_ADAR_{args.dataset_tag or args.data}' \
                          + f'_w{args.seq_len}h{args.pred_len}' \
                          + f'_p{args.ar_patch_len}' \
                          + f'f{args.split_concat_fine_patch_len}' \
                          + f'_k{args.dense_ar_roll_patches}' \
                          + (
                              f'_q{args.addressable_delta_train_patches}'
                              if args.addressable_delta_train_patches
                              not in {
                                  0, args.dense_ar_roll_patches
                              } else ''
                          ) \
                          + f'_{args.addressable_delta_mode}' \
                          + f'_ak{args.addressable_delta_key_dim}' \
                          + f'_om{args.addressable_delta_output_max_scale:g}' \
                          + f'_sm{args.addressable_delta_state_max_scale:g}' \
                          + f'_gi{args.addressable_delta_gate_init:g}' \
                          + f'_b{args.addressable_delta_beta_init:g}' \
                          + f'_{args.dense_ar_loss_type}' \
                          + (
                              f'{args.dense_ar_huber_delta:g}'
                              if args.dense_ar_loss_type == 'huber' else ''
                          ) \
                          + (
                              f'_rm{args.addressable_delta_read_multiplier:g}'
                              if args.addressable_delta_read_multiplier != 1
                              else ''
                          ) \
                          + f'_d{args.d_model}h{args.n_heads}' \
                          + f'l{args.e_layers}f{args.d_ff}' \
                          + f'_n{args.ar_transformer_norm}' \
                          + f'_ds{int(args.data_scale)}' \
                          + f'_loader{args.loader_seed}_s{args.seed}'
            elif args.model == 'DeltaPlaceholderAR':
                setting = f'ltf_DPAR_{args.dataset_tag or args.data}' \
                          + f'_w{args.seq_len}h{args.pred_len}' \
                          + f'_p{args.ar_patch_len}' \
                          + f'f{args.delta_placeholder_fine_patch_len}' \
                          + f'_{args.delta_placeholder_history_fusion}' \
                          + f'_{args.delta_placeholder_attention_scope}' \
                          + (
                              f'_afgres'
                              f'{args.delta_placeholder_attention_gate_init:g}'
                              f'm{args.delta_placeholder_attention_gate_max:g}'
                              if args.delta_placeholder_attention_fusion
                              == 'gated_residual' else ''
                          ) \
                          + f'_tl{args.delta_placeholder_transformer_layers}' \
                          + f'_hw{args.delta_placeholder_history_weight:g}' \
                          + f'_{args.dense_ar_loss_type}' \
                          + (
                              f'{args.dense_ar_huber_delta:g}'
                              if args.dense_ar_loss_type == 'huber' else ''
                          ) \
                          + f'_d{args.d_model}h{args.n_heads}' \
                          + f'l{args.e_layers}f{args.d_ff}' \
                          + f'_n{args.ar_transformer_norm}' \
                          + f'_ds{int(args.data_scale)}' \
                          + f'_loader{args.loader_seed}_s{args.seed}'
            elif args.model == 'DeltaBlockRolloutAR':
                delta_rollout_queries = (
                    1 + args.delta_block_rollout_placeholder_blocks)
                delta_rollout_commits = (
                    args.delta_block_rollout_commit_blocks
                    or delta_rollout_queries)
                setting = f'ltf_DBRA_{args.dataset_tag or args.data}' \
                          + f'_w{args.seq_len}h{args.pred_len}' \
                          + f'_p{args.ar_patch_len}' \
                          + f'f{args.delta_placeholder_fine_patch_len}' \
                          + f'_q{delta_rollout_queries}' \
                          + f'r{delta_rollout_commits}' \
                          + (
                              f'_t{args.delta_block_rollout_train_target_blocks}'
                              if args.delta_block_rollout_train_target_blocks
                              else ''
                          ) \
                          + (
                              f'_a{args.delta_placeholder_attention_scope}'
                              f'tl{args.delta_placeholder_transformer_layers}'
                              if args.delta_placeholder_attention_scope
                              != 'none' else ''
                          ) \
                          + (
                              f'_hf{args.delta_placeholder_history_fusion}'
                              if args.delta_placeholder_attention_scope
                              == 'all'
                              and args.delta_placeholder_history_fusion
                              != 'delta_boundary' else ''
                          ) \
                          + (
                              f'_afgres'
                              f'{args.delta_placeholder_attention_gate_init:g}'
                              f'm{args.delta_placeholder_attention_gate_max:g}'
                              if args.delta_placeholder_attention_scope
                              != 'none'
                              and args.delta_placeholder_attention_fusion
                              == 'gated_residual' else ''
                          ) \
                          + (
                              f'_cw'
                              f'{args.delta_block_rollout_second_block_weight:g}'
                              if args.delta_block_rollout_second_block_weight
                              != 1 else ''
                          ) \
                          + (
                              f'_sc'
                              f'{args.delta_block_rollout_state_consistency_weight:g}'
                              if args.delta_block_rollout_state_consistency_weight
                              else ''
                          ) \
                          + (
                              '_cq'
                              if args.delta_block_rollout_conditioned_placeholder
                              else ''
                          ) \
                          + (
                              '_pf'
                              if args.delta_block_rollout_predicted_feedback
                              else ''
                          ) \
                          + (
                              f'_fb'
                              f'{args.delta_block_rollout_state_feedback_bound:g}'
                              if args.delta_block_rollout_state_feedback_bound
                              else ''
                          ) \
                          + (
                              '_ch'
                              if args.delta_block_rollout_independent_closure_head
                              else ''
                          ) \
                          + (
                              '_iso'
                              if args.delta_block_rollout_isolate_closure_gradient
                              else ''
                          ) \
                          + (
                              '_asel'
                              if args.delta_block_rollout_anchor_selection
                              else ''
                          ) \
                          + (
                              '_pc'
                              if args.delta_block_rollout_project_closure_gradient
                              else ''
                          ) \
                          + (
                              f'_oc'
                              f'{args.delta_block_rollout_predictive_consistency_weight:g}'
                              f's{args.delta_block_rollout_predictive_consistency_steps}'
                              if args.delta_block_rollout_predictive_consistency_weight
                              else ''
                          ) \
                          + (
                              f'_alt'
                              f'{args.delta_block_rollout_alternate_closure_period}'
                              if args.delta_block_rollout_alternate_closure_period
                              else ''
                          ) \
                          + f'_hw{args.delta_placeholder_history_weight:g}' \
                          + f'_{args.dense_ar_loss_type}' \
                          + (
                              f'{args.dense_ar_huber_delta:g}'
                              if args.dense_ar_loss_type == 'huber' else ''
                          ) \
                          + f'_d{args.d_model}h{args.n_heads}' \
                          + f'l{args.e_layers}f{args.d_ff}' \
                          + f'_n{args.ar_transformer_norm}' \
                          + f'_ds{int(args.data_scale)}' \
                          + f'_loader{args.loader_seed}_s{args.seed}'
            elif args.model == 'RecurrentQueryDirectAR':
                setting = f'ltf_RQD_{args.dataset_tag or args.data}' \
                          + f'_w{args.seq_len}h{args.pred_len}' \
                          + f'_p{args.ar_patch_len}' \
                          + f'f{args.split_concat_fine_patch_len}' \
                          + f'_k{args.dense_ar_roll_patches}' \
                          + f'_r{args.recurrent_query_steps}' \
                          + f'_{args.recurrent_query_position_mode}' \
                          + (
                              f'_{args.recurrent_query_read_mode}'
                              if args.recurrent_query_read_mode != 'dynamic'
                              else ''
                          ) \
                          + (
                              f'_{args.recurrent_query_output_mode}'
                              if args.recurrent_query_output_mode != 'last'
                              else ''
                          ) \
                          + f'_t{args.recurrent_query_temperature:g}' \
                          + f'_d{args.d_model}h{args.n_heads}' \
                          + f'l{args.e_layers}f{args.d_ff}' \
                          + f'_n{args.ar_transformer_norm}' \
                          + f'_ds{int(args.data_scale)}' \
                          + f'_loader{args.loader_seed}_s{args.seed}'
            elif args.model == 'ProgressiveDirectAR':
                roll_schedule = '-'.join(
                    str(value) for value in args.direct_roll_schedule)
                setting = f'ltf_PDAR_{args.data}' \
                          + f'_w{args.seq_len}h{args.pred_len}' \
                          + f'_p{args.ar_patch_len}q{args.tail_ar_patches}' \
                          + f'_r{roll_schedule}' \
                          + f'_d{args.d_model}l{args.e_layers}' \
                          + f'_n{args.ar_transformer_norm}' \
                          + f'_e{args.ar_norm_eps:g}_s{args.seed}'
            elif args.model == 'OverlapDirectAR':
                setting = f'ltf_ODAR_{args.data}' \
                          + f'_w{args.seq_len}h{args.pred_len}' \
                          + f'_p{args.ar_patch_len}' \
                          + f's{args.direct_history_patch_stride}' \
                          + f'q{args.tail_ar_patches}' \
                          + f'_pe{args.direct_position_encoding}' \
                          + f'_d{args.d_model}h{args.n_heads}' \
                          + f'l{args.e_layers}f{args.d_ff}' \
                          + f'_n{args.ar_transformer_norm}' \
                          + f'_ds{int(args.data_scale)}' \
                          + f'_loader{args.loader_seed}_s{args.seed}'
            elif args.model == 'HistoryOnlyDirectAR':
                setting = f'ltf_HOD_{args.data}' \
                          + f'_w{args.seq_len}h{args.pred_len}' \
                          + f'_f{args.history_only_fine_patch_len}' \
                          + f'g{args.history_only_group_patches}' \
                          + f'_pm{args.history_only_position_mode}' \
                          + f'_d{args.d_model}h{args.n_heads}' \
                          + f'l{args.e_layers}f{args.d_ff}' \
                          + f'_n{args.ar_transformer_norm}' \
                          + f'_ds{int(args.data_scale)}' \
                          + f'_loader{args.loader_seed}_s{args.seed}'
            elif args.model == 'BoundedMemoryDirectAR':
                setting = f'ltf_BMD_{args.dataset_tag or args.data}' \
                          + f'_w{args.seq_len}h{args.pred_len}' \
                          + f'_f{args.history_only_fine_patch_len}' \
                          + f'g{args.history_only_group_patches}' \
                          + f'_k{args.tail_ar_patches}' \
                          + f'_{args.bounded_memory_mode}' \
                          + (
                              f'_mg{args.bounded_memory_granularity}'
                              if args.bounded_memory_granularity != 'coarse'
                              else ''
                          ) \
                          + f'_{args.bounded_memory_stats_mode}' \
                          + (
                              f'_pm{args.bounded_memory_position_mode}'
                              if args.bounded_memory_position_mode != 'none'
                              else ''
                          ) \
                          + (
                              f'_fm{args.bounded_memory_fusion_mode}'
                              if args.bounded_memory_fusion_mode
                              != 'joint_attention'
                              else ''
                          ) \
                          + (
                              f'_fc{args.bounded_memory_far_context_mode}'
                              if args.bounded_memory_far_context_mode
                              != 'teacher_forced'
                              else ''
                          ) \
                          + (
                              '_rb'
                              + str(args.bounded_memory_residual_bound).replace('.', 'p')
                              if args.bounded_memory_residual_bound > 0
                              else ''
                          ) \
                          + f'_fo{args.bounded_memory_sampled_far_origins}' \
                          + (
                              '_fw'
                              + str(args.bounded_memory_far_loss_weight).replace('.', 'p')
                              if args.bounded_memory_far_loss_weight != 1.0
                              else ''
                          ) \
                          + (
                              f'_gm{args.bounded_memory_gradient_mode}'
                              if args.bounded_memory_gradient_mode != 'joint'
                              else ''
                          ) \
                          + f'_d{args.d_model}h{args.n_heads}' \
                          + f'l{args.e_layers}f{args.d_ff}' \
                          + f'_n{args.ar_transformer_norm}' \
                          + f'_ds{int(args.data_scale)}' \
                          + f'_loader{args.loader_seed}_s{args.seed}'
            elif args.model == 'SemigroupFlow':
                setting = f'ltf_SGF_{args.dataset_tag or args.data}' \
                          + f'_w{args.seq_len}h{args.pred_len}' \
                          + f'_b{args.semigroup_base_patch_len}' \
                          + f'_hp{args.semigroup_history_patch_len}' \
                          + f's{args.semigroup_history_stride}' \
                          + f'_{args.semigroup_flow_mode}' \
                          + (
                              f'_a{args.semigroup_anchor}'
                              if args.semigroup_anchor != 'last' else ''
                          ) \
                          + (
                              f'_ls{args.semigroup_loss_space}'
                              if args.semigroup_loss_space != 'normalized'
                              else ''
                          ) \
                          + f'_z{args.semigroup_state_dim}' \
                          + f'_d{args.d_model}h{args.n_heads}' \
                          + f'l{args.e_layers}f{args.d_ff}' \
                          + f'_ds{int(args.data_scale)}' \
                          + f'_loader{args.loader_seed}_s{args.seed}'
            elif args.model == 'AddressableStateDirect':
                setting = f'ltf_ASD_{args.data}' \
                          + f'_w{args.seq_len}h{args.pred_len}' \
                          + f'_p{args.seg_len}' \
                          + f'f{args.addressable_state_fine_patch_len}' \
                          + f'_{args.addressable_state_mode}' \
                          + f'_d{args.d_model}h{args.n_heads}' \
                          + f'_t{args.addressable_state_temperature:g}' \
                          + f'_nm{args.addressable_state_null_margin:g}' \
                          + f'_mr{args.addressable_state_max_residual:g}' \
                          + f'_ce{int(args.horizon_cross_channel_embedding)}' \
                          + f'_ds{int(args.data_scale)}' \
                          + (
                              '_fr1'
                              if args.addressable_state_freeze_base else ''
                          ) \
                          + f'_loader{args.loader_seed}_s{args.seed}'
            elif args.model == 'PersistentSuccessorAR':
                setting = f'ltf_PSAR_{args.dataset_tag or args.data}' \
                          + f'_w{args.seq_len}h{args.pred_len}' \
                          + f'_p{args.ar_patch_len}' \
                          + f'k{args.tail_ar_patches}' \
                          + f'r{args.tail_ar_roll_patches}' \
                          + f'_{args.persistent_successor_mode}' \
                          + f'_hw{args.persistent_successor_history_weight:g}' \
                          + f'_lm{args.persistent_successor_loss_mode}' \
                          + f'_fw{args.persistent_successor_far_weight:g}' \
                          + f'_gg{args.persistent_successor_gradient_group}' \
                          + (
                              f'_sc{args.persistent_successor_far_sample_count}'
                              if args.persistent_successor_loss_mode
                              == 'sampled_far' else ''
                          ) \
                          + f'_rs{args.persistent_successor_residual_scale:g}' \
                          + (
                              f'_rx{args.persistent_successor_residual_max_scale:g}'
                              if args.persistent_successor_residual_max_scale
                              is not None else ''
                          ) \
                          + (
                              f'_rg{args.persistent_successor_residual_gate}'
                              if args.persistent_successor_residual_gate
                              != 'real_fraction' else ''
                          ) \
                          + (
                              f'_sm{args.persistent_successor_stats_mode}'
                              if args.persistent_successor_stats_mode
                              != 'rolling' else ''
                          ) \
                          + (
                              '_ih1'
                              if args.tail_independent_heads else ''
                          ) \
                          + (
                              f'_rr{args.persistent_successor_retained_real_patches}'
                              if args.persistent_successor_retained_real_patches
                              > 0 else ''
                          ) \
                          + f'_t{args.persistent_successor_temperature:g}' \
                          + f'_nm{args.persistent_successor_null_margin:g}' \
                          + f'_d{args.d_model}h{args.n_heads}' \
                          + f'l{args.e_layers}f{args.d_ff}' \
                          + f'_ds{int(args.data_scale)}' \
                          + f'_loader{args.loader_seed}_s{args.seed}'
            elif args.model == 'SlidingStateDirectAR':
                setting = f'ltf_SSD_{args.data}' \
                          + f'_w{args.seq_len}h{args.pred_len}' \
                          + f'_f{args.recon_tail_fine_patch_len}' \
                          + f'g{args.recon_tail_group_patches}' \
                          + f'q{args.recon_tail_future_points}' \
                          + f'_pe{args.direct_position_encoding}' \
                          + f'_d{args.d_model}h{args.n_heads}' \
                          + f'l{args.e_layers}f{args.d_ff}' \
                          + f'_n{args.ar_transformer_norm}' \
                          + f'_ds{int(args.data_scale)}' \
                          + f'_loader{args.loader_seed}_s{args.seed}'
            elif args.model == 'HorizonCrossTransformer':
                setting = f'ltf_HCT_{args.data}' \
                          + f'_w{args.seq_len}h{args.pred_len}' \
                          + f'_p{args.seg_len}_d{args.d_model}' \
                          + f'l{args.e_layers}f{args.d_ff}' \
                          + f'_a{args.horizon_cross_anchor}' \
                          + f'_ce{int(args.horizon_cross_channel_embedding)}' \
                          + f'_s{args.seed}'
            elif args.model == 'TransformerStateDirect':
                setting = f'ltf_TSD_{args.model_id}_{args.data}' \
                          + f'_w{args.seq_len}h{args.pred_len}' \
                          + f'_p{args.seg_len}' \
                          + f'_d{args.d_model}h{args.n_heads}' \
                          + f'l{args.e_layers}f{args.d_ff}' \
                          + f'_a{args.transformer_state_anchor}' \
                          + (
                              f'_rg{args.transformer_state_read_gate_init:g}'
                              if args.transformer_state_read_gate_init > -1
                              else ''
                          ) \
                          + f'_ce{int(args.horizon_cross_channel_embedding)}' \
                          + f'_ds{int(args.data_scale)}' \
                          + f'_loader{args.loader_seed}_s{args.seed}'
            elif args.model == 'SegRNNMemoryTransformer':
                setting = f'ltf_SRMT_{args.data}' \
                          + f'_w{args.seq_len}h{args.pred_len}' \
                          + f'_p{args.seg_len}_d{args.d_model}' \
                          + f'l{args.e_layers}f{args.d_ff}' \
                          + f'_local{args.segrnn_memory_local_segments}' \
                          + f'_mi{args.segrnn_memory_scale_init:g}' \
                          + f'_mx{args.segrnn_memory_max_scale:g}' \
                          + f'_fr{int(args.segrnn_memory_freeze_base)}' \
                          + f'_ce{int(args.horizon_cross_channel_embedding)}' \
                          + f'_s{args.seed}'
            elif args.model == 'StateReadDirect':
                setting = f'ltf_SRD_{args.model_id}' \
                          + f'_w{args.seq_len}h{args.pred_len}' \
                          + f'_p{args.seg_len}' \
                          + (
                              f'_ep{args.state_read_encoder_patch_len}'
                              if args.state_read_encoder_patch_len not in {
                                  0, args.seg_len
                              }
                              else ''
                          ) \
                          + f'_d{args.d_model}h{args.n_heads}' \
                          + f'l{args.e_layers}f{args.d_ff}' \
                          + f'_m{args.ar_mamba_d_state}' \
                          + f'x{args.expand}c{args.d_conv}' \
                          + f'_ff{int(args.ar_mamba_use_ffn)}' \
                          + f'_{args.state_read_mode}' \
                          + f'_nm{args.state_read_null_margin:g}' \
                          + f'_t{args.state_read_temperature:g}' \
                          + (
                              f'_fr{int(args.state_read_freeze_base)}'
                              f'_c{args.state_read_max_residual:g}'
                              if args.state_read_freeze_base
                              or args.state_read_mode in {
                                  'bounded_read',
                                  'analog_read',
                                  'analog_linear_read',
                                  'analog_phase_read',
                                  'stable_analog_read',
                                  'innovation_read',
                                  'state_delta_read',
                                  'state_context_read',
                                  'linear_read',
                                  'phase_linear_read',
                                  'channel_phase_linear_read',
                              }
                              else ''
                          ) \
                          + (
                              f'_hw{args.state_read_history_weight:g}'
                              f'_hl{args.state_read_history_loss_space}'
                              if args.state_read_history_weight > 0
                              else ''
                          ) \
                          + f'_ds{int(args.data_scale)}' \
                          + f'_loader{args.loader_seed}_s{args.seed}'
            elif args.model == 'InternalReadSSM':
                setting = f'ltf_IRS_{args.model_id}' \
                          + f'_w{args.seq_len}h{args.pred_len}' \
                          + f'_p{args.seg_len}' \
                          + f'_d{args.d_model}l{args.e_layers}' \
                          + f'f{args.d_ff}_m{args.ar_mamba_d_state}' \
                          + f'x{args.expand}c{args.d_conv}' \
                          + f'_ff{int(args.ar_mamba_use_ffn)}' \
                          + f'_{args.internal_read_mode}' \
                          + (f'_k{args.orthogonal_state_modes}'
                             if args.internal_read_mode in {
                                 'spectral_state', 'spectral_linear'}
                             else '') \
                          + f'_ds{int(args.data_scale)}' \
                          + f'_loader{args.loader_seed}_s{args.seed}'
            elif args.model == 'StateReconTailAR':
                setting = f'ltf_{args.model_id}_{args.data}_ft{args.features}' \
                           + f'_sl{args.seq_len}_pl{args.pred_len}' \
                           + f'_dm{args.d_model}_nh{args.n_heads}_el{args.e_layers}_df{args.d_ff}' \
                           + f'_{args.des}_{ii}' \
                           + f'_p{args.recon_tail_fine_patch_len}g{args.recon_tail_group_patches}' \
                           + f'f{args.recon_tail_future_points}r{args.recon_tail_roll_groups}' \
                           + f'tr{args.recon_tail_train_rollouts}' \
                           + f'c{args.recon_tail_commit_points}w{args.recon_tail_history_weight:g}' \
                           + f'hm{args.recon_tail_history_mode}' \
                           + f'mr{args.recon_tail_history_mask_ratio:g}' \
                           + f'qc{args.recon_tail_query_context}' \
                           + f'qa{args.recon_tail_query_layer_aggregation}' \
                           + f'ls{args.recon_tail_loss_space}_n{args.ar_transformer_norm}' \
                           + f'_ds{int(args.data_scale)}_s{args.seed}'
            elif args.model == 'ConvTailAR':
                setting += f'_arp{args.ar_patch_len}_tail{args.tail_ar_patches}' \
                           + f'_roll{args.tail_ar_roll_patches}' \
                           + f'_ph{args.tail_placeholder_type}' \
                           + f'_norm{args.ar_transformer_norm}' \
                           + f'_ck{args.conv_tail_kernel_size}' \
                           + f'_ih{int(args.tail_independent_heads)}_datascale{int(args.data_scale)}'
            elif args.model == 'ValueMixTailAR':
                setting += f'_arp{args.ar_patch_len}_tail{args.tail_ar_patches}' \
                           + f'_roll{args.tail_ar_roll_patches}' \
                           + f'_ph{args.tail_placeholder_type}' \
                           + f'_norm{args.ar_transformer_norm}' \
                           + f'_vm{args.value_mixer_hidden_multiplier}' \
                           + f'_ih{int(args.tail_independent_heads)}_datascale{int(args.data_scale)}'
            elif args.model == 'SwinTailAR':
                setting += f'_arp{args.ar_patch_len}_tail{args.tail_ar_patches}' \
                           + f'_roll{args.tail_ar_roll_patches}' \
                           + f'_ph{args.tail_placeholder_type}' \
                           + f'_norm{args.ar_transformer_norm}' \
                           + f'_sw{args.swin_tail_window_size}' \
                           + f'_sh{args.swin_tail_shift_size}' \
                           + f'_hp{int(args.swin_tail_history_prepass)}' \
                           + f'_s1d{args.swin_tail_stage1_dim}' \
                           + f'_s1h{args.swin_tail_stage1_heads}' \
                           + f'_datascale{int(args.data_scale)}'
            elif args.model == 'SplitConcatTailAR':
                setting += f'_arp{args.ar_patch_len}' \
                           + f'_fine{args.split_concat_fine_patch_len}' \
                           + f'_tail{args.tail_ar_patches}' \
                           + f'_roll{args.tail_ar_roll_patches}' \
                           + f'_ph{args.tail_placeholder_type}' \
                           + f'_norm{args.ar_transformer_norm}' \
                           + f'_ih{int(args.tail_independent_heads)}' \
                           + f'_datascale{int(args.data_scale)}'
            elif args.model == 'SSMFuseTailAR':
                setting += f'_arp{args.ar_patch_len}' \
                           + f'_fine{args.ssm_fuse_fine_patch_len}' \
                           + f'_tail{args.tail_ar_patches}' \
                           + f'_roll{args.tail_ar_roll_patches}' \
                           + f'_ph{args.tail_placeholder_type}' \
                           + f'_norm{args.ar_transformer_norm}' \
                           + f'_ih{int(args.tail_independent_heads)}' \
                           + f'_datascale{int(args.data_scale)}'
            elif args.model == 'SSMFuseDenseAR':
                setting += f'_arp{args.ar_patch_len}' \
                           + f'_fine{args.ssm_fuse_fine_patch_len}' \
                           + f'_dense{args.dense_ar_roll_patches}' \
                           + f'_ls{args.dense_ar_loss_space}' \
                           + f'_eps{args.ar_norm_eps:g}' \
                           + f'_norm{args.ar_transformer_norm}' \
                           + f'_datascale{int(args.data_scale)}_s{args.seed}'
            elif args.model == 'SplitConcatDenseAR':
                setting += f'_arp{args.ar_patch_len}' \
                           + f'_fine{args.split_concat_fine_patch_len}' \
                           + f'_dense{args.dense_ar_roll_patches}' \
                           + f'_ls{args.dense_ar_loss_space}' \
                           + f'_eps{args.ar_norm_eps:g}' \
                           + f'_norm{args.ar_transformer_norm}' \
                           + f'_datascale{int(args.data_scale)}_s{args.seed}'
            elif args.model == 'SlidingWindowDenseAR':
                setting += f'_arp{args.ar_patch_len}' \
                           + f'_fine{args.sliding_window_fine_patch_len}' \
                           + f'_proj{args.sliding_window_projection}' \
                           + f'_ls{args.dense_ar_loss_space}' \
                           + f'_eps{args.ar_norm_eps:g}' \
                           + f'_norm{args.ar_transformer_norm}' \
                           + f'_datascale{int(args.data_scale)}_s{args.seed}'
            elif args.model == 'ProgressiveConcatAR':
                schedule = '-'.join(
                    str(value) for value in args.progressive_concat_sizes)
                setting += f'_arp{args.ar_patch_len}' \
                           + f'_fine{args.progressive_concat_fine_patch_len}' \
                           + f'_k{schedule}' \
                           + f'_s{args.progressive_concat_stride}' \
                           + f'_m{args.progressive_concat_max_patches}' \
                           + f'_ls{args.dense_ar_loss_space}' \
                           + f'_eps{args.ar_norm_eps:g}' \
                           + f'_norm{args.ar_transformer_norm}' \
                           + f'_datascale{int(args.data_scale)}_s{args.seed}'
            elif args.model in {'MambaDenseAR', 'SplitConcatMambaAR'}:
                setting += f'_arp{args.ar_patch_len}' \
                           + f'_fine{args.split_concat_fine_patch_len}' \
                           + f'_dense{args.dense_ar_roll_patches}' \
                           + f'_ls{args.dense_ar_loss_space}' \
                           + f'_eps{args.ar_norm_eps:g}' \
                           + f'_ms{args.ar_mamba_d_state}' \
                           + f'_ex{args.expand}_dc{args.d_conv}' \
                           + f'_ffn{int(args.ar_mamba_use_ffn)}' \
                           + f'_datascale{int(args.data_scale)}_s{args.seed}'
            elif args.model == 'RetrievalMambaAR':
                retrieval_lags = '-'.join(
                    str(value) for value in args.ssm_retrieval_lags)
                rollout_tag = (
                    f'_r{args.dense_ar_roll_patches}'
                    if args.dense_ar_roll_patches != 1 else ''
                )
                loss_tag = (
                    f'_huber{args.dense_ar_huber_delta:g}'
                    if args.dense_ar_loss_type == 'huber' else ''
                )
                setting = f'ltf_RMAR_{args.model_id}' \
                          + f'_w{args.seq_len}h{args.pred_len}' \
                          + f'_p{args.ar_patch_len}' \
                          + f'f{args.split_concat_fine_patch_len}' \
                          + f'_d{args.d_model}h{args.n_heads}' \
                          + f'l{args.e_layers}f{args.d_ff}' \
                          + f'_m{args.ar_mamba_d_state}' \
                          + f'x{args.expand}c{args.d_conv}' \
                          + f'_ff{int(args.ar_mamba_use_ffn)}' \
                          + rollout_tag \
                          + loss_tag \
                          + f'_{args.ssm_retrieval_mode}' \
                          + f'_lags{retrieval_lags}' \
                          + f'_rs{args.ssm_retrieval_scale_init:g}' \
                          + f'_ss{args.ssm_mamba_scale_init:g}' \
                          + f'_{args.dense_ar_loss_space}' \
                          + f'_ds{int(args.data_scale)}' \
                          + f'_loader{args.loader_seed}_s{args.seed}'
            elif args.model == 'AddressableMambaAR':
                rollout_tag = (
                    f'_r{args.dense_ar_roll_patches}'
                    if args.dense_ar_roll_patches != 1 else ''
                )
                loss_tag = (
                    f'_huber{args.dense_ar_huber_delta:g}'
                    if args.dense_ar_loss_type == 'huber' else ''
                )
                setting = f'ltf_AMAR_{args.model_id}' \
                          + f'_w{args.seq_len}h{args.pred_len}' \
                          + f'_p{args.ar_patch_len}' \
                          + f'f{args.split_concat_fine_patch_len}' \
                          + f'_d{args.d_model}h{args.n_heads}' \
                          + f'l{args.e_layers}f{args.d_ff}' \
                          + f'_m{args.ar_mamba_d_state}' \
                          + f'x{args.expand}c{args.d_conv}' \
                          + f'_ff{int(args.ar_mamba_use_ffn)}' \
                          + rollout_tag \
                          + loss_tag \
                          + f'_{args.addressable_mamba_mode}' \
                          + f'_k{args.addressable_mamba_key_dim}' \
                          + f'_s{args.addressable_mamba_scale_init:g}' \
                          + f'_g{args.addressable_mamba_gate_init:g}' \
                          + f'_{args.dense_ar_loss_space}' \
                          + f'_ds{int(args.data_scale)}' \
                          + f'_loader{args.loader_seed}_s{args.seed}'
            elif args.model == 'FineMambaCoarseAR':
                setting += f'_arp{args.ar_patch_len}' \
                           + f'_fine{args.ar_mamba_fine_patch_len}' \
                           + f'_ls{args.dense_ar_loss_space}' \
                           + f'_eps{args.ar_norm_eps:g}' \
                           + f'_ms{args.ar_mamba_d_state}' \
                           + f'_ex{args.expand}_dc{args.d_conv}' \
                           + f'_ffn{int(args.ar_mamba_use_ffn)}' \
                           + f'_datascale{int(args.data_scale)}_s{args.seed}'
            elif args.model == 'LaggedMambaAR':
                setting += f'_arp{args.ar_patch_len}' \
                           + f'_lag{args.ar_mamba_lag_points}' \
                           + f'_ls{args.dense_ar_loss_space}' \
                           + f'_eps{args.ar_norm_eps:g}' \
                           + f'_ms{args.ar_mamba_d_state}' \
                           + f'_ex{args.expand}_dc{args.d_conv}' \
                           + f'_ffn{int(args.ar_mamba_use_ffn)}' \
                           + f'_datascale{int(args.data_scale)}_s{args.seed}'
            elif args.model == 'FiniteSSMDenseAR':
                setting += f'_arp{args.ar_patch_len}' \
                           + f'_mem{args.finite_ssm_memory_patches}' \
                           + f'_dense{args.dense_ar_roll_patches}' \
                           + f'_ls{args.dense_ar_loss_space}' \
                           + f'_eps{args.ar_norm_eps:g}' \
                           + f'_norm{args.ar_transformer_norm}' \
                           + f'_datascale{int(args.data_scale)}_s{args.seed}'
            elif args.model == 'FiniteSSM24DenseAR':
                setting += f'_arp{args.ar_patch_len}' \
                           + f'_fine{args.finite_ssm_fine_patch_len}' \
                           + f'_mem{args.finite_ssm_memory_patches}' \
                           + f'_lm{args.finite_ssm_length_mode}' \
                           + f'_dense{args.dense_ar_roll_patches}' \
                           + f'_ls{args.dense_ar_loss_space}' \
                           + f'_eps{args.ar_norm_eps:g}' \
                           + f'_norm{args.ar_transformer_norm}' \
                           + f'_datascale{int(args.data_scale)}_s{args.seed}'
            elif args.model == 'LatentQueryAR':
                setting += f'_arp{args.ar_patch_len}' \
                           + f'_fine{args.finite_ssm_fine_patch_len}' \
                           + f'_mem{args.finite_ssm_memory_patches}' \
                           + f'_q{args.latent_query_states}' \
                           + f'_lh{args.latent_history_state_weight:g}' \
                           + f'_lf{args.latent_future_state_weight:g}' \
                           + f'_lp{args.latent_point_weight:g}' \
                           + f'_norm{args.ar_transformer_norm}' \
                           + f'_datascale{int(args.data_scale)}'
            elif args.model == 'AdaptiveTailAR':
                route_dist = '-'.join(
                    f'{value:g}' for value in args.adaptive_router_target_dist)
                setting += f'_arp{args.ar_patch_len}-{args.adaptive_fine_patch_len}' \
                           + f'_tail{args.tail_ar_patches}_roll{args.tail_ar_roll_patches}' \
                           + f'_ph{args.tail_placeholder_type}' \
                           + f'_norm{args.ar_transformer_norm}' \
                           + f'_rbr{args.adaptive_router_bias_rate:g}_rd{route_dist}' \
                           + f'_datascale{int(args.data_scale)}'
            elif args.model == 'EntropyTailAR':
                # Keep this below Linux's 255-byte single-component limit.  The
                # generic setting already repeats the long model id and model
                # name, so use a compact but complete EntropyTailAR identifier.
                setting = f'ltf_{args.model_id}_{args.data}_ft{args.features}' \
                           + f'_sl{args.seq_len}_pl{args.pred_len}' \
                           + f'_dm{args.d_model}_nh{args.n_heads}_el{args.e_layers}_df{args.d_ff}' \
                           + f'_do{args.dropout:g}_lr{args.learning_rate:g}_{args.des}_{ii}' \
                           + f'_p{args.ar_patch_len}_q{args.entropy_patch_quantile:g}' \
                           + f'_mx{args.entropy_max_patch_len}_ee{args.entropy_encoder_type}' \
                           + f'_t{args.tail_ar_patches}_r{args.tail_ar_roll_patches}' \
                           + f'_ph{args.tail_placeholder_type}_n{args.ar_transformer_norm}' \
                           + f'_m{int(args.entropy_monotonic)}_ds{int(args.data_scale)}'

            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            exp.train(setting)

            if args.skip_final_test:
                print(
                    '>>>>>>>final test skipped by protocol : '
                    '{}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            else:
                print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
                exp.test(setting)
            if args.use_gpu:
                if args.gpu_type == 'mps':
                    torch.backends.mps.empty_cache()
                elif args.gpu_type == 'cuda':
                    torch.cuda.empty_cache()
    else:
        exp = Exp(args)  # set experiments
        ii = 0
        setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_expand{}_dc{}_fc{}_eb{}_dt{}_{}_{}'.format(
            args.task_name,
            args.model_id,
            args.model,
            args.data,
            args.features,
            args.seq_len,
            args.label_len,
            args.pred_len,
            args.d_model,
            args.n_heads,
            args.e_layers,
            args.d_layers,
            args.d_ff,
            args.expand,
            args.d_conv,
            args.factor,
            args.embed,
            args.distil,
            args.des, ii)
        
        # Override setting for specific model to ensure proper checkpoint naming and logging
        if args.model == 'MambaSingleLayer' and args.task_name == 'classification':
            setting = f'{args.task_name}_CLS_{args.model_id}_{args.model}_{args.data}_ft{args.features}' \
                    + f'_sl{args.seq_len}_ll{args.label_len}_pl{args.pred_len}_dm{args.d_model}_ds{args.d_ff}' \
                    + f'_expand{args.expand}_dc{args.d_conv}_nk{args.num_kernels}' \
                    + f'_tvdt{args.tv_dt}_tvB{args.tv_B}_tvC{args.tv_C}_useD{int(args.use_D)}_{args.des}_{ii}'
        elif args.model == 'DenseAR':
            setting += f'_arp{args.ar_patch_len}_dense{args.dense_ar_roll_patches}' \
                       + f'_tail{args.tail_ar_patches}' \
                       + f'_ls{args.dense_ar_loss_space}' \
                       + f'_norm{args.ar_transformer_norm}' \
                       + f'_sep{int(args.tail_separate_projection)}' \
                       + f'_datascale{int(args.data_scale)}_s{args.seed}'
        elif args.model == 'TransformerAblationAR':
            setting = f'ltf_TAB_{args.model_id}_{args.data}' \
                      + f'_sl{args.seq_len}_pl{args.pred_len}' \
                      + f'_p{args.ar_patch_len}_dm{args.d_model}' \
                      + f'h{args.n_heads}l{args.e_layers}f{args.d_ff}' \
                      + f'_q{args.dense_ar_roll_patches}' \
                      + f'c{args.dense_ar_eval_commit_patches}' \
                      + f'_a{args.ablation_attention}' \
                      + (
                          f'{args.ablation_shrinkage_type}'
                          f'{args.ablation_shrinkage_alpha:g}'
                          f'dh{args.ablation_dynamic_heads}'
                          if args.ablation_attention == 'shrinkage'
                          else ''
                      ) \
                      + f'_f{args.ablation_ffn}_n{args.ablation_norm}' \
                      + f'_p{args.ablation_position_encoding}' \
                      + f'_r{int(args.ablation_attention_residual)}' \
                      + f'{int(args.ablation_ffn_residual)}' \
                      + f'_as{args.ablation_patch_assembly}' \
                      + f'{args.ablation_patch_assembly_fine_len}' \
                      + f'{args.ablation_patch_assembly_stem}' \
                      + (
                          f'a{args.ablation_patch_assembly_atom_dim}'
                          f'b{args.ablation_patch_assembly_basis_dim}'
                          if args.ablation_patch_assembly == 'factorized'
                          else ''
                      ) \
                      + f'_oh{args.ablation_output_head}' \
                      + f'{args.ablation_output_fine_patch_len}' \
                      + (
                          f'b{args.ablation_output_basis_dim}'
                          if args.ablation_output_head
                          == 'factorized_atoms'
                          else ''
                      ) \
                      + f'_e{args.ar_norm_eps:g}' \
                      + f'_ds{int(args.data_scale)}_s{args.seed}'
        elif args.model == 'TailAR':
            setting += f'_arp{args.ar_patch_len}_tail{args.tail_ar_patches}' \
                       + f'_roll{args.tail_ar_roll_patches}' \
                       + f'_ph{args.tail_placeholder_type}' \
                       + f'_norm{args.ar_transformer_norm}' \
                       + f'_sep{int(args.tail_separate_projection)}' \
                       + f'_ih{int(args.tail_independent_heads)}_datascale{int(args.data_scale)}'
        elif args.model == 'HybridTailAR':
            setting += f'_arp{args.ar_patch_len}_tail{args.tail_ar_patches}' \
                       + f'_roll{args.tail_ar_roll_patches}' \
                       + f'_ph{args.tail_placeholder_type}' \
                       + f'_norm{args.ar_transformer_norm}' \
                       + f'_hw{args.tail_history_loss_weight:g}' \
                       + f'_sep{int(args.tail_separate_projection)}' \
                       + f'_ih{int(args.tail_independent_heads)}_datascale{int(args.data_scale)}'
        elif args.model == 'LearnedPatchAR':
            setting += f'_arp{args.ar_patch_len}' \
                       + f'_tail{args.tail_ar_patches}' \
                       + f'_roll{args.tail_ar_roll_patches}' \
                       + f'_ph{args.tail_placeholder_type}' \
                       + f'_norm{args.ar_transformer_norm}' \
                       + f'_eps{args.ar_norm_eps:g}' \
                       + f'_datascale{int(args.data_scale)}_s{args.seed}'
        elif args.model == 'GRUSlotAR':
            setting += f'_arp{args.ar_patch_len}' \
                       + f'_g{args.ar_gru_slots}' \
                       + f'_roll{args.dense_ar_roll_patches}' \
                       + f'_ls{args.dense_ar_loss_space}' \
                       + f'_norm{args.ar_transformer_norm}' \
                       + f'_eps{args.ar_norm_eps:g}' \
                       + f'_datascale{int(args.data_scale)}_s{args.seed}'
        elif args.model == 'AddressableDeltaAR':
            setting = f'ltf_ADAR_{args.dataset_tag or args.data}' \
                      + f'_w{args.seq_len}h{args.pred_len}' \
                      + f'_p{args.ar_patch_len}' \
                      + f'f{args.split_concat_fine_patch_len}' \
                      + f'_k{args.dense_ar_roll_patches}' \
                      + (
                          f'_q{args.addressable_delta_train_patches}'
                          if args.addressable_delta_train_patches
                          not in {
                              0, args.dense_ar_roll_patches
                          } else ''
                      ) \
                      + f'_{args.addressable_delta_mode}' \
                      + f'_ak{args.addressable_delta_key_dim}' \
                      + f'_om{args.addressable_delta_output_max_scale:g}' \
                      + f'_sm{args.addressable_delta_state_max_scale:g}' \
                      + f'_gi{args.addressable_delta_gate_init:g}' \
                      + f'_b{args.addressable_delta_beta_init:g}' \
                      + f'_{args.dense_ar_loss_type}' \
                      + (
                          f'{args.dense_ar_huber_delta:g}'
                          if args.dense_ar_loss_type == 'huber' else ''
                      ) \
                      + (
                          f'_rm{args.addressable_delta_read_multiplier:g}'
                          if args.addressable_delta_read_multiplier != 1
                          else ''
                      ) \
                      + f'_d{args.d_model}h{args.n_heads}' \
                      + f'l{args.e_layers}f{args.d_ff}' \
                      + f'_n{args.ar_transformer_norm}' \
                      + f'_ds{int(args.data_scale)}' \
                      + f'_loader{args.loader_seed}_s{args.seed}'
        elif args.model == 'DeltaPlaceholderAR':
            setting = f'ltf_DPAR_{args.dataset_tag or args.data}' \
                      + f'_w{args.seq_len}h{args.pred_len}' \
                      + f'_p{args.ar_patch_len}' \
                      + f'f{args.delta_placeholder_fine_patch_len}' \
                      + f'_{args.delta_placeholder_history_fusion}' \
                      + f'_{args.delta_placeholder_attention_scope}' \
                      + (
                          f'_afgres'
                          f'{args.delta_placeholder_attention_gate_init:g}'
                          f'm{args.delta_placeholder_attention_gate_max:g}'
                          if args.delta_placeholder_attention_fusion
                          == 'gated_residual' else ''
                      ) \
                      + f'_tl{args.delta_placeholder_transformer_layers}' \
                      + f'_hw{args.delta_placeholder_history_weight:g}' \
                      + f'_{args.dense_ar_loss_type}' \
                      + (
                          f'{args.dense_ar_huber_delta:g}'
                          if args.dense_ar_loss_type == 'huber' else ''
                      ) \
                      + f'_d{args.d_model}h{args.n_heads}' \
                      + f'l{args.e_layers}f{args.d_ff}' \
                      + f'_n{args.ar_transformer_norm}' \
                      + f'_ds{int(args.data_scale)}' \
                      + f'_loader{args.loader_seed}_s{args.seed}'
        elif args.model == 'DeltaBlockRolloutAR':
            delta_rollout_queries = (
                1 + args.delta_block_rollout_placeholder_blocks)
            delta_rollout_commits = (
                args.delta_block_rollout_commit_blocks
                or delta_rollout_queries)
            setting = f'ltf_DBRA_{args.dataset_tag or args.data}' \
                      + f'_w{args.seq_len}h{args.pred_len}' \
                      + f'_p{args.ar_patch_len}' \
                      + f'f{args.delta_placeholder_fine_patch_len}' \
                      + f'_q{delta_rollout_queries}' \
                      + f'r{delta_rollout_commits}' \
                      + (
                          f'_t{args.delta_block_rollout_train_target_blocks}'
                          if args.delta_block_rollout_train_target_blocks
                          else ''
                      ) \
                      + (
                          f'_a{args.delta_placeholder_attention_scope}'
                          f'tl{args.delta_placeholder_transformer_layers}'
                          if args.delta_placeholder_attention_scope
                          != 'none' else ''
                      ) \
                      + (
                          f'_hf{args.delta_placeholder_history_fusion}'
                          if args.delta_placeholder_attention_scope
                          == 'all'
                          and args.delta_placeholder_history_fusion
                          != 'delta_boundary' else ''
                      ) \
                      + (
                          f'_afgres'
                          f'{args.delta_placeholder_attention_gate_init:g}'
                          f'm{args.delta_placeholder_attention_gate_max:g}'
                          if args.delta_placeholder_attention_scope
                          != 'none'
                          and args.delta_placeholder_attention_fusion
                          == 'gated_residual' else ''
                      ) \
                      + (
                          f'_cw'
                          f'{args.delta_block_rollout_second_block_weight:g}'
                          if args.delta_block_rollout_second_block_weight
                          != 1 else ''
                      ) \
                      + (
                          f'_sc'
                          f'{args.delta_block_rollout_state_consistency_weight:g}'
                          if args.delta_block_rollout_state_consistency_weight
                          else ''
                      ) \
                      + (
                          '_cq'
                          if args.delta_block_rollout_conditioned_placeholder
                          else ''
                      ) \
                      + (
                          '_pf'
                          if args.delta_block_rollout_predicted_feedback
                          else ''
                      ) \
                      + (
                          f'_fb'
                          f'{args.delta_block_rollout_state_feedback_bound:g}'
                          if args.delta_block_rollout_state_feedback_bound
                          else ''
                      ) \
                      + (
                          '_ch'
                          if args.delta_block_rollout_independent_closure_head
                          else ''
                      ) \
                      + (
                          '_iso'
                          if args.delta_block_rollout_isolate_closure_gradient
                          else ''
                      ) \
                      + (
                          '_asel'
                          if args.delta_block_rollout_anchor_selection
                          else ''
                      ) \
                      + (
                          '_pc'
                          if args.delta_block_rollout_project_closure_gradient
                          else ''
                      ) \
                      + (
                          f'_oc'
                          f'{args.delta_block_rollout_predictive_consistency_weight:g}'
                          f's{args.delta_block_rollout_predictive_consistency_steps}'
                          if args.delta_block_rollout_predictive_consistency_weight
                          else ''
                      ) \
                      + (
                          f'_alt'
                          f'{args.delta_block_rollout_alternate_closure_period}'
                          if args.delta_block_rollout_alternate_closure_period
                          else ''
                      ) \
                      + f'_hw{args.delta_placeholder_history_weight:g}' \
                      + f'_{args.dense_ar_loss_type}' \
                      + (
                          f'{args.dense_ar_huber_delta:g}'
                          if args.dense_ar_loss_type == 'huber' else ''
                      ) \
                      + f'_d{args.d_model}h{args.n_heads}' \
                      + f'l{args.e_layers}f{args.d_ff}' \
                      + f'_n{args.ar_transformer_norm}' \
                      + f'_ds{int(args.data_scale)}' \
                      + f'_loader{args.loader_seed}_s{args.seed}'
        elif args.model == 'RecurrentQueryDirectAR':
            setting = f'ltf_RQD_{args.dataset_tag or args.data}' \
                      + f'_w{args.seq_len}h{args.pred_len}' \
                      + f'_p{args.ar_patch_len}' \
                      + f'f{args.split_concat_fine_patch_len}' \
                      + f'_k{args.dense_ar_roll_patches}' \
                      + f'_r{args.recurrent_query_steps}' \
                      + f'_{args.recurrent_query_position_mode}' \
                      + (
                          f'_{args.recurrent_query_read_mode}'
                          if args.recurrent_query_read_mode != 'dynamic'
                          else ''
                      ) \
                      + (
                          f'_{args.recurrent_query_output_mode}'
                          if args.recurrent_query_output_mode != 'last'
                          else ''
                      ) \
                      + f'_t{args.recurrent_query_temperature:g}' \
                      + f'_d{args.d_model}h{args.n_heads}' \
                      + f'l{args.e_layers}f{args.d_ff}' \
                      + f'_n{args.ar_transformer_norm}' \
                      + f'_ds{int(args.data_scale)}' \
                      + f'_loader{args.loader_seed}_s{args.seed}'
        elif args.model == 'ProgressiveDirectAR':
            roll_schedule = '-'.join(
                str(value) for value in args.direct_roll_schedule)
            setting = f'ltf_PDAR_{args.data}' \
                      + f'_w{args.seq_len}h{args.pred_len}' \
                      + f'_p{args.ar_patch_len}q{args.tail_ar_patches}' \
                      + f'_r{roll_schedule}' \
                      + f'_d{args.d_model}l{args.e_layers}' \
                      + f'_n{args.ar_transformer_norm}' \
                      + f'_e{args.ar_norm_eps:g}_s{args.seed}'
        elif args.model == 'OverlapDirectAR':
            setting = f'ltf_ODAR_{args.data}' \
                      + f'_w{args.seq_len}h{args.pred_len}' \
                      + f'_p{args.ar_patch_len}' \
                      + f's{args.direct_history_patch_stride}' \
                      + f'q{args.tail_ar_patches}' \
                      + f'_pe{args.direct_position_encoding}' \
                      + f'_d{args.d_model}h{args.n_heads}' \
                      + f'l{args.e_layers}f{args.d_ff}' \
                      + f'_n{args.ar_transformer_norm}' \
                      + f'_ds{int(args.data_scale)}' \
                      + f'_loader{args.loader_seed}_s{args.seed}'
        elif args.model == 'HistoryOnlyDirectAR':
            setting = f'ltf_HOD_{args.data}' \
                      + f'_w{args.seq_len}h{args.pred_len}' \
                      + f'_f{args.history_only_fine_patch_len}' \
                      + f'g{args.history_only_group_patches}' \
                      + f'_pm{args.history_only_position_mode}' \
                      + f'_d{args.d_model}h{args.n_heads}' \
                      + f'l{args.e_layers}f{args.d_ff}' \
                      + f'_n{args.ar_transformer_norm}' \
                      + f'_ds{int(args.data_scale)}' \
                      + f'_loader{args.loader_seed}_s{args.seed}'
        elif args.model == 'BoundedMemoryDirectAR':
            setting = f'ltf_BMD_{args.dataset_tag or args.data}' \
                      + f'_w{args.seq_len}h{args.pred_len}' \
                      + f'_f{args.history_only_fine_patch_len}' \
                      + f'g{args.history_only_group_patches}' \
                      + f'_k{args.tail_ar_patches}' \
                      + f'_{args.bounded_memory_mode}' \
                      + (
                          f'_mg{args.bounded_memory_granularity}'
                          if args.bounded_memory_granularity != 'coarse'
                          else ''
                      ) \
                      + f'_{args.bounded_memory_stats_mode}' \
                      + (
                          f'_pm{args.bounded_memory_position_mode}'
                          if args.bounded_memory_position_mode != 'none'
                          else ''
                      ) \
                      + (
                          f'_fm{args.bounded_memory_fusion_mode}'
                          if args.bounded_memory_fusion_mode
                          != 'joint_attention'
                          else ''
                      ) \
                      + (
                          f'_fc{args.bounded_memory_far_context_mode}'
                          if args.bounded_memory_far_context_mode
                          != 'teacher_forced'
                          else ''
                      ) \
                      + (
                          '_rb'
                          + str(args.bounded_memory_residual_bound).replace('.', 'p')
                          if args.bounded_memory_residual_bound > 0
                          else ''
                      ) \
                      + f'_fo{args.bounded_memory_sampled_far_origins}' \
                      + (
                          '_fw'
                          + str(args.bounded_memory_far_loss_weight).replace('.', 'p')
                          if args.bounded_memory_far_loss_weight != 1.0
                          else ''
                      ) \
                      + (
                          f'_gm{args.bounded_memory_gradient_mode}'
                          if args.bounded_memory_gradient_mode != 'joint'
                          else ''
                      ) \
                      + f'_d{args.d_model}h{args.n_heads}' \
                      + f'l{args.e_layers}f{args.d_ff}' \
                      + f'_n{args.ar_transformer_norm}' \
                      + f'_ds{int(args.data_scale)}' \
                      + f'_loader{args.loader_seed}_s{args.seed}'
        elif args.model == 'SemigroupFlow':
            setting = f'ltf_SGF_{args.dataset_tag or args.data}' \
                      + f'_w{args.seq_len}h{args.pred_len}' \
                      + f'_b{args.semigroup_base_patch_len}' \
                      + f'_hp{args.semigroup_history_patch_len}' \
                      + f's{args.semigroup_history_stride}' \
                      + f'_{args.semigroup_flow_mode}' \
                      + (
                          f'_a{args.semigroup_anchor}'
                          if args.semigroup_anchor != 'last' else ''
                      ) \
                      + (
                          f'_ls{args.semigroup_loss_space}'
                          if args.semigroup_loss_space != 'normalized'
                          else ''
                      ) \
                      + f'_z{args.semigroup_state_dim}' \
                      + f'_d{args.d_model}h{args.n_heads}' \
                      + f'l{args.e_layers}f{args.d_ff}' \
                      + f'_ds{int(args.data_scale)}' \
                      + f'_loader{args.loader_seed}_s{args.seed}'
        elif args.model == 'AddressableStateDirect':
            setting = f'ltf_ASD_{args.data}' \
                      + f'_w{args.seq_len}h{args.pred_len}' \
                      + f'_p{args.seg_len}' \
                      + f'f{args.addressable_state_fine_patch_len}' \
                      + f'_{args.addressable_state_mode}' \
                      + f'_d{args.d_model}h{args.n_heads}' \
                      + f'_t{args.addressable_state_temperature:g}' \
                      + f'_nm{args.addressable_state_null_margin:g}' \
                      + f'_mr{args.addressable_state_max_residual:g}' \
                      + f'_ce{int(args.horizon_cross_channel_embedding)}' \
                      + f'_ds{int(args.data_scale)}' \
                      + (
                          '_fr1'
                          if args.addressable_state_freeze_base else ''
                      ) \
                      + f'_loader{args.loader_seed}_s{args.seed}'
        elif args.model == 'PersistentSuccessorAR':
            setting = f'ltf_PSAR_{args.dataset_tag or args.data}' \
                      + f'_w{args.seq_len}h{args.pred_len}' \
                      + f'_p{args.ar_patch_len}' \
                      + f'k{args.tail_ar_patches}' \
                      + f'r{args.tail_ar_roll_patches}' \
                      + f'_{args.persistent_successor_mode}' \
                      + f'_hw{args.persistent_successor_history_weight:g}' \
                      + f'_lm{args.persistent_successor_loss_mode}' \
                      + f'_fw{args.persistent_successor_far_weight:g}' \
                      + f'_gg{args.persistent_successor_gradient_group}' \
                      + (
                          f'_sc{args.persistent_successor_far_sample_count}'
                          if args.persistent_successor_loss_mode
                          == 'sampled_far' else ''
                      ) \
                      + f'_rs{args.persistent_successor_residual_scale:g}' \
                      + (
                          f'_rx{args.persistent_successor_residual_max_scale:g}'
                          if args.persistent_successor_residual_max_scale
                          is not None else ''
                      ) \
                      + (
                          f'_rg{args.persistent_successor_residual_gate}'
                          if args.persistent_successor_residual_gate
                          != 'real_fraction' else ''
                      ) \
                      + (
                          f'_sm{args.persistent_successor_stats_mode}'
                          if args.persistent_successor_stats_mode
                          != 'rolling' else ''
                      ) \
                          + (
                              '_ih1'
                              if args.tail_independent_heads else ''
                          ) \
                          + (
                              f'_rr{args.persistent_successor_retained_real_patches}'
                              if args.persistent_successor_retained_real_patches
                              > 0 else ''
                          ) \
                          + f'_t{args.persistent_successor_temperature:g}' \
                      + f'_nm{args.persistent_successor_null_margin:g}' \
                      + f'_d{args.d_model}h{args.n_heads}' \
                      + f'l{args.e_layers}f{args.d_ff}' \
                      + f'_ds{int(args.data_scale)}' \
                      + f'_loader{args.loader_seed}_s{args.seed}'
        elif args.model == 'SlidingStateDirectAR':
            setting = f'ltf_SSD_{args.data}' \
                      + f'_w{args.seq_len}h{args.pred_len}' \
                      + f'_f{args.recon_tail_fine_patch_len}' \
                      + f'g{args.recon_tail_group_patches}' \
                      + f'q{args.recon_tail_future_points}' \
                      + f'_pe{args.direct_position_encoding}' \
                      + f'_d{args.d_model}h{args.n_heads}' \
                      + f'l{args.e_layers}f{args.d_ff}' \
                      + f'_n{args.ar_transformer_norm}' \
                      + f'_ds{int(args.data_scale)}' \
                      + f'_loader{args.loader_seed}_s{args.seed}'
        elif args.model == 'HorizonCrossTransformer':
            setting = f'ltf_HCT_{args.data}' \
                      + f'_w{args.seq_len}h{args.pred_len}' \
                      + f'_p{args.seg_len}_d{args.d_model}' \
                      + f'l{args.e_layers}f{args.d_ff}' \
                      + f'_a{args.horizon_cross_anchor}' \
                      + f'_ce{int(args.horizon_cross_channel_embedding)}' \
                      + f'_s{args.seed}'
        elif args.model == 'TransformerStateDirect':
            setting = f'ltf_TSD_{args.model_id}_{args.data}' \
                      + f'_w{args.seq_len}h{args.pred_len}' \
                      + f'_p{args.seg_len}' \
                      + f'_d{args.d_model}h{args.n_heads}' \
                      + f'l{args.e_layers}f{args.d_ff}' \
                      + f'_a{args.transformer_state_anchor}' \
                      + (
                          f'_rg{args.transformer_state_read_gate_init:g}'
                          if args.transformer_state_read_gate_init > -1
                          else ''
                      ) \
                      + f'_ce{int(args.horizon_cross_channel_embedding)}' \
                      + f'_ds{int(args.data_scale)}' \
                      + f'_loader{args.loader_seed}_s{args.seed}'
        elif args.model == 'SegRNNMemoryTransformer':
            setting = f'ltf_SRMT_{args.data}' \
                      + f'_w{args.seq_len}h{args.pred_len}' \
                      + f'_p{args.seg_len}_d{args.d_model}' \
                      + f'l{args.e_layers}f{args.d_ff}' \
                      + f'_local{args.segrnn_memory_local_segments}' \
                      + f'_mi{args.segrnn_memory_scale_init:g}' \
                      + f'_mx{args.segrnn_memory_max_scale:g}' \
                      + f'_fr{int(args.segrnn_memory_freeze_base)}' \
                      + f'_ce{int(args.horizon_cross_channel_embedding)}' \
                      + f'_s{args.seed}'
        elif args.model == 'StateReadDirect':
            setting = f'ltf_SRD_{args.model_id}' \
                      + f'_w{args.seq_len}h{args.pred_len}' \
                      + f'_p{args.seg_len}' \
                      + (
                          f'_ep{args.state_read_encoder_patch_len}'
                          if args.state_read_encoder_patch_len not in {
                              0, args.seg_len
                          }
                          else ''
                      ) \
                      + f'_d{args.d_model}h{args.n_heads}' \
                      + f'l{args.e_layers}f{args.d_ff}' \
                      + f'_m{args.ar_mamba_d_state}' \
                      + f'x{args.expand}c{args.d_conv}' \
                      + f'_ff{int(args.ar_mamba_use_ffn)}' \
                      + f'_{args.state_read_mode}' \
                      + f'_nm{args.state_read_null_margin:g}' \
                      + f'_t{args.state_read_temperature:g}' \
                      + (
                          f'_fr{int(args.state_read_freeze_base)}'
                          f'_c{args.state_read_max_residual:g}'
                          if args.state_read_freeze_base
                          or args.state_read_mode in {
                              'bounded_read',
                              'analog_read',
                              'analog_linear_read',
                              'analog_phase_read',
                              'stable_analog_read',
                              'innovation_read',
                              'state_delta_read',
                              'state_context_read',
                              'linear_read',
                              'phase_linear_read',
                              'channel_phase_linear_read',
                          }
                          else ''
                      ) \
                      + (
                          f'_hw{args.state_read_history_weight:g}'
                          f'_hl{args.state_read_history_loss_space}'
                          if args.state_read_history_weight > 0
                          else ''
                      ) \
                      + f'_ds{int(args.data_scale)}' \
                      + f'_loader{args.loader_seed}_s{args.seed}'
        elif args.model == 'InternalReadSSM':
            setting = f'ltf_IRS_{args.model_id}' \
                      + f'_w{args.seq_len}h{args.pred_len}' \
                      + f'_p{args.seg_len}' \
                      + f'_d{args.d_model}l{args.e_layers}' \
                      + f'f{args.d_ff}_m{args.ar_mamba_d_state}' \
                      + f'x{args.expand}c{args.d_conv}' \
                      + f'_ff{int(args.ar_mamba_use_ffn)}' \
                      + f'_{args.internal_read_mode}' \
                      + (f'_k{args.orthogonal_state_modes}'
                         if args.internal_read_mode in {
                             'spectral_state', 'spectral_linear'}
                         else '') \
                      + f'_ds{int(args.data_scale)}' \
                      + f'_loader{args.loader_seed}_s{args.seed}'
        elif args.model == 'StateReconTailAR':
            setting = f'ltf_{args.model_id}_{args.data}_ft{args.features}' \
                      + f'_sl{args.seq_len}_pl{args.pred_len}' \
                      + f'_dm{args.d_model}_nh{args.n_heads}_el{args.e_layers}_df{args.d_ff}' \
                      + f'_{args.des}_{ii}' \
                      + f'_p{args.recon_tail_fine_patch_len}g{args.recon_tail_group_patches}' \
                      + f'f{args.recon_tail_future_points}r{args.recon_tail_roll_groups}' \
                      + f'tr{args.recon_tail_train_rollouts}' \
                      + f'c{args.recon_tail_commit_points}w{args.recon_tail_history_weight:g}' \
                      + f'hm{args.recon_tail_history_mode}' \
                      + f'mr{args.recon_tail_history_mask_ratio:g}' \
                      + f'qc{args.recon_tail_query_context}' \
                      + f'qa{args.recon_tail_query_layer_aggregation}' \
                      + f'ls{args.recon_tail_loss_space}_n{args.ar_transformer_norm}' \
                      + f'_ds{int(args.data_scale)}_s{args.seed}'
        elif args.model == 'ConvTailAR':
            setting += f'_arp{args.ar_patch_len}_tail{args.tail_ar_patches}' \
                       + f'_roll{args.tail_ar_roll_patches}' \
                       + f'_ph{args.tail_placeholder_type}' \
                       + f'_norm{args.ar_transformer_norm}' \
                       + f'_ck{args.conv_tail_kernel_size}' \
                       + f'_ih{int(args.tail_independent_heads)}_datascale{int(args.data_scale)}'
        elif args.model == 'ValueMixTailAR':
            setting += f'_arp{args.ar_patch_len}_tail{args.tail_ar_patches}' \
                       + f'_roll{args.tail_ar_roll_patches}' \
                       + f'_ph{args.tail_placeholder_type}' \
                       + f'_norm{args.ar_transformer_norm}' \
                       + f'_vm{args.value_mixer_hidden_multiplier}' \
                       + f'_ih{int(args.tail_independent_heads)}_datascale{int(args.data_scale)}'
        elif args.model == 'SwinTailAR':
            setting += f'_arp{args.ar_patch_len}_tail{args.tail_ar_patches}' \
                       + f'_roll{args.tail_ar_roll_patches}' \
                       + f'_ph{args.tail_placeholder_type}' \
                       + f'_norm{args.ar_transformer_norm}' \
                       + f'_sw{args.swin_tail_window_size}' \
                       + f'_sh{args.swin_tail_shift_size}' \
                       + f'_hp{int(args.swin_tail_history_prepass)}' \
                       + f'_s1d{args.swin_tail_stage1_dim}' \
                       + f'_s1h{args.swin_tail_stage1_heads}' \
                       + f'_datascale{int(args.data_scale)}'
        elif args.model == 'SplitConcatTailAR':
            setting += f'_arp{args.ar_patch_len}' \
                       + f'_fine{args.split_concat_fine_patch_len}' \
                       + f'_tail{args.tail_ar_patches}' \
                       + f'_roll{args.tail_ar_roll_patches}' \
                       + f'_ph{args.tail_placeholder_type}' \
                       + f'_norm{args.ar_transformer_norm}' \
                       + f'_ih{int(args.tail_independent_heads)}' \
                       + f'_datascale{int(args.data_scale)}'
        elif args.model == 'SSMFuseTailAR':
            setting += f'_arp{args.ar_patch_len}' \
                       + f'_fine{args.ssm_fuse_fine_patch_len}' \
                       + f'_tail{args.tail_ar_patches}' \
                       + f'_roll{args.tail_ar_roll_patches}' \
                       + f'_ph{args.tail_placeholder_type}' \
                       + f'_norm{args.ar_transformer_norm}' \
                       + f'_ih{int(args.tail_independent_heads)}' \
                       + f'_datascale{int(args.data_scale)}'
        elif args.model == 'SSMFuseDenseAR':
            setting += f'_arp{args.ar_patch_len}' \
                       + f'_fine{args.ssm_fuse_fine_patch_len}' \
                       + f'_dense{args.dense_ar_roll_patches}' \
                       + f'_ls{args.dense_ar_loss_space}' \
                       + f'_eps{args.ar_norm_eps:g}' \
                       + f'_norm{args.ar_transformer_norm}' \
                       + f'_datascale{int(args.data_scale)}_s{args.seed}'
        elif args.model == 'SplitConcatDenseAR':
            setting += f'_arp{args.ar_patch_len}' \
                       + f'_fine{args.split_concat_fine_patch_len}' \
                       + f'_dense{args.dense_ar_roll_patches}' \
                       + f'_ls{args.dense_ar_loss_space}' \
                       + f'_eps{args.ar_norm_eps:g}' \
                       + f'_norm{args.ar_transformer_norm}' \
                       + f'_datascale{int(args.data_scale)}_s{args.seed}'
        elif args.model == 'SlidingWindowDenseAR':
            setting += f'_arp{args.ar_patch_len}' \
                       + f'_fine{args.sliding_window_fine_patch_len}' \
                       + f'_proj{args.sliding_window_projection}' \
                       + f'_ls{args.dense_ar_loss_space}' \
                       + f'_eps{args.ar_norm_eps:g}' \
                       + f'_norm{args.ar_transformer_norm}' \
                       + f'_datascale{int(args.data_scale)}_s{args.seed}'
        elif args.model == 'ProgressiveConcatAR':
            schedule = '-'.join(
                str(value) for value in args.progressive_concat_sizes)
            setting += f'_arp{args.ar_patch_len}' \
                       + f'_fine{args.progressive_concat_fine_patch_len}' \
                       + f'_k{schedule}' \
                       + f'_s{args.progressive_concat_stride}' \
                       + f'_m{args.progressive_concat_max_patches}' \
                       + f'_ls{args.dense_ar_loss_space}' \
                       + f'_eps{args.ar_norm_eps:g}' \
                       + f'_norm{args.ar_transformer_norm}' \
                       + f'_datascale{int(args.data_scale)}_s{args.seed}'
        elif args.model in {'MambaDenseAR', 'SplitConcatMambaAR'}:
            setting += f'_arp{args.ar_patch_len}' \
                       + f'_fine{args.split_concat_fine_patch_len}' \
                       + f'_dense{args.dense_ar_roll_patches}' \
                       + f'_ls{args.dense_ar_loss_space}' \
                       + f'_eps{args.ar_norm_eps:g}' \
                       + f'_ms{args.ar_mamba_d_state}' \
                       + f'_ex{args.expand}_dc{args.d_conv}' \
                       + f'_ffn{int(args.ar_mamba_use_ffn)}' \
                       + f'_datascale{int(args.data_scale)}_s{args.seed}'
        elif args.model == 'RetrievalMambaAR':
            retrieval_lags = '-'.join(
                str(value) for value in args.ssm_retrieval_lags)
            rollout_tag = (
                f'_r{args.dense_ar_roll_patches}'
                if args.dense_ar_roll_patches != 1 else ''
            )
            loss_tag = (
                f'_huber{args.dense_ar_huber_delta:g}'
                if args.dense_ar_loss_type == 'huber' else ''
            )
            setting = f'ltf_RMAR_{args.model_id}' \
                      + f'_w{args.seq_len}h{args.pred_len}' \
                      + f'_p{args.ar_patch_len}' \
                      + f'f{args.split_concat_fine_patch_len}' \
                      + f'_d{args.d_model}h{args.n_heads}' \
                      + f'l{args.e_layers}f{args.d_ff}' \
                      + f'_m{args.ar_mamba_d_state}' \
                      + f'x{args.expand}c{args.d_conv}' \
                      + f'_ff{int(args.ar_mamba_use_ffn)}' \
                      + rollout_tag \
                      + loss_tag \
                      + f'_{args.ssm_retrieval_mode}' \
                      + f'_lags{retrieval_lags}' \
                      + f'_rs{args.ssm_retrieval_scale_init:g}' \
                      + f'_ss{args.ssm_mamba_scale_init:g}' \
                      + f'_{args.dense_ar_loss_space}' \
                      + f'_ds{int(args.data_scale)}' \
                      + f'_loader{args.loader_seed}_s{args.seed}'
        elif args.model == 'AddressableMambaAR':
            rollout_tag = (
                f'_r{args.dense_ar_roll_patches}'
                if args.dense_ar_roll_patches != 1 else ''
            )
            loss_tag = (
                f'_huber{args.dense_ar_huber_delta:g}'
                if args.dense_ar_loss_type == 'huber' else ''
            )
            setting = f'ltf_AMAR_{args.model_id}' \
                      + f'_w{args.seq_len}h{args.pred_len}' \
                      + f'_p{args.ar_patch_len}' \
                      + f'f{args.split_concat_fine_patch_len}' \
                      + f'_d{args.d_model}h{args.n_heads}' \
                      + f'l{args.e_layers}f{args.d_ff}' \
                      + f'_m{args.ar_mamba_d_state}' \
                      + f'x{args.expand}c{args.d_conv}' \
                      + f'_ff{int(args.ar_mamba_use_ffn)}' \
                      + rollout_tag \
                      + loss_tag \
                      + f'_{args.addressable_mamba_mode}' \
                      + f'_k{args.addressable_mamba_key_dim}' \
                      + f'_s{args.addressable_mamba_scale_init:g}' \
                      + f'_g{args.addressable_mamba_gate_init:g}' \
                      + f'_{args.dense_ar_loss_space}' \
                      + f'_ds{int(args.data_scale)}' \
                      + f'_loader{args.loader_seed}_s{args.seed}'
        elif args.model == 'FineMambaCoarseAR':
            setting += f'_arp{args.ar_patch_len}' \
                       + f'_fine{args.ar_mamba_fine_patch_len}' \
                       + f'_ls{args.dense_ar_loss_space}' \
                       + f'_eps{args.ar_norm_eps:g}' \
                       + f'_ms{args.ar_mamba_d_state}' \
                       + f'_ex{args.expand}_dc{args.d_conv}' \
                       + f'_ffn{int(args.ar_mamba_use_ffn)}' \
                       + f'_datascale{int(args.data_scale)}_s{args.seed}'
        elif args.model == 'LaggedMambaAR':
            setting += f'_arp{args.ar_patch_len}' \
                       + f'_lag{args.ar_mamba_lag_points}' \
                       + f'_ls{args.dense_ar_loss_space}' \
                       + f'_eps{args.ar_norm_eps:g}' \
                       + f'_ms{args.ar_mamba_d_state}' \
                       + f'_ex{args.expand}_dc{args.d_conv}' \
                       + f'_ffn{int(args.ar_mamba_use_ffn)}' \
                       + f'_datascale{int(args.data_scale)}_s{args.seed}'
        elif args.model == 'FiniteSSMDenseAR':
            setting += f'_arp{args.ar_patch_len}' \
                       + f'_mem{args.finite_ssm_memory_patches}' \
                       + f'_dense{args.dense_ar_roll_patches}' \
                       + f'_ls{args.dense_ar_loss_space}' \
                       + f'_eps{args.ar_norm_eps:g}' \
                       + f'_norm{args.ar_transformer_norm}' \
                       + f'_datascale{int(args.data_scale)}_s{args.seed}'
        elif args.model == 'FiniteSSM24DenseAR':
            setting += f'_arp{args.ar_patch_len}' \
                       + f'_fine{args.finite_ssm_fine_patch_len}' \
                       + f'_mem{args.finite_ssm_memory_patches}' \
                       + f'_lm{args.finite_ssm_length_mode}' \
                       + f'_dense{args.dense_ar_roll_patches}' \
                       + f'_ls{args.dense_ar_loss_space}' \
                       + f'_eps{args.ar_norm_eps:g}' \
                       + f'_norm{args.ar_transformer_norm}' \
                       + f'_datascale{int(args.data_scale)}_s{args.seed}'
        elif args.model == 'LatentQueryAR':
            setting += f'_arp{args.ar_patch_len}' \
                       + f'_fine{args.finite_ssm_fine_patch_len}' \
                       + f'_mem{args.finite_ssm_memory_patches}' \
                       + f'_q{args.latent_query_states}' \
                       + f'_lh{args.latent_history_state_weight:g}' \
                       + f'_lf{args.latent_future_state_weight:g}' \
                       + f'_lp{args.latent_point_weight:g}' \
                       + f'_norm{args.ar_transformer_norm}' \
                       + f'_datascale{int(args.data_scale)}'
        elif args.model == 'AdaptiveTailAR':
            route_dist = '-'.join(
                f'{value:g}' for value in args.adaptive_router_target_dist)
            setting += f'_arp{args.ar_patch_len}-{args.adaptive_fine_patch_len}' \
                       + f'_tail{args.tail_ar_patches}_roll{args.tail_ar_roll_patches}' \
                       + f'_ph{args.tail_placeholder_type}' \
                       + f'_norm{args.ar_transformer_norm}' \
                       + f'_rbr{args.adaptive_router_bias_rate:g}_rd{route_dist}' \
                       + f'_datascale{int(args.data_scale)}'
        elif args.model == 'EntropyTailAR':
            setting = f'ltf_{args.model_id}_{args.data}_ft{args.features}' \
                       + f'_sl{args.seq_len}_pl{args.pred_len}' \
                       + f'_dm{args.d_model}_nh{args.n_heads}_el{args.e_layers}_df{args.d_ff}' \
                       + f'_do{args.dropout:g}_lr{args.learning_rate:g}_{args.des}_{ii}' \
                       + f'_p{args.ar_patch_len}_q{args.entropy_patch_quantile:g}' \
                       + f'_mx{args.entropy_max_patch_len}_ee{args.entropy_encoder_type}' \
                       + f'_t{args.tail_ar_patches}_r{args.tail_ar_roll_patches}' \
                       + f'_ph{args.tail_placeholder_type}_n{args.ar_transformer_norm}' \
                       + f'_m{int(args.entropy_monotonic)}_ds{int(args.data_scale)}'

        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting, test=1)
        if args.use_gpu:
            if args.gpu_type == 'mps':
                torch.backends.mps.empty_cache()
            elif args.gpu_type == 'cuda':
                torch.cuda.empty_cache()
