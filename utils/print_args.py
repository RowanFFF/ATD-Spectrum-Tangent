def print_args(args):
    print("\033[1m" + "Basic Config" + "\033[0m")
    print(f'  {"Task Name:":<20}{args.task_name:<20}{"Is Training:":<20}{args.is_training:<20}')
    print(f'  {"Model ID:":<20}{args.model_id:<20}{"Model:":<20}{args.model:<20}')
    print()

    print("\033[1m" + "Data Loader" + "\033[0m")
    print(f'  {"Data:":<20}{args.data:<20}{"Root Path:":<20}{args.root_path:<20}')
    print(f'  {"Data Path:":<20}{args.data_path:<20}{"Features:":<20}{args.features:<20}')
    print(f'  {"Target:":<20}{args.target:<20}{"Freq:":<20}{args.freq:<20}')
    print(f'  {"Checkpoints:":<20}{args.checkpoints:<20}{"Data Scale:":<20}{args.data_scale:<20}')
    if args.checkpoint_path:
        print(f'  {"Load Checkpoint:":<20}{args.checkpoint_path}')
    print()

    if args.task_name in ['long_term_forecast', 'short_term_forecast']:
        print("\033[1m" + "Forecasting Task" + "\033[0m")
        print(f'  {"Seq Len:":<20}{args.seq_len:<20}{"Label Len:":<20}{args.label_len:<20}')
        print(f'  {"Pred Len:":<20}{args.pred_len:<20}{"Seasonal Patterns:":<20}{args.seasonal_patterns:<20}')
        print(f'  {"Inverse:":<20}{args.inverse:<20}')
        print()

    if args.task_name == 'imputation':
        print("\033[1m" + "Imputation Task" + "\033[0m")
        print(f'  {"Mask Rate:":<20}{args.mask_rate:<20}')
        print()

    if args.task_name == 'anomaly_detection':
        print("\033[1m" + "Anomaly Detection Task" + "\033[0m")
        print(f'  {"Anomaly Ratio:":<20}{args.anomaly_ratio:<20}')
        print()

    print("\033[1m" + "Model Parameters" + "\033[0m")
    print(f'  {"Top k:":<20}{args.top_k:<20}{"Num Kernels:":<20}{args.num_kernels:<20}')
    print(f'  {"Enc In:":<20}{args.enc_in:<20}{"Dec In:":<20}{args.dec_in:<20}')
    print(f'  {"C Out:":<20}{args.c_out:<20}{"d model:":<20}{args.d_model:<20}')
    print(f'  {"n heads:":<20}{args.n_heads:<20}{"e layers:":<20}{args.e_layers:<20}')
    print(f'  {"d layers:":<20}{args.d_layers:<20}{"d FF:":<20}{args.d_ff:<20}')
    print(f'  {"Moving Avg:":<20}{args.moving_avg:<20}{"Factor:":<20}{args.factor:<20}')
    print(f'  {"Distil:":<20}{args.distil:<20}{"Dropout:":<20}{args.dropout:<20}')
    print(f'  {"Embed:":<20}{args.embed:<20}{"Activation:":<20}{args.activation:<20}')
    if args.model in ['DenseAR', 'TailAR', 'HybridTailAR', 'StateReconTailAR',
                      'SlidingStateDirectAR',
                      'AdaptiveTailAR', 'EntropyTailAR',
                      'ConvTailAR', 'ValueMixTailAR', 'SwinTailAR',
                      'SplitConcatTailAR', 'SSMFuseTailAR',
                      'SSMFuseDenseAR', 'FiniteSSMDenseAR',
                      'FiniteSSM24DenseAR', 'LatentQueryAR',
                      'TransformerAblationAR', 'ProgressiveDirectAR',
                      'OverlapDirectAR', 'HistoryOnlyDirectAR',
                      'BoundedMemoryDirectAR', 'RecurrentQueryDirectAR']:
        print(f'  {"AR Patch Len:":<20}{args.ar_patch_len:<20}{"Tail Patches:":<20}{args.tail_ar_patches:<20}')
        print(f'  {"Dense Roll Patches:":<20}{args.dense_ar_roll_patches:<20}{"Tail Roll Patches:":<20}{args.tail_ar_roll_patches:<20}')
        if args.model in ['DenseAR', 'TransformerAblationAR']:
            print(f'  {"Dense Loss Space:":<20}{args.dense_ar_loss_space:<20}')
        print(f'  {"Tail Placeholder:":<20}{args.tail_placeholder_type:<20}{"Tail Separate Proj:":<20}{args.tail_separate_projection:<20}')
        print(f'  {"Independent Heads:":<20}{args.tail_independent_heads:<20}')
        print(f'  {"Tail Loss Sampling:":<20}{args.tail_loss_sampling:<20}{"Loss Patch Count:":<20}{args.tail_loss_sample_count:<20}')
        if args.model == 'HybridTailAR':
            print(f'  {"History Loss Wt:":<20}{args.tail_history_loss_weight:<20}')
        if args.model in ['StateReconTailAR', 'SlidingStateDirectAR']:
            print(f'  {"Recon Fine Patch:":<20}{args.recon_tail_fine_patch_len:<20}{"Group Patches:":<20}{args.recon_tail_group_patches:<20}')
            print(f'  {"Future Points:":<20}{args.recon_tail_future_points:<20}{"Roll Groups:":<20}{args.recon_tail_roll_groups:<20}')
            print(f'  {"Train Rollouts:":<20}{args.recon_tail_train_rollouts:<20}')
            print(f'  {"Commit Points:":<20}{args.recon_tail_commit_points:<20}{"Recon History Wt:":<20}{args.recon_tail_history_weight:<20}')
            print(f'  {"Recon History Mode:":<20}{args.recon_tail_history_mode:<20}{"History Mask Ratio:":<20}{args.recon_tail_history_mask_ratio:<20}')
            print(f'  {"Recon Loss Space:":<20}{args.recon_tail_loss_space:<20}')
        print(f'  {"AR Norm Eps:":<20}{args.ar_norm_eps:<20}{"Transformer Norm:":<20}{args.ar_transformer_norm:<20}')
        print(f'  {"Attention Mode:":<20}{args.ar_attention_mode:<20}')
        if args.model == 'TransformerAblationAR':
            print(f'  {"Ablation Attention:":<20}{args.ablation_attention:<20}{"Ablation FFN:":<20}{args.ablation_ffn:<20}')
            print(f'  {"Ablation Norm:":<20}{args.ablation_norm:<20}{"Position Encoding:":<20}{args.ablation_position_encoding:<20}')
            print(f'  {"Attention Residual:":<20}{args.ablation_attention_residual:<20}{"FFN Residual:":<20}{args.ablation_ffn_residual:<20}')
            print(f'  {"Patch Assembly:":<20}{args.ablation_patch_assembly:<20}{"Fixed Atom Dim:":<20}{args.ablation_patch_assembly_atom_dim:<20}')
            print(f'  {"Assembly Basis:":<20}{args.ablation_patch_assembly_basis_dim:<20}')
            print(f'  {"Output Head:":<20}{args.ablation_output_head:<20}{"Output Basis:":<20}{args.ablation_output_basis_dim:<20}')
        if args.model == 'ProgressiveDirectAR':
            schedule = '/'.join(
                str(value) for value in args.direct_roll_schedule)
            print(f'  {"Direct Roll Schedule:":<20}{schedule:<20}')
        if args.model == 'OverlapDirectAR':
            print(f'  {"History Patch Stride:":<20}{args.direct_history_patch_stride:<20}{"Direct Position:":<20}{args.direct_position_encoding:<20}')
        if args.model == 'HistoryOnlyDirectAR':
            print(f'  {"Fine Patch Len:":<20}{args.history_only_fine_patch_len:<20}{"Concat Patches:":<20}{args.history_only_group_patches:<20}')
            print(f'  {"Address Structure:":<20}{args.history_only_position_mode:<20}{"Future Self-Attn:":<20}{"none":<20}')
        if args.model == 'BoundedMemoryDirectAR':
            print(f'  {"Fine Patch Len:":<20}{args.history_only_fine_patch_len:<20}{"Concat Patches:":<20}{args.history_only_group_patches:<20}')
            print(f'  {"Memory Mode:":<20}{args.bounded_memory_mode:<20}{"Memory Granularity:":<20}{args.bounded_memory_granularity:<20}')
            print(f'  {"Stats Mode:":<20}{args.bounded_memory_stats_mode:<20}{"Position Mode:":<20}{args.bounded_memory_position_mode:<20}')
            print(f'  {"Fusion Mode:":<20}{args.bounded_memory_fusion_mode:<20}')
            print(f'  {"Far Context:":<20}{args.bounded_memory_far_context_mode:<20}')
            print(f'  {"Residual Bound:":<20}{args.bounded_memory_residual_bound:<20}')
            print(f'  {"Sampled Origins:":<20}{args.bounded_memory_sampled_far_origins:<20}{"Far Loss Weight:":<20}{args.bounded_memory_far_loss_weight:<20}')
            print(f'  {"Gradient Mode:":<20}{args.bounded_memory_gradient_mode:<20}')
            print(f'  {"Future Self-Attn:":<20}{"none":<20}')
        if args.model == 'RecurrentQueryDirectAR':
            print(f'  {"Fine Patch Len:":<20}{args.split_concat_fine_patch_len:<20}{"Direct K:":<20}{args.dense_ar_roll_patches:<20}')
            print(f'  {"Retrieval Cycles:":<20}{args.recurrent_query_steps:<20}{"Position Mode:":<20}{args.recurrent_query_position_mode:<20}')
            print(f'  {"Read Mode:":<20}{args.recurrent_query_read_mode:<20}{"Read Temperature:":<20}{args.recurrent_query_temperature:<20}')
            print(f'  {"Readout Mode:":<20}{args.recurrent_query_output_mode:<20}{"Future Self-Attn:":<20}{"none":<20}')
        if args.model == 'SlidingStateDirectAR':
            selected_stride = (
                args.recon_tail_fine_patch_len
                * args.recon_tail_group_patches
            )
            print(f'  {"State Read Stride:":<20}{selected_stride:<20}{"Direct Position:":<20}{args.direct_position_encoding:<20}')
    if args.model == 'GRUSlotAR':
        print(f'  {"AR Patch Len:":<20}{args.ar_patch_len:<20}{"GRU Slots:":<20}{args.ar_gru_slots:<20}')
        print(f'  {"Dense Roll Patches:":<20}{args.dense_ar_roll_patches:<20}{"Dense Loss Space:":<20}{args.dense_ar_loss_space:<20}')
        print(f'  {"AR Norm Eps:":<20}{args.ar_norm_eps:<20}{"Transformer Norm:":<20}{args.ar_transformer_norm:<20}')
        print(f'  {"Attention Mode:":<20}{"causal":<20}')
    if args.model == 'AutoTimesOperatorAR':
        print(f'  {"AutoTimes Mode:":<20}{args.external_autotimes_mode:<20}{"Native Token Len:":<20}{args.external_autotimes_token_len:<20}')
        print(f'  {"AutoTimes MLP Hidden:":<20}{args.external_autotimes_mlp_hidden:<20}{"AutoTimes MLP Layers:":<20}{args.external_autotimes_mlp_layers:<20}')
        print(f'  {"Accepted Checkpoint:":<20}{args.external_autotimes_pretrained}')
    if args.model == 'RetrievalMambaAR':
        retrieval_lags = '/'.join(
            str(value) for value in args.ssm_retrieval_lags)
        print(f'  {"AR Patch Len:":<20}{args.ar_patch_len:<20}{"Fine Patch Len:":<20}{args.split_concat_fine_patch_len:<20}')
        print(f'  {"Commit Patches:":<20}{args.dense_ar_roll_patches:<20}{"Loss Space:":<20}{args.dense_ar_loss_space:<20}')
        print(f'  {"Dense Loss Type:":<20}{args.dense_ar_loss_type:<20}{"Huber Delta:":<20}{args.dense_ar_huber_delta:<20}')
        print(f'  {"Retrieval Mode:":<20}{args.ssm_retrieval_mode:<20}{"Retrieval Lags:":<20}{retrieval_lags:<20}')
    if args.model == 'AddressableMambaAR':
        print(f'  {"AR Patch Len:":<20}{args.ar_patch_len:<20}{"Fine Patch Len:":<20}{args.split_concat_fine_patch_len:<20}')
        print(f'  {"Commit Patches:":<20}{args.dense_ar_roll_patches:<20}{"Loss Space:":<20}{args.dense_ar_loss_space:<20}')
        print(f'  {"Dense Loss Type:":<20}{args.dense_ar_loss_type:<20}{"Huber Delta:":<20}{args.dense_ar_huber_delta:<20}')
        print(f'  {"Address Mode:":<20}{args.addressable_mamba_mode:<20}{"Address Key Dim:":<20}{args.addressable_mamba_key_dim:<20}')
        print(f'  {"Address Scale:":<20}{args.addressable_mamba_scale_init:<20}{"Address Gate:":<20}{args.addressable_mamba_gate_init:<20}')
    if args.model == 'HorizonCrossTransformer':
        print(f'  {"Segment Len:":<20}{args.seg_len:<20}{"Channel Embedding:":<20}{args.horizon_cross_channel_embedding:<20}')
        print(f'  {"History Attention:":<20}{"bidirectional":<20}{"Future Self-Attn:":<20}{"none":<20}')
        print(f'  {"Forecast Anchor:":<20}{args.horizon_cross_anchor:<20}')
    if args.model == 'SegRNNMemoryTransformer':
        print(f'  {"Segment Len:":<20}{args.seg_len:<20}{"Channel Embedding:":<20}{args.horizon_cross_channel_embedding:<20}')
        print(f'  {"Local Segments:":<20}{args.segrnn_memory_local_segments:<20}{"Memory Scale Init:":<20}{args.segrnn_memory_scale_init:<20}')
        print(f'  {"Memory Max Scale:":<20}{args.segrnn_memory_max_scale:<20}{"Freeze SegRNN Base:":<20}{args.segrnn_memory_freeze_base:<20}')
        print(f'  {"History Memory:":<20}{"bidirectional":<20}{"Future Self-Attn:":<20}{"none":<20}')
    if args.model == 'AddressableStateDirect':
        print(f'  {"Segment Len:":<20}{args.seg_len:<20}{"Fine Patch Len:":<20}{args.addressable_state_fine_patch_len:<20}')
        print(f'  {"Future State Mode:":<20}{args.addressable_state_mode:<20}{"Channel Embedding:":<20}{args.horizon_cross_channel_embedding:<20}')
        print(f'  {"Read Temperature:":<20}{args.addressable_state_temperature:<20}{"NULL Margin:":<20}{args.addressable_state_null_margin:<20}')
        print(f'  {"Max Read Residual:":<20}{args.addressable_state_max_residual:<20}')
        print(f'  {"Freeze State Base:":<20}{args.addressable_state_freeze_base:<20}{"State Checkpoint:":<20}{args.addressable_state_pretrained}')
    if args.model == 'PersistentSuccessorAR':
        print(f'  {"AR Patch Len:":<20}{args.ar_patch_len:<20}{"Direct K:":<20}{args.tail_ar_patches:<20}')
        print(f'  {"Commit K:":<20}{args.tail_ar_roll_patches:<20}{"Successor Mode:":<20}{args.persistent_successor_mode:<20}')
        print(f'  {"History Weight:":<20}{args.persistent_successor_history_weight:<20}{"Loss Mode:":<20}{args.persistent_successor_loss_mode:<20}')
        print(f'  {"Far Weight:":<20}{args.persistent_successor_far_weight:<20}{"Gradient Group:":<20}{args.persistent_successor_gradient_group:<20}')
        print(f'  {"Far Samples:":<20}{args.persistent_successor_far_sample_count:<20}{"Stats Mode:":<20}{args.persistent_successor_stats_mode:<20}')
        print(f'  {"Retained Real:":<20}{args.persistent_successor_retained_real_patches:<20}')
        print(f'  {"Read Temperature:":<20}{args.persistent_successor_temperature:<20}{"NULL Margin:":<20}{args.persistent_successor_null_margin:<20}')
        print(f'  {"Residual Scale:":<20}{args.persistent_successor_residual_scale:<20}{"Residual Max:":<20}{args.persistent_successor_residual_max_scale!s:<20}')
        print(f'  {"Residual Gate:":<20}{args.persistent_successor_residual_gate:<20}')
    if args.model == 'SemigroupFlow':
        print(f'  {"Base Patch:":<20}{args.semigroup_base_patch_len:<20}{"History Patch:":<20}{args.semigroup_history_patch_len:<20}')
        print(f'  {"History Stride:":<20}{args.semigroup_history_stride:<20}{"State Dim:":<20}{args.semigroup_state_dim:<20}')
        print(f'  {"Flow Mode:":<20}{args.semigroup_flow_mode:<20}{"Flow Anchor:":<20}{args.semigroup_anchor:<20}')
        print(f'  {"Flow Loss Space:":<20}{args.semigroup_loss_space:<20}')
    if args.model == 'AdaptiveTailAR':
        route_dist = '/'.join(
            f'{value:g}' for value in args.adaptive_router_target_dist)
        print(f'  {"Fine Patch Len:":<20}{args.adaptive_fine_patch_len:<20}{"Router Bias Rate:":<20}{args.adaptive_router_bias_rate:<20}')
        print(f'  {"Router Target C/F/N:":<20}{route_dist:<20}')
    if args.model == 'ConvTailAR':
        print(f'  {"Causal Conv Kernel:":<20}{args.conv_tail_kernel_size:<20}')
    if args.model == 'ValueMixTailAR':
        print(f'  {"Value Mixer Hidden:":<20}{args.value_mixer_hidden_multiplier:<20}')
    if args.model == 'SwinTailAR':
        print(f'  {"Swin Window:":<20}{args.swin_tail_window_size:<20}{"Swin Shift:":<20}{args.swin_tail_shift_size:<20}')
        print(f'  {"History Prepass:":<20}{args.swin_tail_history_prepass:<20}{"Stage1 Dim:":<20}{args.swin_tail_stage1_dim:<20}')
        print(f'  {"Stage1 Heads:":<20}{args.swin_tail_stage1_heads:<20}')
    if args.model == 'SplitConcatTailAR':
        print(f'  {"Fine Patch Len:":<20}{args.split_concat_fine_patch_len:<20}')
    if args.model in ['SSMFuseTailAR', 'SSMFuseDenseAR']:
        print(f'  {"SSM Fine Patch:":<20}{args.ssm_fuse_fine_patch_len:<20}')
    if args.model == 'FiniteSSMDenseAR':
        print(f'  {"Finite SSM Memory:":<20}{args.finite_ssm_memory_patches:<20}')
    if args.model == 'FiniteSSM24DenseAR':
        print(f'  {"Fine Patch Len:":<20}{args.finite_ssm_fine_patch_len:<20}{"Finite SSM Memory:":<20}{args.finite_ssm_memory_patches:<20}')
        print(f'  {"Finite Length Mode:":<20}{args.finite_ssm_length_mode:<20}')
    if args.model == 'SlidingWindowDenseAR':
        print(f'  {"Sliding Fine Len:":<20}{args.sliding_window_fine_patch_len:<20}{"Window Projection:":<20}{args.sliding_window_projection:<20}')
    if args.model == 'ProgressiveConcatAR':
        schedule = ','.join(
            str(value) for value in args.progressive_concat_sizes)
        print(f'  {"Progressive Fine:":<20}{args.progressive_concat_fine_patch_len:<20}{"Concat Schedule:":<20}{schedule:<20}')
        print(f'  {"Concat Stride:":<20}{args.progressive_concat_stride:<20}{"Max Canvas:":<20}{args.progressive_concat_max_patches:<20}')
    if args.model == 'LatentQueryAR':
        print(f'  {"Fine Patch Len:":<20}{args.finite_ssm_fine_patch_len:<20}{"Finite SSM Memory:":<20}{args.finite_ssm_memory_patches:<20}')
        print(f'  {"Latent Queries:":<20}{args.latent_query_states:<20}{"History State Wt:":<20}{args.latent_history_state_weight:<20}')
        print(f'  {"Future State Wt:":<20}{args.latent_future_state_weight:<20}{"Point Loss Wt:":<20}{args.latent_point_weight:<20}')
    if args.model == 'EntropyTailAR':
        print(f'  {"Entropy Quantile:":<20}{args.entropy_patch_quantile:<20}{"Entropy Max Patch:":<20}{args.entropy_max_patch_len:<20}')
        print(f'  {"Entropy Encoder:":<20}{args.entropy_encoder_type:<20}{"Entropy Monotonic:":<20}{args.entropy_monotonic:<20}')
        print(f'  {"Entropy Checkpoint:":<20}{args.entropy_model_checkpoint}')
    print()

    print("\033[1m" + "Run Parameters" + "\033[0m")
    print(f'  {"Num Workers:":<20}{args.num_workers:<20}{"Itr:":<20}{args.itr:<20}')
    print(f'  {"Train Epochs:":<20}{args.train_epochs:<20}{"Batch Size:":<20}{args.batch_size:<20}')
    print(f'  {"Seed:":<20}{args.seed:<20}{"Loader Seed:":<20}{args.loader_seed:<20}')
    print(f'  {"Patience:":<20}{args.patience:<20}{"Learning Rate:":<20}{args.learning_rate:<20}')
    print(f'  {"Checkpoint Select:":<20}{args.checkpoint_selection:<20}')
    print(f'  {"Des:":<20}{args.des:<20}{"Loss:":<20}{args.loss:<20}')
    print(f'  {"Lradj:":<20}{args.lradj:<20}{"Use Amp:":<20}{args.use_amp:<20}')
    print()

    print("\033[1m" + "GPU" + "\033[0m")
    print(f'  {"Use GPU:":<20}{args.use_gpu:<20}{"GPU:":<20}{args.gpu:<20}')
    print(f'  {"Use Multi GPU:":<20}{args.use_multi_gpu:<20}{"Devices:":<20}{args.devices:<20}')
    print()

    print("\033[1m" + "De-stationary Projector Params" + "\033[0m")
    p_hidden_dims_str = ', '.join(map(str, args.p_hidden_dims))
    print(f'  {"P Hidden Dims:":<20}{p_hidden_dims_str:<20}{"P Hidden Layers:":<20}{args.p_hidden_layers:<20}') 
    print()
