class Config:
    def __init__(self, ckpt, cls_ckpt, cls_ckpt_hparams, clean_cls_ckpt, clean_cls_ckpt_hparams, distill_ckpt, distill_ckpt_hparams,
                 ckpt_has_cls, ckpt_has_clean_cls, validate, subset_train_as_val, validate_on_train, validate_on_test,
                 limit_train_batches, batch_size, constant_val_len, accumulate_grad, grad_clip, lr_multiplier,
                 check_grad, no_lr_scheduler, checkpoint_layers, max_steps, max_epochs, lr, check_val_every_n_epoch,
                 limit_val_batches, fid_early_stop, val_loss_es, val_check_interval, ckpt_iterations, random_sequences,
                 taskiran_seq_path, dataset_type, mel_enhancer, overfit, promoter_dataset, toy_simplex_dim, toy_num_cls,
                 toy_num_seq, toy_seq_len, num_workers, cls_guidance, binary_guidance, oversample_target_class, target_class,
                 all_class_inference, cls_free_noclass_ratio, cls_free_guidance, probability_addition, adaptive_prob_add,
                 vectorfield_addition, probability_tilt, score_free_guidance, guidance_scale, analytic_cls_score, scale_cls_score,
                 allow_nan_cfactor, model, cls_model, clean_cls_model, clean_data, mode, simplex_spacing, prior_pseudocount,
                 dropout, num_cnn_stacks, num_layers, hidden_dim, self_condition_ratio, prior_self_condition, no_token_dropout,
                 time_embed, fix_alpha, alpha_scale, alpha_max, cls_expanded_simplex, simplex_encoding_dim, flow_temp,
                 val_pred_type, num_integration_steps, no_tqdm, print_freq, wandb, run_name, commit):
        self.ckpt = ckpt
        self.cls_ckpt = cls_ckpt
        self.cls_ckpt_hparams = cls_ckpt_hparams
        self.clean_cls_ckpt = clean_cls_ckpt
        self.clean_cls_ckpt_hparams = clean_cls_ckpt_hparams
        self.distill_ckpt = distill_ckpt
        self.distill_ckpt_hparams = distill_ckpt_hparams
        self.ckpt_has_cls = ckpt_has_cls
        self.ckpt_has_clean_cls = ckpt_has_clean_cls
        self.validate = validate
        self.subset_train_as_val = subset_train_as_val
        self.validate_on_train = validate_on_train
        self.validate_on_test = validate_on_test
        self.limit_train_batches = limit_train_batches
        self.batch_size = batch_size
        self.constant_val_len = constant_val_len
        self.accumulate_grad = accumulate_grad
        self.grad_clip = grad_clip
        self.lr_multiplier = lr_multiplier
        self.check_grad = check_grad
        self.no_lr_scheduler = no_lr_scheduler
        self.checkpoint_layers = checkpoint_layers
        self.max_steps = max_steps
        self.max_epochs = max_epochs
        self.lr = lr
        self.check_val_every_n_epoch = check_val_every_n_epoch
        self.limit_val_batches = limit_val_batches
        self.fid_early_stop = fid_early_stop
        self.val_loss_es = val_loss_es
        self.val_check_interval = val_check_interval
        self.ckpt_iterations = ckpt_iterations
        self.random_sequences = random_sequences
        self.taskiran_seq_path = taskiran_seq_path
        self.dataset_type = dataset_type
        self.mel_enhancer = mel_enhancer
        self.overfit = overfit
        self.promoter_dataset = promoter_dataset
        self.toy_simplex_dim = toy_simplex_dim
        self.toy_num_cls = toy_num_cls
        self.toy_num_seq = toy_num_seq
        self.toy_seq_len = toy_seq_len
        self.num_workers = num_workers
        self.cls_guidance = cls_guidance
        self.binary_guidance = binary_guidance
        self.oversample_target_class = oversample_target_class
        self.target_class = target_class
        self.all_class_inference = all_class_inference
        self.cls_free_noclass_ratio = cls_free_noclass_ratio
        self.cls_free_guidance = cls_free_guidance
        self.probability_addition = probability_addition
        self.adaptive_prob_add = adaptive_prob_add
        self.vectorfield_addition = vectorfield_addition
        self.probability_tilt = probability_tilt
        self.score_free_guidance = score_free_guidance
        self.guidance_scale = guidance_scale
        self.analytic_cls_score = analytic_cls_score
        self.scale_cls_score = scale_cls_score
        self.allow_nan_cfactor = allow_nan_cfactor
        self.model = model
        self.cls_model = cls_model
        self.clean_cls_model = clean_cls_model
        self.clean_data = clean_data
        self.mode = mode
        self.simplex_spacing = simplex_spacing
        self.prior_pseudocount = prior_pseudocount
        self.dropout = dropout
        self.num_cnn_stacks = num_cnn_stacks
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.self_condition_ratio = self_condition_ratio
        self.prior_self_condition = prior_self_condition
        self.no_token_dropout = no_token_dropout
        self.time_embed = time_embed
        self.fix_alpha = fix_alpha
        self.alpha_scale = alpha_scale
        self.alpha_max = alpha_max
        self.cls_expanded_simplex = cls_expanded_simplex
        self.simplex_encoding_dim = simplex_encoding_dim
        self.flow_temp = flow_temp
        self.val_pred_type = val_pred_type
        self.num_integration_steps = num_integration_steps
        self.no_tqdm = no_tqdm
        self.print_freq = print_freq
        self.wandb = wandb
        self.run_name = run_name
        self.commit = commit

# Example usage
AR_CONFIG = Config(
    ckpt='workdir/promo_lrar_sani_2024-01-31_10-46-33/epoch=69-step=24220-Copy1.ckpt',
    cls_ckpt=None,
    cls_ckpt_hparams=None,
    clean_cls_ckpt=None,
    clean_cls_ckpt_hparams=None,
    distill_ckpt=None,
    distill_ckpt_hparams=None,
    ckpt_has_cls=False,
    ckpt_has_clean_cls=False,
    validate=True,
    subset_train_as_val=False,
    validate_on_train=False,
    validate_on_test=True,
    limit_train_batches=None,
    batch_size=128,
    constant_val_len=None,
    accumulate_grad=1,
    grad_clip=1.0,
    lr_multiplier=1.0,
    check_grad=False,
    no_lr_scheduler=False,
    checkpoint_layers=False,
    max_steps=450000,
    max_epochs=100000,
    lr=0.0005,
    check_val_every_n_epoch=5,
    limit_val_batches=None,
    fid_early_stop=False,
    val_loss_es=False,
    val_check_interval=None,
    ckpt_iterations=None,
    random_sequences=False,
    taskiran_seq_path=None,
    dataset_type='argmax',
    mel_enhancer=False,
    overfit=False,
    promoter_dataset=False,
    toy_simplex_dim=4,
    toy_num_cls=3,
    toy_num_seq=1000,
    toy_seq_len=20,
    num_workers=4,
    cls_guidance=False,
    binary_guidance=False,
    oversample_target_class=False,
    target_class=0,
    all_class_inference=False,
    cls_free_noclass_ratio=0.3,
    cls_free_guidance=False,
    probability_addition=False,
    adaptive_prob_add=False,
    vectorfield_addition=False,
    probability_tilt=False,
    score_free_guidance=False,
    guidance_scale=0.5,
    analytic_cls_score=False,
    scale_cls_score=False,
    allow_nan_cfactor=False,
    model='650M',
    cls_model='cnn',
    clean_cls_model='cnn',
    clean_data=False,
    mode='lrar',
    simplex_spacing=1000,
    prior_pseudocount=2,
    dropout=0.0,
    num_cnn_stacks=1,
    num_layers=1,
    hidden_dim=128,
    self_condition_ratio=0,
    prior_self_condition=False,
    no_token_dropout=False,
    time_embed=False,
    fix_alpha=None,
    alpha_scale=2,
    alpha_max=8,
    cls_expanded_simplex=False,
    simplex_encoding_dim=64,
    flow_temp=1.0,
    val_pred_type='argmax',
    num_integration_steps=100,
    no_tqdm=False,
    print_freq=100,
    wandb=True,
    run_name='language_model',
    commit='2fcc07557c02220859d485bfab870e65de394bfb'
)