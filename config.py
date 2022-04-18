class Config:
    phase = ['train', 'test']
    prefix = ""
    seed = 2000
    Dataset = 'tiktok'
    out_log_dir = 'log'
    is_finetune = 0
    is_orthogonal = 0
    best_ckpt_path = 'runs/'

    BATCH_SIZE = 512
    learner = 'adam'
    learning_rate = 0.001
    weight_decay = 0
    max_seq_len = 50

    user_inter_num_interval = '[5,inf)'
    item_inter_num_interval = '[5,inf)'

    test_step = 100
    max_epoch = 200
    ks = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    topk = 50
    patience = 15

    n_layers = 2
    n_heads = 2
    k_interests = 20
    hidden_size = 64
    inner_size = 256
    hidden_dropout_prob = 0.5
    attn_dropout_prob = 0.5
    hidden_act = 'gelu'
    layer_norm_eps = 1e-12
    initializer_range = 0.02
    loss_type = 'CE'

    lmd = 0.1
    tau = 1
    sim = 'dot'
