class PRMConfig:
    """Configuration class for PRM hyperparameters and settings"""
    # MC config
    model_name:             str = "Qwen/Qwen2.5-Math-7B"    # "Qwen/Qwen2.5-Math-7B", "Qwen/Qwen2.5-Math-7B-Instruct" , "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B", "meta-llama/Llama-3.1-8B"
    max_new_tokens:         int = 512
    num_rollouts:           int = 8      
    samples_per_question:   int = 1
    use_llm:                bool = True  # Use llm for masking
    reward_type:            str = "cmi"  # ori, contri, mi, cmi
    # PRM Model config 
    hidden_size:        int = 512      # 256-1024 범위에서 적절
    num_layers:         int = 3        # 2-4 범위에서 적절
    dropout:            float = 0.2    # 0.1-0.3 범위에서 적절
    # PRMTrainer config 
    batch_size:         int = 16       # 12 → 16으로 증가 (더 안정적)
    learning_rate:      float = 3e-4   # 5e-4 → 3e-4로 감소 (더 안정적)
    num_workers:        int = 4        # 적절
    weight_decay:       float = 1e-2   # 적절
    lr_scheduler:       str = "cosine" # 적절
    dataset_size:       int = 0
    warmup_steps:       int = 40       # 22 → 50으로 증가 (더 안정적)
    grad_clip:          float = 1.0    # 적절
    epochs:             int = 20       # 25 → 15로 감소 (early stopping 고려)
    # Misc config
    use_wandb:          bool = True
    wandb_project:      str = "mc_prm"
    run_name:           str = "test_400_0715"
    checkpoint_dir:     str = "./checkpoints/0715/contri"
    seed:               int = 42
    # Inference config
    num_candidates:     int = 4