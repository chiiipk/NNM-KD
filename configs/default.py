"""
Default configuration for NNM-KD v3.
All hyperparameters are centralised here — import CFG everywhere else.
"""

CFG = dict(
    # ── Models ──
    teacher_id        = "Qwen/Qwen2.5-Math-1.5B-Instruct",
    student_id        = "Qwen/Qwen2.5-0.5B",

    # ── Dataset ──
    train_dataset     = "VoCuc/MetaMathQA-50k-256",
    max_prompt_length = 256,
    max_length        = 768,          # v3: tăng từ 512 → 768 (giữ full reasoning)

    # ── KL config ──
    kl_temp_max       = 3.0,           # v3: temperature annealing start
    kl_temp_min       = 1.0,           # v3: temperature annealing end
    ce_weight         = 0.3,           # v3: giảm từ 0.5 (KL dominant)
    kl_weight         = 0.6,           # v3: tăng từ 0.3
    kl_mode           = "skewed_forward",
    kl_skew_lam       = 0.1,
    top_k_logits      = 50,            # v3: top-K logit KD

    # ── Token filtering (STAPO + Beyond 80/20) ──
    s2t_tau_p         = 0.01,          # spurious token prob threshold
    s2t_tau_h         = 0.5,           # spurious token entropy threshold
    high_ent_rho      = 0.3,           # keep top-30% entropy tokens for KL

    # ── Difficulty-aware weighting ──
    use_difficulty_weight   = True,
    difficulty_early_layer  = 2,       # early layer index for JSD

    # ── NNM config ──
    lambda_nnm        = 0.10,
    nnm_warmup        = 100,
    K_centroids       = 128,
    d_prime           = 256,
    eta_centroid      = 0.05,
    T_dead            = 50,
    sigma_layer       = 0.15,
    n_mid_layers      = 4,
    do_teacher_correction = True,
    tc_lambda         = 0.10,
    tc_steps          = 1,
    ns_iters          = 5,

    # ── Training config ──
    batch_size        = 1,
    grad_accum        = 16,
    lr                = 4e-5,
    epochs            = 1,
    warmup_ratio      = 0.08,
    max_grad_norm     = 1.0,
    weight_decay      = 0.01,
    max_train_batches = 5000,
    fp16              = True,
    use_4bit          = True,
    save_dir          = "./nnm_kd_v3_outputs",
    log_every         = 20,
    n_eval_gsm8k      = 99999,
    n_eval_math       = 99999,

    # ── On-policy Phase 2 ──
    do_on_policy        = True,
    on_policy_interval  = 200,        # generate every N steps
    on_policy_batch     = 4,          # samples per on-policy step
    on_policy_max_new   = 512,        # max new tokens for student generation
    on_policy_weight    = 0.3,        # weight for on-policy KL loss

    # ── Teacher filtering ──
    filter_by_teacher           = True,
    teacher_filter_batch_size   = 16,
    teacher_filter_max_samples  = 5000,
)

# ── Qwen chat template (kept verbatim) ──
QWEN_CHAT_TEMPLATE = (
    '{%- if tools %}    {{- \'<|im_start|>system\\n\' }}    {%- if messages[0][\'role\'] == \'system\' %}        {{- messages[0][\'content\'] }}    {%- else %}        {{- \'You are a helpful assistant.\' }}    {%- endif %}    {{- "\\n\\n# Tools\\n\\nYou may call one or more functions to assist with the user query.\\n\\nYou are provided with function signatures within <tools></tools> XML tags:\\n<tools>" }}    {%- for tool in tools %}        {{- "\\n" }}        {{- tool | tojson }}    {%- endfor %}    {{- "\\n</tools>\\n\\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\\n<tool_call>\\n{\\"name\\": <function-name>, \\"arguments\\": <args-json-object>}\\n</tool_call><|im_end|>\\n" }} {%- else %}    {%- if messages[0][\'role\'] == \'system\' %}        {{- \'<|im_start|>system\\n\' + messages[0][\'content\'] + \'<|im_end|>\\n\' }}    {%- else %}        {{- \'<|im_start|>system\\nYou are a helpful assistant.<|im_end|>\\n\' }}    {%- endif %} {%- endif %} {%- for message in messages %}    {%- if (message.role == "user") or (message.role == "system" and not loop.first) or (message.role == "assistant" and not message.tool_calls) %}        {{- \'<|im_start|>\' + message.role + \'\\n\' + message.content + \'<|im_end|>\' + \'\\n\' }}    {%- elif message.role == "assistant" %}        {{- \'<|im_start|>\' + message.role }}        {%- if message.content %}            {{- \'\\n\' + message.content }}        {%- endif %}        {%- for tool_call in message.tool_calls %}            {%- if tool_call.function is defined %}                {%- set tool_call = tool_call.function %}            {%- endif %}            {{- \'\\n<tool_call>\\n{"name": "\' }}            {{- tool_call.name }}            {{- \'", "arguments": \' }}            {{- tool_call.arguments | tojson }}            {{- \'}\\n</tool_call>\' }}        {%- endfor %}        {{- \'<|im_end|>\\n\' }}    {%- elif message.role == "tool" %}        {%- if (loop.index0 == 0) or (messages[loop.index0 - 1].role != "tool") %}            {{- \'<|im_start|>user\' }}        {%- endif %}        {{- \'\\n<tool_response>\\n\' }}        {{- message.content }}        {{- \'\\n</tool_response>\' }}        {%- if loop.last or (messages[loop.index0 + 1].role != "tool") %}            {{- \'<|im_end|>\\n\' }}        {%- endif %}    {%- endif %} {%- endfor %} {%- if add_generation_prompt %}    {{- \'<|im_start|>assistant\\n\' }} {%- endif %}'
)

SYSTEM_PROMPT = "Put your final answer within \\boxed{}."

SEED = 42
