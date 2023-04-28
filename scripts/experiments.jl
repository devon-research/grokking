using DrWatson

function run_training(params)
    option_list = []
    for (key, value) in params
        push!(option_list, string("--", key))
        isnothing(value) && continue # continue if value is nothing
        push!(option_list, string(value))
    end
    run(`python scripts/train.py $option_list`)
end

param_options = Dict(
    # Relevant for the data and model...
    "random_seed" => [23093, 9082],
    "modular_base" => [97, 113],
    # Relevant for the data...
    "train_fraction" => [0.3],
    # Relevant for the model...
    "model" => ["GromovMLP"],
    "hidden_dim" => [512],
    # Relevant for the optimization...
    "optimizer" => ["Adam"],
    "loss_function" => ["CrossEntropy"],
    "learning_rate" => [0.001],
    "batch_size" => [128],
);

@time map(run_training, dict_list(param_options))