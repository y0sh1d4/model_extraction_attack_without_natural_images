# Train Victim model
python3 train_victim_model.py --multirun \
    victim=mnist,fashion_mnist \
    victim.model=small,medium,large
