# python3 train_victim_model.py --multirun \
#     victim=mnist,fashion_mnist \
#     victim.model=medium

# python3 model_extraction_attack.py --multirun \
#     victim=mnist,fashion_mnist \
#     victim.model=medium \
#     attack=mnist,fashion_mnist,fractaldb_1k,fractaldb_60 \
#     attack.model=small,medium,large \
#     attack.Early_stopping_patience=10 \
#     attack.strategy=random_query

# python3 create_adversarial_examples.py --multirun \
#     victim=mnist,fashion_mnist \
#     victim.model=medium \
#     attack=mnist,fashion_mnist,fractaldb_1k,fractaldb_60 \
#     attack.model=small,medium,large \
#     create_AEs.round="'range(100)'"

python3 model_extraction_attack.py --multirun \
    victim=fashion_mnist \
    victim.model=medium \
    attack=fractaldb_60 \
    attack.model=large \
    attack.Early_stopping_patience=10 \
    attack.strategy=random_query
