## `create_gifs.py`

Much faster on the GPU
```
python src/scripts/create_gifs.py --num_frames 10 --img_path imgs/example-data/chair4.png --model_path train_results/original/chairs.pt --shift all --save_name org

python src/scripts/create_gifs.py --num_frames 10 --shift all --model_path train_results/2023-05-10_12-31_roto_lr2e-4/best_model.pt --img_path imgs/example-data/half_rot_train.png --save_name rot

python src/scripts/create_gifs.py --num_frames 10 --shift all --model_path train_results/2023-05-14_16-54_trans_lr2e-4/best_model.pt --img_path imgs/example-data/half_trans_train.png --save_name trans

python src/scripts/create_gifs.py --num_frames 10 --shift all --model_path train_results/2023-05-25_15-20_one_rototrans_lr2e-5_64/best_model.pt --img_path imgs/example-data/one_train.png --save_name one
```