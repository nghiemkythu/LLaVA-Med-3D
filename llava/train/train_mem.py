import wandb
wandb.login(key="b40bed659a04b35785b0534935b459f944bdf6e8")

from llava.train.llama_flash_attn_monkey_patch import replace_llama_attn_with_flash_attn

replace_llama_attn_with_flash_attn()

from llava.train.train import train

if __name__ == "__main__":
    train()

#from llava.train.train import train

#if __name__ == "__main__":
#    train(attn_implementation="flash_attention_2")
