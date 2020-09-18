from train import SDDR

if __name__ == '__main__':
    # config = get_config() load config file
    sddr = SDDR(config)
    sddr.train()