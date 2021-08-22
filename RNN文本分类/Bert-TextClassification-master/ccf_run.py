# coding=utf-8
from main import main


if __name__ == "__main__":

    model_name = "BertOrigin"

    label_list = ['0', '1','2']
    data_dir = "ccf_data"
    output_dir = ".ccf_output/" 
    cache_dir = ".ccf_cache"
    log_dir = ".ccf_log/" 

    model_times = "model_1/"   # 第几次保存的模型，主要是用来获取最佳结果

    # bert-base
    bert_vocab_file = "bert-base-chinese-vocab.txt"
    bert_model_dir = "bert-base-chinese/"

    # # bert-large
    # bert_vocab_file = "/search/hadoop02/suanfa/songyingxin/pytorch_Bert/bert-large-uncased-vocab.txt"
    # bert_model_dir = "/search/hadoop02/suanfa/songyingxin/pytorch_Bert/bert-large-uncased"

    if model_name == "BertOrigin":
        from BertOrigin import args

    elif model_name == "BertCNN":
        from BertCNN import args

    elif model_name == "BertATT":
        from BertATT import args

    elif model_name == "BertRCNN":
        from BertRCNN import args

    elif model_name == "BertCNNPlus":
        from BertCNNPlus import args

    config = args.get_args(data_dir, output_dir, cache_dir,
                           bert_vocab_file, bert_model_dir, log_dir)

    main(config, config.save_name, label_list)
        

