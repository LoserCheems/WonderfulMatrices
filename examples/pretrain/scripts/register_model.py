from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, AutoTokenizer

from wonderful_matrices.doge import DogeConfig
from wonderful_matrices.doge import DogeForCausalLM, DogeForSequenceClassification, DogeModel, DogePreTrainedModel


# 注册模型
# Register the model
AutoConfig.register("doge", DogeConfig)
AutoModel.register(DogeConfig, DogeModel)
AutoModelForCausalLM.register(DogeConfig, DogeForCausalLM)
DogeConfig.register_for_auto_class()
DogeModel.register_for_auto_class("AutoModel")
DogeForCausalLM.register_for_auto_class("AutoModelForCausalLM")

# 上传到hub
# Push to hub
doge = DogeForCausalLM.from_pretrained("./results/doge_197M/checkpoint-13400")
# doge.push_to_hub("JingzeShi/Doge-76M")
doge.save_pretrained("./results/Doge_Eval/197M/checkpoint-13400")

tokenizer = AutoTokenizer.from_pretrained("./examples/tokenizer")
# tokenizer.push_to_hub("JingzeShi/Doge-76M")
tokenizer.save_pretrained("./results/Doge_Eval/197M/checkpoint-13400")