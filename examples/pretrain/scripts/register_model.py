from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, AutoTokenizer

from model.doge import DogeConfig
from model.doge import DogeForCausalLM, DogeForSequenceClassification, DogeModel, DogePreTrainedModel


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
doge = DogeForCausalLM.from_pretrained("./checkpoint-10000")
doge.push_to_hub("JingzeShi/Doge-76M")
# doge.save_pretrained("./results/Doge_Eval_200M/checkpoint-2000")

tokenizer = AutoTokenizer.from_pretrained("./checkpoint-10000")
tokenizer.push_to_hub("JingzeShi/Doge-76M")
# tokenizer.save_pretrained("./results/Doge_Eval_200M/checkpoint-2000")