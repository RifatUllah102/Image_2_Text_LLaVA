# Image_2_Text_LLaVA
 Part of my Summer Research

 #Steps to execute the script:
 1. Create an environment and install all the libraries from requirements.txt file.
 2. download the pre-trained model from the given link "https://github.com/haotian-liu/LLaVA/blob/main/docs/MODEL_ZOO.md", and make sure the script is loading the model properly.
 3. If you face any kind of error such as "TypeError: LlavaLlamaForCausalLM.forward() got an unexpected keyword argument 'cache_position'", it is possible that you will face this error. In that case, you can check https://github.com/huggingface/transformers/issues/29426 or add cache_position=None to the forward() method in Class LlavaLlamaForCausalLM to the particular file.
