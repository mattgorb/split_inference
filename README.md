# U-Shaped Split Inference

See this page for more details on split inference: [About Secure Llama](https://d2lt92f6mmvv10.cloudfront.net/about.html)

The implementation of this model is in the llama3_7b folder, see this folder for more details. 

The debug folder has scripts I used during initial development.  For example I used llama3_7b_model_split.py to test the split model in one file.  This script is a good way to understand split inference in PyTorch/NCCL since the code is all in a single script.  

I implemented the gpt2 first since it is a smaller model and is easier to work with. It can probably be deleted since gpt2 sucks, but I'm leaving it for now. 



[comment]: <img src="llama3_7b/client/frontend/image.jpg" alt="Alt text"  style="max-width:40%; height:auto;">