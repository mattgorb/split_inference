## General Overview: 
There are three folders each with specific terraform.  I run them in this order: 

1. peering
2. client
3. server

I run everything in the file ```deploy_all.sh```.  This file is kind of buggy and may not work another time around, but the overall logic is there. Here is an overview of the logic:

1. In peering terraform I create a server VPC_ID.  
2. I add the ```server_vpc_id``` to the server terraform terraform.tfvars and the client terraform.tfvars.  The client needs the ```server_vpc_id``` to create a ```vpc_peering_connection``` between the client and server. 
3. The client terraform creates a server with a peering connection and shares its private IP with the server. I share the private IP in the bash script -- in a real world implementation this would need to be done via an s3 bucket or another secure remote data sharing method.  
4. In the ```deploy_client.sh``` the client code is in the client/app folder.  I use main.py (API) and model_executor.py (model). See the Dockerfile and docker-compose.yml for additional details. 
5. In the ```deploy_client.sh``` I also deploy the frontend.  The frontend code is in the frontend folder and uses the template file index.html.tpl to fill in the private IP.  See the terraform for more details. 
6. Finally I deploy the server in the server folder ```deploy_server.sh```

The cool part of the code is in client/app/model_executor.py (```model_communication```) and server/app/model_backend.py (```generate```)


## Stuff to do: 

**Frontend**

- Add domain name to cloudfront URL
- Move conversation history processing to backend potentially (not sure how history processing is typically done in industry, look this up). REDIS? needs to be for specific user/session
- Improve chat UI to include code blocks for markdown etc. 
- Login, account setup, and typical boring frontend stuff

**Backend (Lots to do)**
- Why does text generation slow down when the conversation gets slightly longer?
- Clean up code/endpoints. 
- Add configuration and separate API from model better.
- Add better logging
- Add Transformer caching -- not completely sure how this works but could help immensely with generation time.  Look up "inference time caching for transformers"
- eventually transition to kubernetes so I can add other services 
-  How to scale multiple client servers to multiple model servers?  (1:1 seems infeasible at scale). Test different usages -- current GPU can only run one user.  Adding autoscaling will be weird weird with multiple machines performing model inference. 


**Distributed Model Inference**
- Scale to bigger model--llama3 has a 70b version and will need at least two A100 GPUs
- Scale to enormous model in 400b range. 
- Both above will both require using distributed neural network techniques such as  pipeline parallelism and/or tensor parallelism. Giant LLAMAs on HuggingFace: meta-llama/Llama-3.1-70B-Instruct, meta-llama/Llama-3.1-405B-Instruct


**Architecture**
- IMPORTANT*** Stitch together Client/Server backends better--right now terraform first creates a server VPC for peering with client server, then creates client, then creates server, all in few scripts.  In a real world scenario I'd want to share relevant data (VPC peering connection, private IP) via AWS buckets and across remote environments, rather than running simple bash files. We need to handle this for an autoscaling scenario as well.  
- Restructure code base

**Future**
- Move backend to a Trusted Execution Environment (TEE)--This would add a ton of security but could be a nightmare to implement because the NCCL communication might not be capable.  AWS has an option for this (AWS Enclave) 
- Move to different cloud providers (GCP/AZURE)