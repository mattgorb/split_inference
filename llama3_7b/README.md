There are three folders each with specific terraform.  I run them in this order: 

1. peering
2. client
3. server

I run everything in the file ```deploy_all.sh```.  This file is kind of janky and may not work another time around, but the overall logic is there. 

1. In peering terraform I create a server VPC_ID`.  
2. I add the ```server_vpc_id``` to the server terraform (terraform.tfvars) and the client terraform vars.  The client needs the ```server_vpc_id``` to create a ```vpc_peering_connection``` between the client and server. 
3. The client terraform creates a server with a peering connection and shares its private IP with the server.  
4. In the ```deploy_client.sh``` the client code is in the client/app folder.  I use main.py (API) and model_executor.py (model). See the Dockerfile and docker-compose.yml for additional details. 
5. In the ```deploy_client.sh``` I also deploy the frontend.  The frontend code is in the frontend folder and uses the template file index.html.tpl to fill in the private IP.  See the terraform for more details. 
6. Finally I deploy the server in the server folder ```deploy_server.sh```

The cool part of the code is in client/app/model_executor.py (```model_communication```) and server/app/model_backend.py (```generate```)