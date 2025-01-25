import os
import datetime
import logging
import torch.distributed as dist

# Enable verbose logging
logging.basicConfig(level=logging.DEBUG)

# Much longer timeout
timeout = datetime.timedelta(seconds=60)  # 5 minutes


# Ensure these match EXACTLY what's on master
os.environ['MASTER_ADDR'] = '3.91.13.68'
os.environ['MASTER_PORT'] = '29500'
os.environ['WORLD_SIZE'] = '2'
os.environ['RANK'] = '1'
os.environ['LOCAL_RANK'] = '0'
#os.environ['GLOO_SOCKET_FAMILY'] = 'IPv6'

#os.environ['GLOO_DEBUG'] = '1'
os.environ['GLOO_SOCKET_IFNAME'] = 'eno1'

print("CLIENT DEBUG INFO:")
print(f"MASTER_ADDR: {os.environ.get('MASTER_ADDR', 'not set')}")
print(f"MASTER_PORT: {os.environ.get('MASTER_PORT', 'not set')}")
print(f"WORLD_SIZE: {os.environ.get('WORLD_SIZE', 'not set')}")
print(f"RANK: {os.environ.get('RANK', 'not set')}")
print(f"LOCAL_RANK: {os.environ.get('LOCAL_RANK', 'not set')}")
print(f"GLOO_SOCKET_IFNAME: {os.environ.get('GLOO_SOCKET_IFNAME', 'not set')}")

os.environ[ "TORCH_DISTRIBUTED_DEBUG" ] = "DETAIL" 


#dist.init_process_group('gloo', init_method='env://')
print('hererere')





dist.init_process_group('nccl', init_method='env://')