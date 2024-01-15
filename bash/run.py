import os

script = '''#!/bin/bash

WANDB_MODE=offline python -m torch.distributed.run --master_port {} --nproc_per_node=1 unlearn.py \
            --unlearn_method {} \
            --backbone {} \
            --task {} \
            --df_size {} \
            --cfg-path configs/original/{}/{}.yaml
'''


i = 0
for m in ['ft', 'neggrad', 'dtd', 'vlul']:
    for b in ['albef', 'blip']:
        for t in ['retrieval_flickr30k', 'nlvr', 'snli_ve']:
            for s in range(1,6):
                config = [m, b, t, s*1000] + [29510+i] + [m, b, t, s*1000] + [b, t]
                with open(f'{m}_{b}_{t}_{s}.sh', 'w') as f:
                    f.write(script.format(*config))

                os.system(f'bash {m}_{b}_{t}_{s}.sh')
                # raise
                i += 1
