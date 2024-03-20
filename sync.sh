rsync -auvP /home/zhangyixiang/wfp/data/ 2080server:/data/users/zhangyixiang/wfp_zyx_new/data/ --exclude=captured --exclude=wang
rsync -auvP 2080server:/data/users/zhangyixiang/wfp_zyx_new/data/ /home/zhangyixiang/wfp/data/ --exclude=captured --exclude=wang
rsync -auvP /home/zhangyixiang/wfp/run/ 2080server:/data/users/zhangyixiang/wfp_zyx_new/run/ --exclude=captured --exclude=wang
rsync -auvP 2080server:/data/users/zhangyixiang/wfp_zyx_new/run/ /home/zhangyixiang/wfp/run/ --exclude=captured --exclude=wang
