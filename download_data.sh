cd benchmarks/data
# the download may cost long time ...
if [ ! -f imagenet-a.tar.gz ]; then
wget http://alisec-competition.oss-cn-shanghai.aliyuncs.com/xiaofeng/easy_robust/benchmark_datasets/imagenet-a.tar.gz
tar -zxvf imagenet-a.tar.gz
fi

if [ ! -f imagenet-r.tar.gz ]; then
wget http://alisec-competition.oss-cn-shanghai.aliyuncs.com/xiaofeng/easy_robust/benchmark_datasets/imagenet-r.tar.gz
tar -zxvf imagenet-r.tar.gz
fi

if [ ! -f imagenetv2.tar.gz ]; then
wget http://alisec-competition.oss-cn-shanghai.aliyuncs.com/xiaofeng/easy_robust/benchmark_datasets/imagenetv2.tar.gz
tar -zxvf imagenetv2.tar.gz
fi

if [ ! -f imagenet-sketch.tar.gz ]; then
wget http://alisec-competition.oss-cn-shanghai.aliyuncs.com/xiaofeng/easy_robust/benchmark_datasets/imagenet-sketch.tar.gz
tar -zxvf imagenet-sketch.tar.gz
fi

if [ ! -f imagenet-val.tar.gz ]; then
wget http://alisec-competition.oss-cn-shanghai.aliyuncs.com/xiaofeng/easy_robust/benchmark_datasets/imagenet-val.tar.gz
tar -zxvf imagenet-val.tar.gz
fi

if [ ! -f imagenet-style.tar.gz ]; then
wget http://alisec-competition.oss-cn-shanghai.aliyuncs.com/xiaofeng/easy_robust/benchmark_datasets/imagenet-style.tar.gz
tar -zxvf imagenet-style.tar.gz
fi

if [ ! -f imagenet-c.tar.gz ]; then
wget http://alisec-competition.oss-cn-shanghai.aliyuncs.com/xiaofeng/easy_robust/benchmark_datasets/imagenet-c.tar.gz
tar -zxvf imagenet-c.tar.gz
fi


# ObjectNet is optional, if you want to download please visit https://objectnet.dev/download.html