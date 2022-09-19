FROM pytorch/pytorch:1.8.0-cuda11.1-cudnn8-runtime
RUN apt-get update
RUN apt-get install -y git wget 
RUN git clone https://github.com/alibaba/easyrobust.git
RUN cd easyrobust
RUN pip install -e . -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip install -r requirements/optional.txt -i https://pypi.tuna.tsinghua.edu.cn/simple