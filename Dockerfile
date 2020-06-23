FROM geminihub.oa.com:80/yard/env:cuda10.0-py36-env-2.1
WORKDIR /data/user/
ADD . /data/user/
#RUN wget "http://download.devcloud.oa.com/iProxy.sh" -O iProxy.sh
#RUN source ./iProxy.sh -install
RUN pip install -i http://mirror-sng.oa.com/pypi/web/simple --trusted-host mirror-sng.oa.com -r /data/user/requirements.txt
RUN pip install detectron2-0.1.3+cu100-cp36-cp36m-linux_x86_64.whl
RUN python setup.py install
