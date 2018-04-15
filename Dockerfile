FROM pccl/pytorch as builder 
WORKDIR /opt/dcnn
COPY requirements.txt ./
RUN pip install -r requirements.txt
RUN python -m spacy download en

FROM pccl/pytorch
COPY --from=builder /root/.cache /root/.cache
COPY --from=builder /opt/conda/lib/python3.6/site-packages/en_core_web_sm /opt/conda/lib/python3.6/site-packages/en_core_web_sm
COPY --from=builder /opt/conda/lib/python3.6/site-packages/spacy/data/en /opt/conda/lib/python3.6/site-packages/spacy/data/en
COPY ./ ./
RUN pip install -r requirements.txt && rm -rf /root/.cache
CMD experiments/./best.sh
