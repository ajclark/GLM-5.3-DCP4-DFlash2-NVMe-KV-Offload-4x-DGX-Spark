ARG BASE_IMAGE=spark-vllm:0.29.0-dcp1
FROM ${BASE_IMAGE}
COPY spark_nvme /usr/local/lib/python3.12/dist-packages/spark_nvme
# Register without resolving or changing any pinned serving dependency.
RUN mkdir -p /usr/local/lib/python3.12/dist-packages/spark_nvme_loader-0.1.0.dist-info && \
    printf '[vllm.general_plugins]\nspark_nvme = spark_nvme:register\n' > /usr/local/lib/python3.12/dist-packages/spark_nvme_loader-0.1.0.dist-info/entry_points.txt && \
    printf 'Metadata-Version: 2.1\nName: spark-nvme-loader\nVersion: 0.1.0\n' > /usr/local/lib/python3.12/dist-packages/spark_nvme_loader-0.1.0.dist-info/METADATA && \
    python3 -m compileall -q /usr/local/lib/python3.12/dist-packages/spark_nvme
