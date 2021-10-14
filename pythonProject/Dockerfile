FROM nvidia/cuda:11.4.0-runtime-ubuntu20.04
RUN echo "Copying data"
RUN mkdir /SNAPPcapAnalysis
COPY requirements.txt /SNAPPcapAnalysis
COPY LossPredictionLoss.py /SNAPPcapAnalysis
COPY NNdataMlabs.pickle /SNAPPcapAnalysis
RUN echo "Installing requirements"
RUN apt-get update && apt-get install -y python3 python3-dev build-essential python3-pip
RUN pip3 install -r /SNAPPcapAnalysis/requirements.txt
RUN echo "Finished setup"
RUN cd /SNAPPcapAnalysis
CMD ["python3", "/SNAPPcapAnalysis/LossPrediction.py"]